"""
Data augmentation module for audio classification.
Implements SpecAugment (Time Mask + Frequency Mask) and same-label interpolation.
"""

import numpy as np
from typing import Optional, Tuple, List
import random


def time_mask(
    spectrogram: np.ndarray,
    T: int = 40,
    num_masks: int = 1,
    replace_value: float = 0.0
) -> np.ndarray:
    """
    Apply time masking to spectrogram.

    Randomly masks consecutive time frames with a fixed value.
    Part of SpecAugment: https://arxiv.org/abs/1904.08779

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram of shape (n_features, n_frames)
    T : int
        Maximum width of each time mask
    num_masks : int
        Number of time masks to apply
    replace_value : float
        Value to fill masked regions (default 0.0)

    Returns
    -------
    np.ndarray
        Augmented spectrogram with same shape
    """
    spec = spectrogram.copy()
    n_features, n_frames = spec.shape

    for _ in range(num_masks):
        # Random mask width
        t = np.random.randint(0, min(T, n_frames) + 1)
        if t == 0:
            continue

        # Random starting position
        t0 = np.random.randint(0, max(1, n_frames - t + 1))

        # Apply mask
        spec[:, t0:t0 + t] = replace_value

    return spec


def frequency_mask(
    spectrogram: np.ndarray,
    F: int = 27,
    num_masks: int = 1,
    replace_value: float = 0.0
) -> np.ndarray:
    """
    Apply frequency masking to spectrogram.

    Randomly masks consecutive frequency bins with a fixed value.
    Part of SpecAugment: https://arxiv.org/abs/1904.08779

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram of shape (n_features, n_frames)
    F : int
        Maximum width of each frequency mask
    num_masks : int
        Number of frequency masks to apply
    replace_value : float
        Value to fill masked regions (default 0.0)

    Returns
    -------
    np.ndarray
        Augmented spectrogram with same shape
    """
    spec = spectrogram.copy()
    n_features, n_frames = spec.shape

    for _ in range(num_masks):
        # Random mask width
        f = np.random.randint(0, min(F, n_features) + 1)
        if f == 0:
            continue

        # Random starting position
        f0 = np.random.randint(0, max(1, n_features - f + 1))

        # Apply mask
        spec[f0:f0 + f, :] = replace_value

    return spec


def spec_augment(
    spectrogram: np.ndarray,
    time_mask_param: int = 40,
    freq_mask_param: int = 27,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    replace_value: float = 0.0
) -> np.ndarray:
    """
    Apply SpecAugment to spectrogram.

    Combines time masking and frequency masking.
    Reference: https://arxiv.org/abs/1904.08779

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram of shape (n_features, n_frames)
    time_mask_param : int
        Maximum time mask width (T)
    freq_mask_param : int
        Maximum frequency mask width (F)
    num_time_masks : int
        Number of time masks
    num_freq_masks : int
        Number of frequency masks
    replace_value : float
        Value to fill masked regions

    Returns
    -------
    np.ndarray
        Augmented spectrogram
    """
    # Apply time masking
    spec = time_mask(
        spectrogram,
        T=time_mask_param,
        num_masks=num_time_masks,
        replace_value=replace_value
    )

    # Apply frequency masking
    spec = frequency_mask(
        spec,
        F=freq_mask_param,
        num_masks=num_freq_masks,
        replace_value=replace_value
    )

    return spec


def same_label_mixup(
    spec1: np.ndarray,
    spec2: np.ndarray,
    alpha_range: Tuple[float, float] = (0.3, 0.7)
) -> np.ndarray:
    """
    Mix two spectrograms of the same label using linear interpolation.

    Exploits STFT linearity: STFT(α*x1 + β*x2) = α*STFT(x1) + β*STFT(x2)

    Reference: [Ref-1] 张鑫恺等团队 - 利用STFT线性性进行数据增强

    Parameters
    ----------
    spec1 : np.ndarray
        First spectrogram of shape (n_features, n_frames)
    spec2 : np.ndarray
        Second spectrogram of shape (n_features, n_frames)
        Must have same shape as spec1
    alpha_range : Tuple[float, float]
        Range for mixing coefficient (min, max)

    Returns
    -------
    np.ndarray
        Mixed spectrogram

    Notes
    -----
    Both spectrograms should be from the same class.
    The mixing is done in spectral domain to preserve linearity.
    """
    if spec1.shape != spec2.shape:
        raise ValueError(f"Shape mismatch: {spec1.shape} vs {spec2.shape}")

    # Random mixing coefficient
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    # Linear interpolation
    mixed = alpha * spec1 + (1 - alpha) * spec2

    return mixed


def time_warp(
    spectrogram: np.ndarray,
    W: int = 80
) -> np.ndarray:
    """
    Apply time warping to spectrogram (simplified version).

    Part of SpecAugment but often omitted due to complexity.
    This is a simplified random shift version.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram of shape (n_features, n_frames)
    W : int
        Maximum warp distance

    Returns
    -------
    np.ndarray
        Time-warped spectrogram
    """
    spec = spectrogram.copy()
    n_features, n_frames = spec.shape

    if n_frames < W * 2:
        return spec

    # Random warp point in the middle region
    center = n_frames // 2
    warp_point = np.random.randint(W, n_frames - W)

    # Random warp distance
    distance = np.random.randint(-W, W + 1)

    if distance == 0:
        return spec

    # Create warped spectrogram using interpolation
    # Simplified: just shift a portion
    new_spec = np.zeros_like(spec)

    if distance > 0:
        # Stretch left part, compress right part
        left_indices = np.linspace(0, warp_point, warp_point + distance, dtype=int)
        right_indices = np.linspace(warp_point, n_frames - 1, n_frames - warp_point - distance, dtype=int)

        for i, idx in enumerate(left_indices):
            if i < n_frames and idx < n_frames:
                new_spec[:, i] = spec[:, min(idx, n_frames - 1)]

        for i, idx in enumerate(right_indices):
            new_idx = warp_point + distance + i
            if new_idx < n_frames and idx < n_frames:
                new_spec[:, new_idx] = spec[:, idx]
    else:
        # Compress left part, stretch right part
        distance = abs(distance)
        left_indices = np.linspace(0, warp_point, warp_point - distance, dtype=int)
        right_indices = np.linspace(warp_point, n_frames - 1, n_frames - warp_point + distance, dtype=int)

        for i, idx in enumerate(left_indices):
            if i < n_frames and idx < n_frames:
                new_spec[:, i] = spec[:, min(idx, n_frames - 1)]

        for i, idx in enumerate(right_indices):
            new_idx = warp_point - distance + i
            if new_idx < n_frames and idx < n_frames:
                new_spec[:, new_idx] = spec[:, idx]

    return new_spec


def random_gain(
    spectrogram: np.ndarray,
    gain_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Apply random gain to spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram
    gain_range : Tuple[float, float]
        Range for random gain factor

    Returns
    -------
    np.ndarray
        Scaled spectrogram
    """
    gain = np.random.uniform(gain_range[0], gain_range[1])
    return spectrogram * gain


def add_noise(
    spectrogram: np.ndarray,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    Add Gaussian noise to spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram
    noise_level : float
        Standard deviation of noise relative to spectrogram std

    Returns
    -------
    np.ndarray
        Noisy spectrogram
    """
    noise = np.random.randn(*spectrogram.shape) * noise_level * spectrogram.std()
    return spectrogram + noise


class AudioAugmenter:
    """
    Configurable audio augmentation pipeline.

    Example
    -------
    >>> augmenter = AudioAugmenter(
    ...     use_time_mask=True,
    ...     use_freq_mask=True,
    ...     use_mixup=False
    ... )
    >>> augmented = augmenter(spectrogram)
    """

    def __init__(
        self,
        use_time_mask: bool = True,
        use_freq_mask: bool = True,
        use_mixup: bool = False,
        use_time_warp: bool = False,
        use_random_gain: bool = False,
        use_noise: bool = False,
        time_mask_param: int = 40,
        freq_mask_param: int = 27,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        mixup_alpha_range: Tuple[float, float] = (0.3, 0.7),
        time_warp_param: int = 80,
        gain_range: Tuple[float, float] = (0.8, 1.2),
        noise_level: float = 0.01,
        p: float = 1.0
    ):
        """
        Initialize augmenter.

        Parameters
        ----------
        use_time_mask : bool
            Whether to use time masking
        use_freq_mask : bool
            Whether to use frequency masking
        use_mixup : bool
            Whether to use same-label mixup (requires pair of spectrograms)
        use_time_warp : bool
            Whether to use time warping
        use_random_gain : bool
            Whether to use random gain
        use_noise : bool
            Whether to add noise
        p : float
            Probability of applying augmentation
        """
        self.use_time_mask = use_time_mask
        self.use_freq_mask = use_freq_mask
        self.use_mixup = use_mixup
        self.use_time_warp = use_time_warp
        self.use_random_gain = use_random_gain
        self.use_noise = use_noise

        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mixup_alpha_range = mixup_alpha_range
        self.time_warp_param = time_warp_param
        self.gain_range = gain_range
        self.noise_level = noise_level
        self.p = p

    def __call__(
        self,
        spectrogram: np.ndarray,
        pair_spectrogram: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply augmentation pipeline.

        Parameters
        ----------
        spectrogram : np.ndarray
            Input spectrogram
        pair_spectrogram : np.ndarray, optional
            Second spectrogram for mixup (must be same label)

        Returns
        -------
        np.ndarray
            Augmented spectrogram
        """
        # Check if we should apply augmentation
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()

        # Apply augmentations in order
        if self.use_time_warp:
            spec = time_warp(spec, W=self.time_warp_param)

        if self.use_time_mask:
            spec = time_mask(
                spec,
                T=self.time_mask_param,
                num_masks=self.num_time_masks
            )

        if self.use_freq_mask:
            spec = frequency_mask(
                spec,
                F=self.freq_mask_param,
                num_masks=self.num_freq_masks
            )

        if self.use_mixup and pair_spectrogram is not None:
            spec = same_label_mixup(
                spec,
                pair_spectrogram,
                alpha_range=self.mixup_alpha_range
            )

        if self.use_random_gain:
            spec = random_gain(spec, gain_range=self.gain_range)

        if self.use_noise:
            spec = add_noise(spec, noise_level=self.noise_level)

        return spec


# Preset configurations
def get_spec_augment_light() -> AudioAugmenter:
    """Light SpecAugment configuration."""
    return AudioAugmenter(
        use_time_mask=True,
        use_freq_mask=True,
        time_mask_param=20,
        freq_mask_param=13,
        num_time_masks=1,
        num_freq_masks=1
    )


def get_spec_augment_medium() -> AudioAugmenter:
    """Medium SpecAugment configuration."""
    return AudioAugmenter(
        use_time_mask=True,
        use_freq_mask=True,
        time_mask_param=40,
        freq_mask_param=27,
        num_time_masks=2,
        num_freq_masks=2
    )


def get_spec_augment_strong() -> AudioAugmenter:
    """Strong SpecAugment configuration with additional augmentations."""
    return AudioAugmenter(
        use_time_mask=True,
        use_freq_mask=True,
        use_random_gain=True,
        use_noise=True,
        time_mask_param=60,
        freq_mask_param=40,
        num_time_masks=3,
        num_freq_masks=3,
        noise_level=0.005
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Data Augmentation Test")
    print("=" * 70)

    # Create a dummy spectrogram
    n_features, n_frames = 128, 200
    spec = np.random.randn(n_features, n_frames)

    print(f"\nOriginal spectrogram shape: {spec.shape}")
    print(f"Original min/max: {spec.min():.3f} / {spec.max():.3f}")

    # Test time masking
    print("\n[Test 1] Time Masking")
    masked = time_mask(spec, T=40, num_masks=2)
    zero_cols = np.sum(np.all(masked == 0, axis=0))
    print(f"  Masked columns: {zero_cols}")

    # Test frequency masking
    print("\n[Test 2] Frequency Masking")
    masked = frequency_mask(spec, F=27, num_masks=2)
    zero_rows = np.sum(np.all(masked == 0, axis=1))
    print(f"  Masked rows: {zero_rows}")

    # Test SpecAugment
    print("\n[Test 3] SpecAugment (combined)")
    augmented = spec_augment(spec, num_time_masks=2, num_freq_masks=2)
    print(f"  Output shape: {augmented.shape}")

    # Test same-label mixup
    print("\n[Test 4] Same-label Mixup")
    spec2 = np.random.randn(n_features, n_frames)
    mixed = same_label_mixup(spec, spec2, alpha_range=(0.3, 0.7))
    print(f"  Output shape: {mixed.shape}")

    # Test AudioAugmenter class
    print("\n[Test 5] AudioAugmenter Pipeline")
    augmenter = get_spec_augment_medium()
    augmented = augmenter(spec)
    print(f"  Output shape: {augmented.shape}")

    # Test with mixup
    print("\n[Test 6] AudioAugmenter with Mixup")
    augmenter_mixup = AudioAugmenter(
        use_time_mask=True,
        use_freq_mask=True,
        use_mixup=True
    )
    augmented = augmenter_mixup(spec, pair_spectrogram=spec2)
    print(f"  Output shape: {augmented.shape}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
