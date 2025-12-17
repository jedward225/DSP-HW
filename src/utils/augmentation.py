"""
Audio augmentation utilities for robustness testing.

Provides functions for:
- Adding noise at various SNR levels
- Volume scaling
- Speed perturbation
- Pitch shifting

Note on Architecture Convention:
    This module is an exception to the "only dsp_core imports librosa/scipy" rule.
    Augmentation functions may import scipy.signal directly for efficiency and
    pitch_shift imports from dsp_core for consistency. This exception is documented
    because augmentation is a utility module, not part of the core feature extraction
    pipeline where the convention is most important.
"""

import numpy as np
from typing import Optional, Tuple


def add_noise(
    y: np.ndarray,
    snr_db: float,
    noise_type: str = 'gaussian',
    clip: bool = True,
    clip_range: tuple = (-1.0, 1.0)
) -> np.ndarray:
    """
    Add noise to audio signal at specified SNR.

    Args:
        y: Audio signal
        snr_db: Target signal-to-noise ratio in dB
        noise_type: Type of noise ('gaussian', 'uniform')
        clip: If True, clip output to clip_range to prevent distortion
        clip_range: Min/max values for clipping (default: [-1, 1])

    Returns:
        Noisy audio signal
    """
    # Calculate signal power
    signal_power = np.mean(y ** 2)

    # Calculate noise power based on SNR
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate noise
    if noise_type == 'gaussian':
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    elif noise_type == 'uniform':
        # Uniform noise with same power
        noise_std = np.sqrt(noise_power)
        noise = np.random.uniform(-np.sqrt(3) * noise_std, np.sqrt(3) * noise_std, len(y))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    result = y + noise

    if clip:
        result = np.clip(result, clip_range[0], clip_range[1])

    return result


def scale_volume(
    y: np.ndarray,
    gain_db: float,
    clip: bool = True,
    clip_range: tuple = (-1.0, 1.0)
) -> np.ndarray:
    """
    Scale audio volume by specified gain.

    Args:
        y: Audio signal
        gain_db: Gain in decibels (positive = louder, negative = quieter)
        clip: If True, clip output to clip_range to prevent distortion
        clip_range: Min/max values for clipping (default: [-1, 1])

    Returns:
        Volume-scaled audio signal
    """
    # Convert dB to linear scale
    # gain_linear = 10^(gain_db/20)
    gain_linear = 10 ** (gain_db / 20)
    result = y * gain_linear

    if clip:
        result = np.clip(result, clip_range[0], clip_range[1])

    return result


def time_shift(
    y: np.ndarray,
    shift_samples: int = None,
    shift_fraction: float = None,
    wrap: bool = True
) -> np.ndarray:
    """
    Shift audio in time (circular or zero-pad).

    Args:
        y: Audio signal
        shift_samples: Number of samples to shift (positive = right)
        shift_fraction: Fraction of signal length to shift (alternative to shift_samples)
        wrap: If True, wrap around (circular shift). If False, zero-pad.

    Returns:
        Time-shifted audio signal
    """
    if shift_samples is None and shift_fraction is None:
        raise ValueError("Provide either shift_samples or shift_fraction")

    if shift_fraction is not None:
        shift_samples = int(len(y) * shift_fraction)

    if wrap:
        return np.roll(y, shift_samples)
    else:
        result = np.zeros_like(y)
        if shift_samples > 0:
            result[shift_samples:] = y[:-shift_samples]
        elif shift_samples < 0:
            result[:shift_samples] = y[-shift_samples:]
        else:
            result = y.copy()
        return result


def change_speed(
    y: np.ndarray,
    rate: float,
    sr: int
) -> np.ndarray:
    """
    Change playback speed using time stretching.

    Args:
        y: Audio signal
        rate: Speed factor (< 1 = slower, > 1 = faster)
        sr: Sample rate

    Returns:
        Speed-changed audio signal (length will differ)
    """
    # Import librosa only in dsp_core, so use scipy here
    from scipy.signal import resample

    # Time stretch via resampling
    # rate > 1: faster playback = fewer samples
    # rate < 1: slower playback = more samples
    target_length = int(len(y) / rate)
    return resample(y, target_length)


def pitch_shift(
    y: np.ndarray,
    semitones: float,
    sr: int
) -> np.ndarray:
    """
    Shift pitch by specified semitones.

    Uses dsp_core wrapper (which calls librosa).

    Args:
        y: Audio signal
        semitones: Number of semitones to shift (positive = up, negative = down)
        sr: Sample rate

    Returns:
        Pitch-shifted audio signal
    """
    # Import from dsp_core
    from src.dsp_core import pitch_shift as dsp_pitch_shift
    return dsp_pitch_shift(y, sr=sr, n_steps=semitones)


def random_crop(
    y: np.ndarray,
    crop_length: int,
    sr: int = None
) -> Tuple[np.ndarray, int]:
    """
    Randomly crop a segment from audio.

    Args:
        y: Audio signal
        crop_length: Length of crop in samples
        sr: Sample rate (for returning time info)

    Returns:
        Tuple of (cropped audio, start sample index)
    """
    if crop_length >= len(y):
        return y, 0

    max_start = len(y) - crop_length
    start = np.random.randint(0, max_start + 1)
    return y[start:start + crop_length], start


def pad_or_trim(
    y: np.ndarray,
    target_length: int,
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad or trim audio to target length.

    Args:
        y: Audio signal
        target_length: Target length in samples
        pad_mode: Padding mode ('constant', 'wrap', 'edge')
        pad_value: Value for constant padding

    Returns:
        Audio of exactly target_length samples
    """
    if len(y) == target_length:
        return y
    elif len(y) > target_length:
        return y[:target_length]
    else:
        pad_amount = target_length - len(y)
        if pad_mode == 'constant':
            return np.pad(y, (0, pad_amount), mode='constant', constant_values=pad_value)
        else:
            return np.pad(y, (0, pad_amount), mode=pad_mode)


class AudioAugmenter:
    """
    Class for applying audio augmentations.

    Provides a consistent interface for augmentation pipelines.
    """

    def __init__(self, sr: int = 22050):
        """Initialize augmenter with sample rate."""
        self.sr = sr

    def add_noise(self, y: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Gaussian noise at specified SNR."""
        return add_noise(y, snr_db, noise_type='gaussian')

    def scale_volume(self, y: np.ndarray, gain_db: float) -> np.ndarray:
        """Scale volume by gain in dB."""
        return scale_volume(y, gain_db)

    def change_speed(self, y: np.ndarray, rate: float) -> np.ndarray:
        """Change playback speed."""
        return change_speed(y, rate, self.sr)

    def pitch_shift(self, y: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by semitones."""
        return pitch_shift(y, semitones, self.sr)

    def time_shift(self, y: np.ndarray, shift_fraction: float = 0.1) -> np.ndarray:
        """Shift audio in time."""
        return time_shift(y, shift_fraction=shift_fraction, wrap=True)

    def random_crop(self, y: np.ndarray, duration_s: float) -> np.ndarray:
        """Random crop of specified duration."""
        crop_length = int(duration_s * self.sr)
        cropped, _ = random_crop(y, crop_length)
        return cropped
