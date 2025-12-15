"""
Feature Extraction Module for Sound Retrieval

This module provides a unified interface for extracting various audio features
used in the retrieval experiments.

Supported Features:
    - MFCC: Mel-Frequency Cepstral Coefficients (using our hand-written DSP core)
    - MFCC + Delta: MFCC with first and second derivatives
    - Mel-Spectrogram: Log-Mel spectrogram
    - STFT: Short-Time Fourier Transform magnitude

Usage:
    >>> from src.retrieval.features import extract_features, FeatureExtractor
    >>> features = extract_features(audio, sr=22050, feature_type='mfcc')
"""

import numpy as np
from typing import Optional, Union, Literal, Dict, Any
from tqdm import tqdm


FeatureType = Literal['mfcc', 'mfcc_delta', 'mel_spectrogram', 'stft']


def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    feature_type: FeatureType = 'mfcc',
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    n_mfcc: int = 13,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    delta_width: int = 9,
    **kwargs
) -> np.ndarray:
    """
    Extract audio features using our hand-written DSP implementations.

    Parameters
    ----------
    y : np.ndarray
        Audio signal, shape (n_samples,)
    sr : int
        Sample rate (default: 22050)
    feature_type : str
        Type of features to extract:
        - 'mfcc': MFCC coefficients
        - 'mfcc_delta': MFCC with delta and delta-delta
        - 'mel_spectrogram': Log-Mel spectrogram
        - 'stft': STFT magnitude spectrum
    n_fft : int
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length between frames (default: n_fft // 4)
    n_mfcc : int
        Number of MFCC coefficients (default: 13)
    n_mels : int
        Number of Mel bands (default: 128)
    fmin : float
        Minimum frequency for Mel filterbank (default: 0.0)
    fmax : float, optional
        Maximum frequency for Mel filterbank (default: sr / 2)
    delta_width : int
        Width for delta computation (default: 9)
    **kwargs
        Additional arguments passed to feature extractors

    Returns
    -------
    np.ndarray
        Extracted features:
        - 'mfcc': shape (n_mfcc, n_frames)
        - 'mfcc_delta': shape (n_mfcc * 3, n_frames)
        - 'mel_spectrogram': shape (n_mels, n_frames)
        - 'stft': shape (n_fft // 2 + 1, n_frames)
    """
    from src.dsp_core import mfcc as mfcc_fn, stft, delta
    from src.dsp_core.mfcc import mel_filterbank
    from src.dsp_core.stft import power_to_db

    if hop_length is None:
        hop_length = n_fft // 4
    if fmax is None:
        fmax = sr / 2.0

    y = np.asarray(y, dtype=np.float64)

    if feature_type == 'mfcc':
        # Extract MFCC using our implementation
        features = mfcc_fn(
            y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

    elif feature_type == 'mfcc_delta':
        # Extract MFCC + delta + delta-delta
        mfcc_features = mfcc_fn(
            y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        delta_features = delta(mfcc_features, width=delta_width, order=1)
        delta2_features = delta(mfcc_features, width=delta_width, order=2)

        # Stack: (n_mfcc * 3, n_frames)
        features = np.vstack([mfcc_features, delta_features, delta2_features])

    elif feature_type == 'mel_spectrogram':
        # Compute log-Mel spectrogram
        S = stft(y, n_fft=n_fft, hop_length=hop_length)
        power_spec = np.abs(S) ** 2

        mel_basis = mel_filterbank(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        mel_spec = np.dot(mel_basis, power_spec)
        features = power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

    elif feature_type == 'stft':
        # Compute STFT magnitude
        S = stft(y, n_fft=n_fft, hop_length=hop_length)
        features = np.abs(S)

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return features


class FeatureExtractor:
    """
    Feature extractor with consistent configuration for batch processing.

    This class maintains feature extraction parameters and provides
    methods for extracting features from single audio or batches.

    Parameters
    ----------
    feature_type : str
        Type of features to extract
    sr : int
        Target sample rate
    n_fft : int
        FFT window size
    hop_length : int, optional
        Hop length between frames
    n_mfcc : int
        Number of MFCC coefficients
    n_mels : int
        Number of Mel bands
    **kwargs
        Additional parameters

    Examples
    --------
    >>> extractor = FeatureExtractor(feature_type='mfcc', sr=22050)
    >>> features = extractor.extract(audio)
    >>> batch_features = extractor.extract_batch(audio_list)
    """

    def __init__(
        self,
        feature_type: FeatureType = 'mfcc',
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        n_mfcc: int = 13,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        delta_width: int = 9,
        **kwargs
    ):
        self.feature_type = feature_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2.0
        self.delta_width = delta_width
        self.kwargs = kwargs

    def extract(self, y: np.ndarray) -> np.ndarray:
        """Extract features from a single audio signal."""
        return extract_features(
            y,
            sr=self.sr,
            feature_type=self.feature_type,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            delta_width=self.delta_width,
            **self.kwargs
        )

    def extract_batch(
        self,
        audio_list: list,
        show_progress: bool = True
    ) -> list:
        """
        Extract features from a batch of audio signals.

        Parameters
        ----------
        audio_list : list
            List of audio signals (each as np.ndarray)
        show_progress : bool
            Show progress bar

        Returns
        -------
        list
            List of feature arrays
        """
        features_list = []

        iterator = tqdm(audio_list, desc="Extracting features") if show_progress else audio_list

        for audio in iterator:
            features = self.extract(audio)
            features_list.append(features)

        return features_list

    def get_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return {
            'feature_type': self.feature_type,
            'sr': self.sr,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mfcc': self.n_mfcc,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'delta_width': self.delta_width
        }

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor(feature_type='{self.feature_type}', "
            f"sr={self.sr}, n_fft={self.n_fft}, hop_length={self.hop_length})"
        )


def extract_features_from_dataset(
    dataset,
    feature_type: FeatureType = 'mfcc',
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    n_mfcc: int = 13,
    n_mels: int = 128,
    show_progress: bool = True,
    **kwargs
) -> tuple:
    """
    Extract features from an ESC50Dataset.

    This function loads audio from the dataset and extracts features
    using the specified configuration.

    Parameters
    ----------
    dataset : ESC50Dataset
        Dataset to extract features from
    feature_type : str
        Type of features to extract
    sr : int
        Target sample rate
    n_fft : int
        FFT window size
    hop_length : int, optional
        Hop length
    n_mfcc : int
        Number of MFCC coefficients
    n_mels : int
        Number of Mel bands
    show_progress : bool
        Show progress bar
    **kwargs
        Additional arguments

    Returns
    -------
    features : np.ndarray
        Extracted features, shape (n_samples, n_features, n_frames)
    labels : np.ndarray
        Labels for each sample, shape (n_samples,)
    filenames : list
        List of filenames
    """
    import librosa

    extractor = FeatureExtractor(
        feature_type=feature_type,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        **kwargs
    )

    features_list = []
    labels_list = []
    filenames_list = []

    iterator = range(len(dataset))
    if show_progress:
        iterator = tqdm(iterator, desc=f"Extracting {feature_type} features")

    for idx in iterator:
        # Get raw waveform from dataset
        # The dataset returns (features, label, filename)
        # We need raw audio, so we'll load it directly
        row = dataset.metadata.iloc[idx]
        filename = row['filename']
        label = row['target']

        # Load audio
        import os
        audio_path = os.path.join(dataset.root, 'audio', filename)
        y, _ = librosa.load(audio_path, sr=sr)

        # Extract features
        features = extractor.extract(y)

        features_list.append(features)
        labels_list.append(label)
        filenames_list.append(filename)

    # Convert to numpy arrays
    # Note: features may have different n_frames due to variable audio lengths
    # For ESC-50, all clips are 5 seconds, so n_frames should be consistent
    labels = np.array(labels_list)

    # Stack features (assuming consistent shape)
    try:
        features = np.stack(features_list, axis=0)
    except ValueError:
        # Variable length features - return as list
        features = features_list

    return features, labels, filenames_list


def aggregate_features(
    features: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate frame-level features to clip-level features.

    Parameters
    ----------
    features : np.ndarray
        Frame-level features, shape (n_features, n_frames) or (n_samples, n_features, n_frames)
    method : str
        Aggregation method:
        - 'mean': Mean across frames
        - 'max': Max across frames
        - 'std': Standard deviation across frames
        - 'mean_std': Concatenate mean and std

    Returns
    -------
    np.ndarray
        Aggregated features:
        - 'mean'/'max'/'std': shape (n_features,) or (n_samples, n_features)
        - 'mean_std': shape (n_features * 2,) or (n_samples, n_features * 2)
    """
    if features.ndim == 2:
        # Single sample: (n_features, n_frames)
        if method == 'mean':
            return np.mean(features, axis=1)
        elif method == 'max':
            return np.max(features, axis=1)
        elif method == 'std':
            return np.std(features, axis=1)
        elif method == 'mean_std':
            mean_feat = np.mean(features, axis=1)
            std_feat = np.std(features, axis=1)
            return np.concatenate([mean_feat, std_feat])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    elif features.ndim == 3:
        # Batch: (n_samples, n_features, n_frames)
        if method == 'mean':
            return np.mean(features, axis=2)
        elif method == 'max':
            return np.max(features, axis=2)
        elif method == 'std':
            return np.std(features, axis=2)
        elif method == 'mean_std':
            mean_feat = np.mean(features, axis=2)
            std_feat = np.std(features, axis=2)
            return np.concatenate([mean_feat, std_feat], axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {features.shape}")


if __name__ == "__main__":
    print("=" * 70)
    print("Feature Extraction Module - Unit Tests")
    print("=" * 70)

    # Generate test audio
    np.random.seed(42)
    sr = 22050
    duration = 5.0  # 5 seconds like ESC-50
    y = np.random.randn(int(sr * duration))

    # Test 1: MFCC
    print("\n[Test 1] MFCC Extraction")
    mfcc_features = extract_features(y, sr=sr, feature_type='mfcc', n_mfcc=13)
    print(f"  Input shape: {y.shape}")
    print(f"  MFCC shape: {mfcc_features.shape}")
    expected_frames = 1 + (len(y) + 2048 // 2 * 2 - 2048) // 512
    print(f"  Expected ~{expected_frames} frames")
    print(f"  ✓ PASS" if mfcc_features.shape[0] == 13 else "  ✗ FAIL")

    # Test 2: MFCC + Delta
    print("\n[Test 2] MFCC + Delta Extraction")
    mfcc_delta = extract_features(y, sr=sr, feature_type='mfcc_delta', n_mfcc=13)
    print(f"  MFCC + Delta shape: {mfcc_delta.shape}")
    print(f"  ✓ PASS" if mfcc_delta.shape[0] == 39 else "  ✗ FAIL")  # 13 * 3

    # Test 3: Mel Spectrogram
    print("\n[Test 3] Mel Spectrogram Extraction")
    mel_spec = extract_features(y, sr=sr, feature_type='mel_spectrogram', n_mels=128)
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    print(f"  ✓ PASS" if mel_spec.shape[0] == 128 else "  ✗ FAIL")

    # Test 4: STFT
    print("\n[Test 4] STFT Extraction")
    stft_features = extract_features(y, sr=sr, feature_type='stft', n_fft=2048)
    print(f"  STFT shape: {stft_features.shape}")
    print(f"  ✓ PASS" if stft_features.shape[0] == 1025 else "  ✗ FAIL")  # 2048 // 2 + 1

    # Test 5: Feature Extractor class
    print("\n[Test 5] FeatureExtractor Class")
    extractor = FeatureExtractor(feature_type='mfcc', sr=22050, n_mfcc=20)
    features = extractor.extract(y)
    print(f"  Extractor: {extractor}")
    print(f"  Features shape: {features.shape}")
    print(f"  Config: {extractor.get_config()}")
    print(f"  ✓ PASS" if features.shape[0] == 20 else "  ✗ FAIL")

    # Test 6: Feature aggregation
    print("\n[Test 6] Feature Aggregation")
    for method in ['mean', 'max', 'std', 'mean_std']:
        agg = aggregate_features(mfcc_features, method=method)
        expected_dim = 13 * 2 if method == 'mean_std' else 13
        status = "✓" if agg.shape[0] == expected_dim else "✗"
        print(f"  {method:>8s}: shape={agg.shape} {status}")

    # Test 7: Batch aggregation
    print("\n[Test 7] Batch Feature Aggregation")
    batch_features = np.random.randn(10, 13, 44)  # 10 samples
    batch_agg = aggregate_features(batch_features, method='mean')
    print(f"  Input: {batch_features.shape}")
    print(f"  Output: {batch_agg.shape}")
    print(f"  ✓ PASS" if batch_agg.shape == (10, 13) else "  ✗ FAIL")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
