"""
Feature extraction module for audio classification.
Wraps the DSP core implementations (FFT/STFT/MFCC) for convenient use.
"""

import numpy as np
from typing import Optional, Literal, Tuple
import os
import pickle
from pathlib import Path

# Import our DSP implementations
from ..dsp_core.stft import stft, power_to_db
from ..dsp_core.mfcc import mfcc, mel_filterbank, delta


FeatureType = Literal['stft', 'mel', 'mfcc', 'mfcc_delta']


def extract_stft(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    to_db: bool = True
) -> np.ndarray:
    """
    Extract STFT magnitude spectrogram.

    Returns
    -------
    np.ndarray
        Shape: (n_fft//2+1, n_frames)
    """
    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S)

    if to_db:
        return power_to_db(magnitude ** 2, ref=1.0, amin=1e-10, top_db=80.0)
    return magnitude


def extract_mel(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract log-Mel spectrogram.

    Returns
    -------
    np.ndarray
        Shape: (n_mels, n_frames)
    """
    if fmax is None:
        fmax = sr / 2.0

    # Compute STFT
    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spec = np.abs(S) ** 2

    # Apply Mel filterbank
    mel_basis = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_basis, power_spec)

    # Convert to dB
    return power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)


def extract_mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 40,
    n_mels: int = 128,
    with_delta: bool = False,
    delta_width: int = 9
) -> np.ndarray:
    """
    Extract MFCC features, optionally with delta and delta-delta.

    Parameters
    ----------
    with_delta : bool
        If True, append delta and delta-delta features.
        Output shape becomes (n_mfcc * 3, n_frames)

    Returns
    -------
    np.ndarray
        Shape: (n_mfcc, n_frames) or (n_mfcc * 3, n_frames) if with_delta
    """
    mfccs = mfcc(
        y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    if with_delta:
        delta1 = delta(mfccs, width=delta_width)
        delta2 = delta(mfccs, width=delta_width, order=2)
        return np.vstack([mfccs, delta1, delta2])

    return mfccs


def extract_features(
    y: np.ndarray,
    sr: int = 22050,
    feature_type: FeatureType = 'mel',
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 40,
    **kwargs
) -> np.ndarray:
    """
    Unified feature extraction interface.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    feature_type : str
        One of 'stft', 'mel', 'mfcc', 'mfcc_delta'
    n_fft : int
        FFT window size
    hop_length : int
        Hop length between frames
    n_mels : int
        Number of Mel bands
    n_mfcc : int
        Number of MFCCs

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_features, n_frames)
    """
    if feature_type == 'stft':
        return extract_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    elif feature_type == 'mel':
        return extract_mel(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    elif feature_type == 'mfcc':
        return extract_mfcc(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                           n_mfcc=n_mfcc, n_mels=n_mels, with_delta=False)
    elif feature_type == 'mfcc_delta':
        return extract_mfcc(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                           n_mfcc=n_mfcc, n_mels=n_mels, with_delta=True)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def pad_or_truncate(features: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Pad or truncate features to a fixed number of frames.

    Parameters
    ----------
    features : np.ndarray
        Shape: (n_features, n_frames)
    target_frames : int
        Target number of frames

    Returns
    -------
    np.ndarray
        Shape: (n_features, target_frames)
    """
    n_features, n_frames = features.shape

    if n_frames < target_frames:
        # Pad with zeros on the right
        pad_width = target_frames - n_frames
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    elif n_frames > target_frames:
        # Truncate
        features = features[:, :target_frames]

    return features


def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize features.

    Parameters
    ----------
    features : np.ndarray
        Shape: (n_features, n_frames)
    method : str
        'standard' for zero mean, unit variance
        'minmax' for [0, 1] scaling
    """
    if method == 'standard':
        mean = features.mean()
        std = features.std()
        if std > 0:
            return (features - mean) / std
        return features - mean
    elif method == 'minmax':
        fmin, fmax = features.min(), features.max()
        if fmax > fmin:
            return (features - fmin) / (fmax - fmin)
        return features - fmin
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class FeatureCache:
    """
    Cache extracted features to disk to avoid recomputation.
    """
    def __init__(self, cache_dir: str = '.feature_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        audio_path: str,
        feature_type: str,
        n_fft: int,
        hop_length: int,
        **kwargs
    ) -> str:
        """Generate a unique cache key for the feature configuration."""
        basename = Path(audio_path).stem
        key_parts = [basename, feature_type, f"nfft{n_fft}", f"hop{hop_length}"]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}{v}")
        return "_".join(key_parts) + ".pkl"

    def get(self, audio_path: str, feature_type: str, n_fft: int,
            hop_length: int, **kwargs) -> Optional[np.ndarray]:
        """Retrieve cached features if available."""
        cache_key = self._get_cache_key(audio_path, feature_type, n_fft, hop_length, **kwargs)
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def put(self, audio_path: str, feature_type: str, n_fft: int,
            hop_length: int, features: np.ndarray, **kwargs):
        """Cache extracted features."""
        cache_key = self._get_cache_key(audio_path, feature_type, n_fft, hop_length, **kwargs)
        cache_path = self.cache_dir / cache_key

        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

    def clear(self):
        """Clear all cached features."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test feature extraction
    import soundfile as sf

    print("=" * 70)
    print("Feature Extraction Test")
    print("=" * 70)

    # Generate a test signal
    sr = 22050
    duration = 5.0
    y = np.random.randn(int(sr * duration))

    for feat_type in ['stft', 'mel', 'mfcc', 'mfcc_delta']:
        features = extract_features(y, sr=sr, feature_type=feat_type)
        print(f"{feat_type:12s} -> shape: {features.shape}")

    print("\nPad/Truncate test:")
    mel = extract_features(y, sr=sr, feature_type='mel')
    print(f"Original: {mel.shape}")
    print(f"Padded to 300: {pad_or_truncate(mel, 300).shape}")
    print(f"Truncated to 100: {pad_or_truncate(mel, 100).shape}")
