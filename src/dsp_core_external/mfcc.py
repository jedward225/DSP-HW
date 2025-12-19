"""
MFCC wrapper module - provides MFCC functions using librosa.
Only this module (dsp_core) is allowed to import librosa.
"""

import numpy as np
import librosa
import scipy.fftpack


def mfcc(
    y: np.ndarray = None,
    sr: int = 22050,
    S: np.ndarray = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: str = 'ortho',
    lifter: int = 0,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'constant',
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    power: float = 2.0,
    htk: bool = False,
) -> np.ndarray:
    """
    Mel-frequency cepstral coefficients (MFCCs).

    Args:
        y: Audio time series
        sr: Sampling rate of y
        S: Pre-computed mel spectrogram (optional, if provided y is ignored)
        n_mfcc: Number of MFCCs to return
        dct_type: DCT type (1, 2, or 3)
        norm: DCT normalization ('ortho' or None)
        lifter: Liftering parameter (0 = no liftering)
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Window length (default: n_fft)
        window: Window function type
        center: If True, frames are centered
        pad_mode: Padding mode
        n_mels: Number of Mel bands
        fmin: Minimum frequency for Mel filterbank
        fmax: Maximum frequency for Mel filterbank (default: sr/2)
        power: Exponent for the magnitude spectrogram
        htk: Use HTK formula for Mel scale (default: Slaney)

    Returns:
        MFCC sequence, shape (n_mfcc, n_frames)
    """
    if win_length is None:
        win_length = n_fft

    # Handle rectangular window
    if window == 'rectangular' or window == 'rect':
        window = 'boxcar'

    # If htk=True, compute mel spectrogram with HTK formula and apply DCT directly
    # to avoid double mel filtering that occurs when passing S to librosa.mfcc()
    if htk and S is None:
        # Compute STFT
        stft = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode
        )
        # Power spectrogram
        S_power = np.abs(stft) ** power
        # Create HTK mel filterbank
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=True,
            norm='slaney'
        )
        # Apply mel filterbank
        S_mel = np.dot(mel_basis, S_power)

        # Convert to log scale (dB)
        S_log = librosa.power_to_db(S_mel, ref=np.max)

        # Apply DCT directly to get MFCCs
        # DCT operates along frequency axis (axis=0 for shape n_mels x n_frames)
        dct_norm = 'ortho' if norm == 'ortho' else None
        mfccs = scipy.fftpack.dct(S_log, axis=0, type=dct_type, norm=dct_norm)[:n_mfcc]

        # Apply liftering if specified
        if lifter > 0:
            mfccs = lifter_cepstrum(mfccs, lifter)

        return mfccs

    return librosa.feature.mfcc(
        y=y if S is None else None,
        sr=sr,
        S=S,
        n_mfcc=n_mfcc,
        dct_type=dct_type,
        norm=norm,
        lifter=lifter,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )


def delta(
    data: np.ndarray,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = 'interp'
) -> np.ndarray:
    """
    Compute delta features (local estimate of the derivative).

    Args:
        data: Input data matrix (e.g., MFCC)
        width: Number of frames over which to compute the delta
        order: Order of the derivative (1 for delta, 2 for delta-delta)
        axis: Axis along which to compute deltas
        mode: Padding mode ('interp', 'constant', 'nearest', 'mirror', 'wrap')

    Returns:
        Delta features with same shape as input
    """
    return librosa.feature.delta(data, width=width, order=order, axis=axis, mode=mode)


def pre_emphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio signal.

    Pre-emphasis boosts high frequencies to balance the spectrum.

    Args:
        y: Audio time series
        coef: Pre-emphasis coefficient (typically 0.95-0.97)

    Returns:
        Pre-emphasized audio signal
    """
    return np.append(y[0], y[1:] - coef * y[:-1])


def lifter_cepstrum(
    cepstrum: np.ndarray,
    lifter: int = 22,
) -> np.ndarray:
    """
    Apply cepstral liftering to emphasize higher-order cepstral coefficients.

    Liftering applies a sinusoidal window to the cepstrum:
        L(n) = 1 + (lifter/2) * sin(π * n / lifter)

    This helps to reduce the dynamic range of the cepstral coefficients
    and can improve speaker verification and retrieval performance.

    Args:
        cepstrum: Cepstral coefficients, shape (n_ceps, n_frames)
        lifter: Liftering parameter (0 = no liftering, typical: 22)

    Returns:
        Liftered cepstral coefficients with same shape
    """
    if lifter == 0:
        return cepstrum

    n_ceps = cepstrum.shape[0]

    # Compute liftering window
    # L(n) = 1 + (lifter/2) * sin(π * n / lifter)
    n = np.arange(n_ceps)
    lift_window = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)

    # Apply liftering
    return cepstrum * lift_window[:, np.newaxis]


def cmvn(
    features: np.ndarray,
    axis: int = -1,
    variance_normalization: bool = True
) -> np.ndarray:
    """
    Cepstral mean and variance normalization (CMVN).

    Args:
        features: Input feature matrix
        axis: Axis along which to compute statistics
        variance_normalization: If True, also normalize variance

    Returns:
        Normalized features
    """
    mean = np.mean(features, axis=axis, keepdims=True)
    features = features - mean

    if variance_normalization:
        std = np.std(features, axis=axis, keepdims=True)
        std = np.maximum(std, 1e-10)  # Avoid division by zero
        features = features / std

    return features
