"""
STFT wrapper module - provides STFT functions using librosa.
Only this module (dsp_core) is allowed to import librosa.
"""

import numpy as np
import librosa


def stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = None,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'constant'
) -> np.ndarray:
    """
    Short-time Fourier transform (STFT).

    Args:
        y: Input signal (1-D array)
        n_fft: FFT window size
        hop_length: Number of samples between successive frames (default: n_fft // 4)
        win_length: Window length (default: n_fft)
        window: Window function type ('hann', 'hamming', 'rectangular', etc.)
        center: If True, input signal is padded so frames are centered
        pad_mode: Padding mode for centered frames

    Returns:
        Complex-valued matrix of STFT coefficients, shape (1 + n_fft/2, n_frames)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Handle rectangular window
    if window == 'rectangular' or window == 'rect':
        window = 'boxcar'

    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode
    )


def istft(
    stft_matrix: np.ndarray,
    hop_length: int = None,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    length: int = None
) -> np.ndarray:
    """
    Inverse short-time Fourier transform (ISTFT).

    Args:
        stft_matrix: STFT matrix from stft()
        hop_length: Number of samples between successive frames
        win_length: Window length
        window: Window function type
        center: If True, output is centered
        length: Output signal length

    Returns:
        Reconstructed time-domain signal
    """
    n_fft = 2 * (stft_matrix.shape[0] - 1)

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Handle rectangular window
    if window == 'rectangular' or window == 'rect':
        window = 'boxcar'

    return librosa.istft(
        stft_matrix=stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length
    )


def magphase(D: np.ndarray) -> tuple:
    """
    Separate a complex-valued spectrogram into magnitude and phase.

    Args:
        D: Complex-valued spectrogram

    Returns:
        Tuple of (magnitude, phase)
    """
    return librosa.magphase(D)


def amplitude_to_db(
    S: np.ndarray,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Convert an amplitude spectrogram to dB-scaled spectrogram.

    Args:
        S: Input amplitude spectrogram
        ref: Reference amplitude
        amin: Minimum amplitude threshold
        top_db: Maximum dynamic range in dB

    Returns:
        dB-scaled spectrogram
    """
    return librosa.amplitude_to_db(S, ref=ref, amin=amin, top_db=top_db)


def db_to_amplitude(S_db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """
    Convert a dB-scaled spectrogram to amplitude.

    Args:
        S_db: dB-scaled spectrogram
        ref: Reference amplitude

    Returns:
        Amplitude spectrogram
    """
    return librosa.db_to_amplitude(S_db, ref=ref)


def power_to_db(
    S: np.ndarray,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Convert a power spectrogram to dB-scaled spectrogram.

    Args:
        S: Input power spectrogram
        ref: Reference power
        amin: Minimum power threshold
        top_db: Maximum dynamic range in dB

    Returns:
        dB-scaled spectrogram
    """
    return librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)


def db_to_power(S_db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """
    Convert a dB-scaled spectrogram to power.

    Args:
        S_db: dB-scaled spectrogram
        ref: Reference power

    Returns:
        Power spectrogram
    """
    return librosa.db_to_power(S_db, ref=ref)


def get_window(window: str, win_length: int, fftbins: bool = True) -> np.ndarray:
    """
    Get a window function.

    Args:
        window: Window type ('hann', 'hamming', 'rectangular', etc.)
        win_length: Window length in samples
        fftbins: If True, create a "periodic" window for FFT

    Returns:
        Window function array
    """
    if window == 'rectangular' or window == 'rect':
        window = 'boxcar'
    return librosa.filters.get_window(window, win_length, fftbins=fftbins)
