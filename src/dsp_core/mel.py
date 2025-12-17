"""
Mel spectrogram wrapper module - provides Mel functions using librosa.
Only this module (dsp_core) is allowed to import librosa.
"""

import numpy as np
import librosa


def melspectrogram(
    y: np.ndarray = None,
    sr: int = 22050,
    S: np.ndarray = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'constant',
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None
) -> np.ndarray:
    """
    Compute a mel-scaled spectrogram.

    Args:
        y: Audio time series
        sr: Sampling rate
        S: Pre-computed power spectrogram (optional)
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Window length (default: n_fft)
        window: Window function type
        center: If True, frames are centered
        pad_mode: Padding mode
        power: Exponent for the magnitude spectrogram
        n_mels: Number of Mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (default: sr/2)

    Returns:
        Mel spectrogram, shape (n_mels, n_frames)
    """
    if win_length is None:
        win_length = n_fft

    # Handle rectangular window
    if window == 'rectangular' or window == 'rect':
        window = 'boxcar'

    return librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )


def mel_frequencies(n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0) -> np.ndarray:
    """
    Compute Mel frequencies.

    Args:
        n_mels: Number of Mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz

    Returns:
        Array of n_mels frequencies in Hz
    """
    return librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)


def mel_filter_bank(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    htk: bool = False,
    norm: str = 'slaney'
) -> np.ndarray:
    """
    Create a Mel filter bank.

    Args:
        sr: Sampling rate
        n_fft: FFT window size
        n_mels: Number of Mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (default: sr/2)
        htk: Use HTK formula instead of Slaney
        norm: Normalization ('slaney' or None)

    Returns:
        Mel filter bank matrix, shape (n_mels, 1 + n_fft/2)
    """
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm
    )


def hz_to_mel(frequencies: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert Hz to Mel scale.

    Args:
        frequencies: Frequencies in Hz
        htk: Use HTK formula

    Returns:
        Frequencies in Mel scale
    """
    return librosa.hz_to_mel(frequencies, htk=htk)


def mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert Mel scale to Hz.

    Args:
        mels: Frequencies in Mel scale
        htk: Use HTK formula

    Returns:
        Frequencies in Hz
    """
    return librosa.mel_to_hz(mels, htk=htk)


def log_melspectrogram(
    y: np.ndarray = None,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Compute log-scaled mel spectrogram (in dB).

    This is a convenience function that computes mel spectrogram
    and converts it to dB scale.

    Args:
        y: Audio time series
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Window length
        window: Window function type
        n_mels: Number of Mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        ref: Reference power for dB conversion
        amin: Minimum amplitude threshold
        top_db: Maximum dynamic range

    Returns:
        Log-mel spectrogram in dB, shape (n_mels, n_frames)
    """
    mel_spec = melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    return librosa.power_to_db(mel_spec, ref=ref, amin=amin, top_db=top_db)


def pcen(
    S: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-Channel Energy Normalization (PCEN).

    PCEN applies automatic gain control per frequency channel,
    providing better noise robustness than log scaling.

    PCEN computes:
        output = (S / (eps + M)^gain + bias)^power - bias^power
    where M is a smoothed version of S (IIR filter).

    Args:
        S: Input magnitude spectrogram (non-negative)
        sr: Sampling rate
        hop_length: Hop length in samples
        gain: Gain factor (typically < 1)
        bias: Bias point for nonlinear compression
        power: Compression exponent (0-0.5)
        time_constant: IIR filter time constant in seconds
        eps: Numerical stability constant

    Returns:
        PCEN-normalized spectrogram, same shape as input
    """
    return librosa.pcen(
        S=S,
        sr=sr,
        hop_length=hop_length,
        gain=gain,
        bias=bias,
        power=power,
        time_constant=time_constant,
        eps=eps
    )


def pcen_melspectrogram(
    y: np.ndarray = None,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = 'hann',
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    gain: float = 0.98,
    bias: float = 2.0,
    power: float = 0.5,
    time_constant: float = 0.4,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute PCEN-normalized mel spectrogram.

    Convenience function combining mel spectrogram + PCEN.
    PCEN provides automatic gain control and is more robust to
    noise compared to log-mel spectrograms.

    Args:
        y: Audio time series
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Number of samples between frames
        win_length: Window length
        window: Window function type
        n_mels: Number of Mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        gain: PCEN gain factor
        bias: PCEN bias point
        power: PCEN compression exponent
        time_constant: PCEN IIR filter time constant
        eps: Numerical stability constant

    Returns:
        PCEN-normalized mel spectrogram, shape (n_mels, n_frames)
    """
    # Compute magnitude mel spectrogram (power=1 for magnitude, not power)
    mel_spec = melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0  # magnitude spectrogram for PCEN
    )

    return pcen(
        S=mel_spec,
        sr=sr,
        hop_length=hop_length,
        gain=gain,
        bias=bias,
        power=power,
        time_constant=time_constant,
        eps=eps
    )
