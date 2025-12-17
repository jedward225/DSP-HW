"""
DSP Core Module

This module provides wrappers for digital signal processing functions.
Only this module is allowed to import scipy and librosa.
All other modules should import from dsp_core.
"""

# FFT functions
from .fft import (
    fft,
    ifft,
    rfft,
    irfft,
    fft2,
    ifft2,
    fftfreq,
    rfftfreq,
)

# STFT functions
from .stft import (
    stft,
    istft,
    magphase,
    amplitude_to_db,
    db_to_amplitude,
    power_to_db,
    db_to_power,
    get_window,
)

# MFCC functions
from .mfcc import (
    mfcc,
    delta,
    pre_emphasis,
    cmvn,
)

# Mel spectrogram functions
from .mel import (
    melspectrogram,
    mel_frequencies,
    mel_filter_bank,
    hz_to_mel,
    mel_to_hz,
    log_melspectrogram,
    pcen,
    pcen_melspectrogram,
)

# Audio loading (using librosa)
import librosa as _librosa
import numpy as _np


def load_audio(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float = None
) -> tuple:
    """
    Load an audio file.

    Args:
        path: Path to the audio file
        sr: Target sampling rate (None to preserve original)
        mono: Convert to mono
        offset: Start reading at this time (seconds)
        duration: Only load this much audio (seconds)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    y, sr_out = _librosa.load(
        path,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration
    )
    return y, sr_out


def get_duration(
    y: _np.ndarray = None,
    sr: int = 22050,
    filename: str = None
) -> float:
    """
    Get the duration of an audio signal or file.

    Args:
        y: Audio time series
        sr: Sampling rate
        filename: Path to audio file (used if y is None)

    Returns:
        Duration in seconds
    """
    return _librosa.get_duration(y=y, sr=sr, filename=filename)


def resample(y: _np.ndarray, orig_sr: int, target_sr: int) -> _np.ndarray:
    """
    Resample audio from orig_sr to target_sr.

    Args:
        y: Audio time series
        orig_sr: Original sampling rate
        target_sr: Target sampling rate

    Returns:
        Resampled audio
    """
    return _librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def pitch_shift(
    y: _np.ndarray,
    sr: int = 22050,
    n_steps: float = 0.0,
    bins_per_octave: int = 12
) -> _np.ndarray:
    """
    Shift the pitch of a waveform by n_steps semitones.

    Args:
        y: Audio time series
        sr: Sampling rate
        n_steps: Number of semitones to shift (positive = up, negative = down)
        bins_per_octave: Number of steps per octave

    Returns:
        Pitch-shifted audio
    """
    return _librosa.effects.pitch_shift(
        y=y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave
    )


def time_stretch(y: _np.ndarray, rate: float) -> _np.ndarray:
    """
    Time-stretch an audio signal by a fixed rate.

    Args:
        y: Audio time series
        rate: Stretch factor. rate > 1 speeds up, rate < 1 slows down.

    Returns:
        Time-stretched audio
    """
    return _librosa.effects.time_stretch(y=y, rate=rate)


__all__ = [
    # FFT
    'fft', 'ifft', 'rfft', 'irfft', 'fft2', 'ifft2', 'fftfreq', 'rfftfreq',
    # STFT
    'stft', 'istft', 'magphase', 'amplitude_to_db', 'db_to_amplitude',
    'power_to_db', 'db_to_power', 'get_window',
    # MFCC
    'mfcc', 'delta', 'pre_emphasis', 'cmvn',
    # Mel
    'melspectrogram', 'mel_frequencies', 'mel_filter_bank',
    'hz_to_mel', 'mel_to_hz', 'log_melspectrogram',
    'pcen', 'pcen_melspectrogram',
    # Audio
    'load_audio', 'get_duration', 'resample', 'pitch_shift', 'time_stretch',
]
