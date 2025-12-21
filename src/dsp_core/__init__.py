"""
DSP Core Module - Hand-written FFT, STFT, and MFCC Implementations

This module provides from-scratch implementations of fundamental DSP algorithms
used in audio signal processing, designed to match the precision and functionality
of standard libraries like scipy and librosa.

Modules:
    - fft: Fast Fourier Transform (Cooley-Tukey algorithm)
    - stft: Short-Time Fourier Transform
    - mfcc: Mel-Frequency Cepstral Coefficients

Author: DSP-HW Team
Date: 2025-12-07
"""

from .fft import fft, ifft, rfft, irfft
from .stft import stft, istft, get_window, power_to_db, amplitude_to_db
from .mfcc import mfcc, mel_filterbank, delta, hz_to_mel, mel_to_hz, dct

__all__ = [
    # FFT functions
    'fft',
    'ifft',
    'rfft',
    'irfft',
    # STFT functions
    'stft',
    'istft',
    'get_window',
    'power_to_db',
    'amplitude_to_db',
    # MFCC functions
    'mfcc',
    'mel_filterbank',
    'delta',
    'hz_to_mel',
    'mel_to_hz',
    'dct',
]

__version__ = '1.0.0'
