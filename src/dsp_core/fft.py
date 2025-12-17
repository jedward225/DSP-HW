"""
FFT wrapper module - provides FFT functions using scipy.
Only this module (dsp_core) is allowed to import scipy.
"""

import numpy as np
from scipy import fft as scipy_fft


def fft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """
    Compute the 1-D discrete Fourier Transform.

    Args:
        x: Input array
        n: Length of the transformed axis of the output. If n is smaller than
           the length of the input, the input is cropped. If it is larger, the
           input is padded with zeros.
        axis: Axis over which to compute the FFT

    Returns:
        Complex ndarray of the FFT result
    """
    return scipy_fft.fft(x, n=n, axis=axis)


def ifft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """
    Compute the 1-D inverse discrete Fourier Transform.

    Args:
        x: Input array
        n: Length of the transformed axis of the output
        axis: Axis over which to compute the inverse FFT

    Returns:
        Complex ndarray of the inverse FFT result
    """
    return scipy_fft.ifft(x, n=n, axis=axis)


def rfft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """
    Compute the 1-D discrete Fourier Transform for real input.

    Args:
        x: Input array (real-valued)
        n: Length of the transformed axis of the output
        axis: Axis over which to compute the FFT

    Returns:
        Complex ndarray containing the positive frequency terms
    """
    return scipy_fft.rfft(x, n=n, axis=axis)


def irfft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """
    Compute the inverse of rfft.

    Args:
        x: Input array
        n: Length of the transformed axis of the output
        axis: Axis over which to compute the inverse FFT

    Returns:
        Real ndarray of the inverse FFT result
    """
    return scipy_fft.irfft(x, n=n, axis=axis)


def fft2(x: np.ndarray, s: tuple = None, axes: tuple = (-2, -1)) -> np.ndarray:
    """
    Compute the 2-D discrete Fourier Transform.

    Args:
        x: Input array
        s: Shape of the output along each transformed axis
        axes: Axes over which to compute the FFT

    Returns:
        Complex ndarray of the 2-D FFT result
    """
    return scipy_fft.fft2(x, s=s, axes=axes)


def ifft2(x: np.ndarray, s: tuple = None, axes: tuple = (-2, -1)) -> np.ndarray:
    """
    Compute the 2-D inverse discrete Fourier Transform.

    Args:
        x: Input array
        s: Shape of the output along each transformed axis
        axes: Axes over which to compute the inverse FFT

    Returns:
        Complex ndarray of the 2-D inverse FFT result
    """
    return scipy_fft.ifft2(x, s=s, axes=axes)


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        n: Window length
        d: Sample spacing (inverse of the sampling rate)

    Returns:
        Array of length n containing the sample frequencies
    """
    return scipy_fft.fftfreq(n, d=d)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies for rfft.

    Args:
        n: Window length
        d: Sample spacing (inverse of the sampling rate)

    Returns:
        Array containing the sample frequencies
    """
    return scipy_fft.rfftfreq(n, d=d)
