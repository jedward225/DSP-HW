"""
This module implements the Cooley-Tukey FFT algorithm from scratch using only NumPy.
The implementation is designed to match the precision and functionality of scipy.fft.
- X[k] = Σ(n=0 to N-1) x[n] * exp(-j * 2π * k * n / N)
"""

import numpy as np
from typing import Union, Optional


def fft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    """
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
    >>> X = fft(x)
    >>> # Should match scipy.fft.fft(x)
    """

    x = np.asarray(x)
    if n is None:
        n = x.shape[axis]

    # Move target axis to the last position for easier processing
    x = np.moveaxis(x, axis, -1)

    # Crop or zero-pad to desired length
    if x.shape[-1] < n:
        # Zero-pad
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
        x = np.pad(x, pad_width, mode='constant', constant_values=0)
    elif x.shape[-1] > n:
        # Crop
        x = x[..., :n]

    result = _fft_recursive(x.astype(np.complex128))

    if norm == "ortho":
        result /= np.sqrt(n)
    elif norm == "forward":
        result /= n

    result = np.moveaxis(result, -1, axis)

    return result


def _fft_recursive(x: np.ndarray) -> np.ndarray:
    N = x.shape[-1]

    if N <= 1:
        return x

    # For non-power-of-2, use naive DFT (fallback)
    if N & (N - 1) != 0:  # Check if N is not a power of 2
        return _dft_naive(x)

    # Divide: separate even and odd indices
    even = _fft_recursive(x[..., ::2])
    odd = _fft_recursive(x[..., 1::2])
    
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)

    for _ in range(x.ndim - 1):
        W = np.expand_dims(W, axis=0)

    first_half = even + W * odd
    second_half = even - W * odd

    result = np.concatenate([first_half, second_half], axis=-1)
    return result


def _dft_naive(x: np.ndarray) -> np.ndarray:
    """
    Naive DFT implementation for non-power-of-2 lengths.
    """
    N = x.shape[-1]
    n = np.arange(N)
    k = n.reshape(-1, 1)

    # W[k, n] = exp(-2πj * k * n / N)
    W = np.exp(-2j * np.pi * k * n / N)

    for _ in range(x.ndim - 1):
        W = np.expand_dims(W, axis=0)

    result = np.sum(x[..., np.newaxis, :] * W, axis=-1)
    return result


def ifft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    x = np.asarray(x)

    # IFFT(x) = conj(FFT(conj(x))) / N
    x_conj = np.conj(x)

    if norm == "backward":
        result = fft(x_conj, n=n, axis=axis, norm="forward")
    elif norm == "forward":
        result = fft(x_conj, n=n, axis=axis, norm="backward")
    else:
        result = fft(x_conj, n=n, axis=axis, norm="ortho")

    return np.conj(result)


def rfft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    X = fft(x, n=n, axis=axis, norm=norm)

    N = X.shape[axis]

    slices = [slice(None)] * X.ndim
    slices[axis] = slice(0, N // 2 + 1)
    return X[tuple(slices)]


def irfft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)

    # Determine output length
    if n is None:
        n = 2 * (x.shape[axis] - 1)

    # Reconstruct full spectrum using Hermitian symmetry
    # X[0], X[1], ..., X[N//2], X[N//2-1]*, ..., X[1]*
    x = np.moveaxis(x, axis, -1)

    
    if n % 2 == 0:
        neg_freqs = np.conj(x[..., -2:0:-1])
    else:
        neg_freqs = np.conj(x[..., -1:0:-1])

    X_full = np.concatenate([x, neg_freqs], axis=-1)
    X_full = np.moveaxis(X_full, -1, axis)

    result = ifft(X_full, n=n, axis=axis, norm=norm)
    return np.real(result)


if __name__ == "__main__":
    # python -m src.dsp_core.fft
    print("=" * 70)
    print("FFT Implementation Test")
    print("=" * 70)

    # Test 1: Simple signal
    print("\n[Test 1] Simple signal (power of 2)")
    x = np.array([1.0, 2.0, 1.0, -1.0, 1.5, 1.0, 0.5, -0.5])
    X_ours = fft(x)

    from scipy.fft import fft as scipy_fft
    X_scipy = scipy_fft(x)

    error = np.abs(X_ours - X_scipy)
    print(f"  Input length: {len(x)}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")

    # Test 2: Non-power-of-2 length
    print("\n[Test 2] Non-power-of-2 length")
    x = np.random.randn(100)
    X_ours = fft(x)
    X_scipy = scipy_fft(x)
    error = np.abs(X_ours - X_scipy)
    print(f"  Input length: {len(x)}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")

    # Test 3: Real FFT
    print("\n[Test 3] Real FFT (rfft)")
    from scipy.fft import rfft as scipy_rfft
    x = np.random.randn(256)
    X_ours = rfft(x)
    X_scipy = scipy_rfft(x)
    error = np.abs(X_ours - X_scipy)
    print(f"  Input length: {len(x)}")
    print(f"  Output length: {len(X_ours)} (should be {len(x)//2 + 1})")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")
