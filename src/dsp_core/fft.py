"""
Fast FFT Implementation using Numba JIT

This module implements the Cooley-Tukey FFT algorithm with Numba JIT acceleration.
Optimizations:
1. Numba JIT compilation (nopython mode)
2. Iterative (non-recursive) implementation - avoids Python call overhead
3. In-place bit-reversal permutation
4. Cache compiled functions

Performance: ~20-100x faster than pure Python recursive version
"""

import numpy as np
from numba import jit, prange
from typing import Optional
import math


@jit(nopython=True, cache=True)
def _bit_reverse(x: int, n_bits: int) -> int:
    """Reverse the bits of x with n_bits."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


@jit(nopython=True, cache=True)
def _fft_radix2_iter(x: np.ndarray) -> np.ndarray:
    """
    Iterative Cooley-Tukey radix-2 DIT FFT (Numba JIT).

    Much faster than recursive version due to:
    1. No function call overhead
    2. Better cache locality
    3. JIT compilation to native code
    """
    N = len(x)
    n_bits = int(math.log2(N))

    # Bit-reversal permutation
    X = np.empty(N, dtype=np.complex128)
    for i in range(N):
        j = _bit_reverse(i, n_bits)
        X[j] = x[i]

    # Iterative FFT butterfly operations
    # Process stages: size 2, 4, 8, ..., N
    stage_size = 2
    while stage_size <= N:
        half_size = stage_size // 2
        w_step = -2j * np.pi / stage_size

        # Process each group in this stage
        for k in range(0, N, stage_size):
            w = 1.0 + 0j
            w_mult = np.exp(w_step)

            for j in range(half_size):
                even_idx = k + j
                odd_idx = k + j + half_size

                even = X[even_idx]
                odd = X[odd_idx] * w

                X[even_idx] = even + odd
                X[odd_idx] = even - odd

                w = w * w_mult

        stage_size *= 2

    return X


@jit(nopython=True, cache=True)
def _dft_naive_jit(x: np.ndarray) -> np.ndarray:
    """Naive DFT for non-power-of-2 lengths (JIT compiled)."""
    N = len(x)
    X = np.empty(N, dtype=np.complex128)

    for k in range(N):
        s = 0j
        for n in range(N):
            s += x[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = s

    return X


@jit(nopython=True, cache=True)
def _fft_core(x: np.ndarray) -> np.ndarray:
    """Core FFT: handles both power-of-2 and arbitrary lengths."""
    N = len(x)

    # Check if power of 2
    if N & (N - 1) == 0 and N > 0:
        return _fft_radix2_iter(x)
    else:
        return _dft_naive_jit(x)


def fft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    """
    Compute the 1-D discrete Fourier Transform using Cooley-Tukey FFT.

    Parameters
    ----------
    x : np.ndarray
        Input array
    n : int, optional
        Length of the transformed axis. If None, uses the length of x.
    axis : int
        Axis along which to compute the FFT (default: -1)
    norm : str
        Normalization mode: "backward", "ortho", or "forward"

    Returns
    -------
    np.ndarray
        The transformed array

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 1.0, -1.0, 1.5, 1.0, 0.5, -0.5])
    >>> X = fft(x)
    >>> # Should match scipy.fft.fft(x)
    """
    x = np.asarray(x)

    if n is None:
        n = x.shape[axis]

    # Move target axis to the last position
    x = np.moveaxis(x, axis, -1)

    # Pad or truncate to desired length
    if x.shape[-1] < n:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
        x = np.pad(x, pad_width, mode='constant', constant_values=0)
    elif x.shape[-1] > n:
        x = x[..., :n]

    # Handle multi-dimensional input
    original_shape = x.shape
    if x.ndim > 1:
        # Flatten to 2D, apply FFT to each row
        x_2d = x.reshape(-1, n)
        result_2d = np.empty_like(x_2d, dtype=np.complex128)
        for i in range(x_2d.shape[0]):
            result_2d[i] = _fft_core(x_2d[i].astype(np.complex128))
        result = result_2d.reshape(original_shape)
    else:
        result = _fft_core(x.astype(np.complex128))

    # Apply normalization
    if norm == "ortho":
        result = result / np.sqrt(n)
    elif norm == "forward":
        result = result / n

    result = np.moveaxis(result, -1, axis)
    return result


def ifft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    """
    Compute the 1-D inverse discrete Fourier Transform.

    IFFT(x) = conj(FFT(conj(x))) / N
    """
    x = np.asarray(x)
    x_conj = np.conj(x)

    if norm == "backward":
        result = fft(x_conj, n=n, axis=axis, norm="forward")
    elif norm == "forward":
        result = fft(x_conj, n=n, axis=axis, norm="backward")
    else:
        result = fft(x_conj, n=n, axis=axis, norm="ortho")

    return np.conj(result)


def rfft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    """
    Compute the 1-D FFT for real input.

    Returns only the non-negative frequency terms (one-sided spectrum).
    """
    x = np.asarray(x, dtype=np.float64)

    X = fft(x, n=n, axis=axis, norm=norm)

    N = X.shape[axis]
    slices = [slice(None)] * X.ndim
    slices[axis] = slice(0, N // 2 + 1)

    return X[tuple(slices)]


@jit(nopython=True, cache=True)
def _ifft_core(X: np.ndarray) -> np.ndarray:
    """Core IFFT using conjugate trick."""
    N = len(X)
    X_conj = np.conj(X)
    result = _fft_core(X_conj)
    return np.conj(result) / N


def irfft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: str = "backward") -> np.ndarray:
    """
    Compute the inverse FFT of a real-spectrum signal.
    """
    x = np.asarray(x, dtype=np.complex128)

    if n is None:
        n = 2 * (x.shape[axis] - 1)

    x = np.moveaxis(x, axis, -1)

    # Reconstruct full spectrum using Hermitian symmetry
    if n % 2 == 0:
        neg_freqs = np.conj(x[..., -2:0:-1])
    else:
        neg_freqs = np.conj(x[..., -1:0:-1])

    X_full = np.concatenate([x, neg_freqs], axis=-1)
    X_full = np.moveaxis(X_full, -1, axis)

    result = ifft(X_full, n=n, axis=axis, norm=norm)
    return np.real(result)


# ============== Batch operations for STFT ==============

@jit(nopython=True, cache=True, parallel=True)
def rfft_batch(frames: np.ndarray) -> np.ndarray:
    """
    Batch real FFT for multiple frames (optimized for STFT).

    Parameters
    ----------
    frames : np.ndarray
        Windowed frames, shape (n_frames, n_fft)

    Returns
    -------
    np.ndarray
        FFT results, shape (n_frames, n_fft // 2 + 1)
    """
    n_frames, n_fft = frames.shape
    n_bins = n_fft // 2 + 1
    result = np.empty((n_frames, n_bins), dtype=np.complex128)

    for i in prange(n_frames):
        X = _fft_radix2_iter(frames[i].astype(np.complex128))
        result[i] = X[:n_bins]

    return result


if __name__ == "__main__":
    import time
    from scipy.fft import fft as scipy_fft, rfft as scipy_rfft

    print("=" * 70)
    print("Fast FFT Implementation Test (Numba JIT)")
    print("=" * 70)

    # Warm up JIT compilation
    print("\nWarming up JIT...")
    _ = fft(np.random.randn(1024))
    _ = rfft(np.random.randn(1024))
    print("JIT warm-up complete.")

    # Test 1: Correctness
    print("\n[Test 1] Correctness vs scipy")
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        x = np.random.randn(N)
        X_ours = fft(x)
        X_scipy = scipy_fft(x)
        error = np.abs(X_ours - X_scipy).max()
        status = "✓" if error < 1e-10 else "✗"
        print(f"  N={N:5d}: max_error={error:.2e} {status}")

    # Test 2: Non-power-of-2
    print("\n[Test 2] Non-power-of-2 lengths")
    for N in [100, 200, 500, 1000]:
        x = np.random.randn(N)
        X_ours = fft(x)
        X_scipy = scipy_fft(x)
        error = np.abs(X_ours - X_scipy).max()
        status = "✓" if error < 1e-10 else "✗"
        print(f"  N={N:5d}: max_error={error:.2e} {status}")

    # Test 3: RFFT correctness
    print("\n[Test 3] RFFT Correctness")
    for N in [256, 512, 1024, 2048]:
        x = np.random.randn(N)
        X_ours = rfft(x)
        X_scipy = scipy_rfft(x)
        error = np.abs(X_ours - X_scipy).max()
        status = "✓" if error < 1e-10 else "✗"
        print(f"  N={N:5d}: max_error={error:.2e} {status}")

    # Test 4: Performance benchmark
    print("\n[Test 4] Performance vs scipy")
    print(f"{'N':>6} | {'Ours (ms)':>10} | {'Scipy (ms)':>10} | {'Ratio':>8}")
    print("-" * 50)

    for N in [256, 512, 1024, 2048, 4096]:
        x = np.random.randn(N)
        n_iter = 500

        # Benchmark our version
        start = time.time()
        for _ in range(n_iter):
            _ = fft(x)
        time_ours = (time.time() - start) / n_iter * 1000

        # Benchmark scipy
        start = time.time()
        for _ in range(n_iter):
            _ = scipy_fft(x)
        time_scipy = (time.time() - start) / n_iter * 1000

        ratio = time_ours / time_scipy
        print(f"{N:6d} | {time_ours:10.4f} | {time_scipy:10.4f} | {ratio:7.2f}x")

    # Test 5: Batch RFFT for STFT
    print("\n[Test 5] Batch RFFT Performance (for STFT)")
    n_frames = 200
    n_fft = 2048
    frames = np.random.randn(n_frames, n_fft)

    # Warm up
    _ = rfft_batch(frames[:10])

    start = time.time()
    result = rfft_batch(frames)
    batch_time = time.time() - start

    print(f"  {n_frames} frames x {n_fft} FFT")
    print(f"  Total: {batch_time*1000:.2f} ms")
    print(f"  Per frame: {batch_time/n_frames*1000:.4f} ms")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
