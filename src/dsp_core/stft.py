import numpy as np
from typing import Union, Optional, Callable
from .fft import rfft

def get_window(window: Union[str, tuple, np.ndarray], win_length: int) -> np.ndarray:
    """
    Generate a window function for STFT.

    Parameters
    ----------
    window : str, tuple, or np.ndarray
        Window specification:
        - 'hann': Hann window (default in librosa)
        - 'hamming': Hamming window
        - 'blackman': Blackman window
        - 'bartlett': Bartlett (triangular) window
        - ('tukey', alpha): Tukey window with parameter alpha
        - np.ndarray: custom window (must have length win_length)
    win_length : int
        Length of the window

    Returns
    -------
    np.ndarray
        Window function of length win_length

    Notes
    -----
    Window functions reduce spectral leakage by smoothly tapering the signal
    at frame boundaries. The Hann window is most commonly used in audio processing.
    """
    if isinstance(window, np.ndarray):
        if len(window) != win_length:
            raise ValueError(f"Custom window length {len(window)} != win_length {win_length}")
        return window

    if isinstance(window, tuple):
        window_type, *params = window
    else:
        window_type = window
        params = []

    # Generate window based on type
    # Note: using N (not N-1) for normalization to match librosa's fftbins=True mode
    if window_type == 'hann':
        # Hann window: w[n] = 0.5 * (1 - cos(2πn / N))
        # This is the "periodic" or "DFT-even" version used by librosa
        n = np.arange(win_length)
        w = 0.5 - 0.5 * np.cos(2 * np.pi * n / win_length)
        return w

    elif window_type == 'hamming':
        # Hamming window: w[n] = 0.54 - 0.46 * cos(2πn / N)
        n = np.arange(win_length)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * n / win_length)
        return w

    elif window_type == 'blackman':
        # Blackman window: w[n] = 0.42 - 0.5*cos(2πn/N) + 0.08*cos(4πn/N)
        n = np.arange(win_length)
        w = (0.42
             - 0.5 * np.cos(2 * np.pi * n / win_length)
             + 0.08 * np.cos(4 * np.pi * n / win_length))
        return w

    elif window_type == 'bartlett':
        # Bartlett (triangular) window
        n = np.arange(win_length)
        w = 1.0 - np.abs((n - (win_length - 1) / 2) / ((win_length - 1) / 2))
        return w

    elif window_type == 'tukey':
        # Tukey window (tapered cosine)
        alpha = params[0] if params else 0.5
        n = np.arange(win_length)
        w = np.ones(win_length)

        # Taper width
        width = int(alpha * (win_length - 1) / 2)

        # Left taper
        left_idx = np.arange(width)
        w[left_idx] = 0.5 * (1 + np.cos(np.pi * (2 * left_idx / (alpha * (win_length - 1)) - 1)))

        # Right taper
        right_idx = np.arange(win_length - width, win_length)
        w[right_idx] = 0.5 * (1 + np.cos(np.pi * (2 * (win_length - 1 - right_idx) / (alpha * (win_length - 1)) - 1)))

        return w
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Union[str, np.ndarray] = 'hann',
    center: bool = True,
    pad_mode: str = 'constant'
) -> np.ndarray:
    """
    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(22050)  # 1 second at 22050 Hz
    >>> D = stft(y, n_fft=2048, hop_length=512)
    >>> D.shape  # (1025, 44) -> 1025 freq bins, 44 time frames
    """
    # Input validation
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"Input must be 1D, got shape {y.shape}")

    # Set default parameters
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Center padding: pad signal so that frame t is centered at y[t * hop_length]
    if center:
        # Pad on both sides
        pad_length = n_fft // 2
        y = np.pad(y, pad_length, mode=pad_mode)

    # Generate window function
    window_func = get_window(window, win_length)

    # If win_length < n_fft, zero-pad the window
    if win_length < n_fft:
        # Center the window in the FFT frame
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window_func = np.pad(window_func, (pad_left, pad_right), mode='constant')
    elif win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) cannot be larger than n_fft ({n_fft})")

    # Calculate number of frames
    n_frames = 1 + (len(y) - n_fft) // hop_length

    # Extract all frames at once (vectorized)
    # Create frame indices
    frame_starts = np.arange(n_frames) * hop_length
    frame_indices = frame_starts[:, np.newaxis] + np.arange(n_fft)

    # Extract frames: shape (n_frames, n_fft)
    frames = y[frame_indices]

    # Apply window to all frames at once
    windowed_frames = frames * window_func

    # Batch FFT using optimized function
    from .fft import rfft_batch
    stft_matrix = rfft_batch(windowed_frames).T  # Transpose to (n_bins, n_frames)

    return stft_matrix


def istft(
    stft_matrix: np.ndarray,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Union[str, np.ndarray] = 'hann',
    center: bool = True,
    length: Optional[int] = None
) -> np.ndarray:
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    n_frames = stft_matrix.shape[1]

    # Set default parameters
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Generate window
    window_func = get_window(window, win_length)

    # Pad window if necessary
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window_func = np.pad(window_func, (pad_left, pad_right), mode='constant')

    # Calculate output length
    expected_length = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_length)

    # Window normalization array (for COLA)
    window_sum = np.zeros(expected_length)

    # Inverse FFT and overlap-add
    from .fft import irfft

    for i in range(n_frames):
        # Inverse FFT
        frame = irfft(stft_matrix[:, i], n=n_fft)

        # Apply synthesis window
        windowed_frame = frame * window_func

        # Overlap-add
        start = i * hop_length
        end = start + n_fft
        y[start:end] += windowed_frame
        window_sum[start:end] += window_func ** 2

    # Normalize by window overlap
    # Avoid division by zero
    window_sum[window_sum < 1e-10] = 1.0
    y = y / window_sum

    # Remove center padding and crop to desired length
    if center:
        pad_length = n_fft // 2
        if length is not None:
            # If length specified, crop from padded signal
            y = y[pad_length:pad_length + length]
        else:
            # Otherwise just remove padding from both ends
            y = y[pad_length:-pad_length]
    else:
        # No center padding, just crop if needed
        if length is not None:
            y = y[:length]

    return y


def power_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    S = np.maximum(amin, S)
    # Convert to dB
    S_db = 10.0 * np.log10(S / ref)

    # Clip to top_db range
    S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db

def amplitude_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-5, top_db: float = 80.0) -> np.ndarray:
    # dB = 20 * log10(|S| / ref)
    # Power = amplitude^2, so use power_to_db with squared inputs
    magnitude = np.abs(S)
    return power_to_db(magnitude ** 2, ref=ref ** 2, amin=amin ** 2, top_db=top_db)

if __name__ == "__main__":
    # python -m src.dsp_core.stft
    import librosa

    print("=" * 70)
    print("STFT Implementation Test")
    print("=" * 70)

    # Test 1: STFT on random signal
    print("\n[Test 1] STFT on random signal")
    y = np.random.randn(22050)  # 1 second at 22050 Hz

    # Our implementation
    D_ours = stft(y, n_fft=2048, hop_length=512, window='hann', center=True)

    # Librosa reference
    D_librosa = librosa.stft(y, n_fft=2048, hop_length=512, window='hann', center=True)

    error = np.abs(D_ours - D_librosa)
    print(f"  Input shape: {y.shape}")
    print(f"  Output shape: {D_ours.shape}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")

    # Test 2: ISTFT reconstruction
    print("\n[Test 2] ISTFT reconstruction")
    y_reconstructed = istft(D_ours, hop_length=512, window='hann', center=True, length=len(y))
    reconstruction_error = np.abs(y - y_reconstructed)
    print(f"  Reconstruction max error: {reconstruction_error.max():.2e}")
    print(f"  Reconstruction mean error: {reconstruction_error.mean():.2e}")
    print(f"  ✓ PASS" if reconstruction_error.max() < 1e-6 else "  ✗ FAIL")

    # Test 3: Different window functions
    print("\n[Test 3] Different window functions")
    windows = ['hann', 'hamming', 'blackman']
    for win in windows:
        D_ours = stft(y, n_fft=2048, hop_length=512, window=win)
        D_librosa = librosa.stft(y, n_fft=2048, hop_length=512, window=win)
        error = np.abs(D_ours - D_librosa)
        status = "✓ PASS" if error.max() < 1e-10 else "✗ FAIL"
        print(f"  {win:12s} - Max error: {error.max():.2e}  {status}")

