
import numpy as np
from typing import Optional, Union
from .stft import stft

def hz_to_mel(frequencies: np.ndarray, htk: bool = False) -> np.ndarray:
    frequencies = np.asarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    else:
        # Slaney formula (librosa default)
        # Constants
        f_min = 0.0
        f_sp = 200.0 / 3.0  # ~66.67 Hz per mel
        min_log_hz = 1000.0  # Transition point
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0  # Log step size

        # Avoid log(0) by using np.maximum
        mels = np.where(
            frequencies < min_log_hz,
            # Linear part (below 1000 Hz)
            (frequencies - f_min) / f_sp,
            # Logarithmic part (above 1000 Hz)
            min_log_mel + np.log(np.maximum(frequencies, 1e-10) / min_log_hz) / logstep
        )
        return mels


def mel_to_hz(mels: np.ndarray, htk: bool = False) -> np.ndarray:
    mels = np.asarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        # Slaney formula (inverse)
        # Constants (must match hz_to_mel)
        f_min = 0.0
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        # Piecewise inverse
        freqs = np.where(
            mels < min_log_mel,
            # Linear part (inverse)
            f_min + f_sp * mels,
            # Logarithmic part (inverse)
            min_log_hz * np.exp(logstep * (mels - min_log_mel))
        )
        return freqs


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0

    # Number of FFT bins
    n_freqs = n_fft // 2 + 1

    # FFT bin frequencies in Hz
    fft_freqs = np.linspace(0, sr / 2, n_freqs)

    # Mel scale: min, max, and evenly spaced points
    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]

    # Create n_mels + 2 points on Mel scale (including edges)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert back to Hz
    hz_points = mel_to_hz(mel_points)

    # Initialize filterbank matrix
    filterbank = np.zeros((n_mels, n_freqs))

    # Create triangular filters using frequency-domain approach
    # This matches librosa's implementation more closely
    for i in range(n_mels):
        # Three frequencies define the triangle
        left_hz = hz_points[i]
        center_hz = hz_points[i + 1]
        right_hz = hz_points[i + 2]

        # For each FFT bin, compute the filter response
        for j in range(n_freqs):
            freq = fft_freqs[j]

            # Triangular filter response
            if left_hz <= freq <= center_hz:
                if center_hz > left_hz:
                    filterbank[i, j] = (freq - left_hz) / (center_hz - left_hz)
            elif center_hz < freq <= right_hz:
                if right_hz > center_hz:
                    filterbank[i, j] = (right_hz - freq) / (right_hz - center_hz)

    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]

    return filterbank


def dct(x: np.ndarray, n_coeffs: Optional[int] = None, norm: str = 'ortho') -> np.ndarray:
    x = np.asarray(x)
    N = x.shape[-1]

    if n_coeffs is None:
        n_coeffs = N

    n = np.arange(N)
    k = np.arange(n_coeffs)[:, np.newaxis]

    # DCT-II formula: cos(π * k * (n + 0.5) / N)
    dct_matrix = np.cos(np.pi * k * (n + 0.5) / N)

    if norm == 'ortho':
        # First coefficient scaled by 1/sqrt(N), others by sqrt(2/N)
        dct_matrix[0] *= 1.0 / np.sqrt(N)
        dct_matrix[1:] *= np.sqrt(2.0 / N)

    # Apply DCT: matrix multiplication along last axis
    result = np.dot(x, dct_matrix.T)
    return result


def mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = 'hann',
    center: bool = True,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    preemphasis: float = 0.0,
    lifter: int = 0
) -> np.ndarray:
    """
    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.randn(22050)  # 1 second audio
    >>> mfccs = mfcc(y, sr=22050, n_mfcc=13)
    >>> mfccs.shape  # (13, n_frames)
    """
    # Input validation
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"Input must be 1D, got shape {y.shape}")

    # Set default parameters
    if hop_length is None:
        hop_length = n_fft // 4
    if fmax is None:
        fmax = sr / 2.0

    # Step 1: Pre-emphasis (high-pass filter)
    # This compensates for the -6 dB/octave roll-off in typical speech
    if preemphasis > 0:
        y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # Step 2: Compute STFT
    S = stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )

    # Step 3: Power spectrum
    # |S|² gives power (energy per unit frequency)
    power_spectrum = np.abs(S) ** 2

    # Step 4: Create Mel filterbank
    mel_basis = mel_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # Step 5: Apply Mel filters to power spectrum
    # mel_spectrum[m, t] = Σ_k mel_basis[m, k] * power_spectrum[k, t]
    mel_spectrum = np.dot(mel_basis, power_spectrum)

    # Step 6: Convert to dB scale
    # Librosa uses power_to_db: 10 * log10(S / ref)
    from .stft import power_to_db
    log_mel_spectrum = power_to_db(mel_spectrum, ref=1.0, amin=1e-10, top_db=80.0)

    # Step 7: Apply DCT to get MFCCs

    N = log_mel_spectrum.shape[0]  # n_mels
    n = np.arange(N)
    k = np.arange(n_mfcc)[:, np.newaxis]

    # DCT-II matrix for axis 0
    dct_matrix = np.cos(np.pi * k * (n + 0.5) / N)

    # Orthonormal normalization
    dct_matrix[0] *= 1.0 / np.sqrt(N)
    dct_matrix[1:] *= np.sqrt(2.0 / N)

    # Apply DCT: mfccs[k, t] = Σ_n dct_matrix[k, n] * log_mel[n, t]
    mfccs = np.dot(dct_matrix, log_mel_spectrum)

    # Step 8 (Optional): Liftering
    # Liftering is a cepstral domain filtering that de-emphasizes
    # higher-order coefficients which are more susceptible to noise
    if lifter > 0:
        lifter_coeffs = 1 + (lifter / 2) * np.sin(np.pi * np.arange(n_mfcc) / lifter)
        mfccs *= lifter_coeffs[:, np.newaxis]

    return mfccs


def delta(features: np.ndarray, width: int = 9, order: int = 1) -> np.ndarray:
    """
    Compute delta (first derivative) features.
    Delta features capture temporal dynamics of the signal.
    - delta[t] = Σ(n=-w to w) n * features[t + n] / Σ(n=-w to w) n²
    """
    features = np.asarray(features)

    if width < 3 or width % 2 == 0:
        raise ValueError("Width must be odd and >= 3")

    half_width = width // 2

    # Pad edges by repeating first/last frames
    padded = np.pad(features, ((0, 0), (half_width, half_width)), mode='edge')

    # Compute delta using regression formula
    # This is equivalent to convolving with [w, w-1, ..., -w+1, -w] / denominator
    n = np.arange(-half_width, half_width + 1)
    denominator = np.sum(n ** 2)

    delta_features = np.zeros_like(features)

    for t in range(features.shape[1]):
        # Extract window centered at t (accounting for padding)
        window = padded[:, t:t + width]
        # Weighted sum
        delta_features[:, t] = np.sum(window * n, axis=1) / denominator

    # Recursively compute higher-order deltas
    if order > 1:
        return delta(delta_features, width=width, order=order - 1)

    return delta_features


if __name__ == "__main__":
    # python -m src.dsp_core.mfcc
    import librosa

    print("=" * 70)
    print("MFCC Implementation Test")
    print("=" * 70)

    # Test 1: MFCC on random signal
    print("\n[Test 1] MFCC on random signal")
    y = np.random.randn(22050)  # 1 second at 22050 Hz
    sr = 22050

    # Our implementation
    mfccs_ours = mfcc(y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

    # Librosa reference
    mfccs_librosa = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
    )

    error = np.abs(mfccs_ours - mfccs_librosa)
    print(f"  Input shape: {y.shape}")
    print(f"  MFCC shape: {mfccs_ours.shape}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-5 else "  ✗ FAIL")

    # Test 2: Mel filterbank
    print("\n[Test 2] Mel filterbank")
    mel_fb_ours = mel_filterbank(sr=22050, n_fft=2048, n_mels=128)
    mel_fb_librosa = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)

    error = np.abs(mel_fb_ours - mel_fb_librosa)
    print(f"  Filterbank shape: {mel_fb_ours.shape}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")

    # Test 3: Delta features
    print("\n[Test 3] Delta features")
    delta_ours = delta(mfccs_ours, width=9)
    delta_librosa = librosa.feature.delta(mfccs_librosa, width=9)

    error = np.abs(delta_ours - delta_librosa)
    print(f"  Delta shape: {delta_ours.shape}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")

    # Test 4: DCT
    print("\n[Test 4] DCT (Type-II)")
    from scipy.fftpack import dct as scipy_dct
    x = np.random.randn(128)
    dct_ours = dct(x, norm='ortho')
    dct_scipy = scipy_dct(x, type=2, norm='ortho')

    error = np.abs(dct_ours - dct_scipy)
    print(f"  Input length: {len(x)}")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    print(f"  ✓ PASS" if error.max() < 1e-10 else "  ✗ FAIL")
