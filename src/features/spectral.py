"""
Spectral feature extraction using PyTorch.

All features are computed using torch operations for GPU acceleration.
Input spectrograms should be torch tensors.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict


def spectral_centroid(
    S: torch.Tensor,
    freqs: torch.Tensor,
    dim: int = -2
) -> torch.Tensor:
    """
    Compute the spectral centroid (center of mass of the spectrum).

    The spectral centroid is the "brightness" of a sound.

    Args:
        S: Magnitude spectrogram, shape (..., n_freqs, n_frames)
        freqs: Frequency values in Hz, shape (n_freqs,)
        dim: Dimension along which frequencies are arranged

    Returns:
        Spectral centroid for each frame, shape (..., n_frames)
    """
    # Ensure freqs is broadcastable
    freqs = freqs.view(-1, 1) if freqs.dim() == 1 else freqs

    # Normalize spectrogram
    S_norm = S / (S.sum(dim=dim, keepdim=True) + 1e-10)

    # Compute weighted mean
    centroid = (freqs * S_norm).sum(dim=dim)

    return centroid


def spectral_bandwidth(
    S: torch.Tensor,
    freqs: torch.Tensor,
    centroid: Optional[torch.Tensor] = None,
    p: float = 2.0,
    dim: int = -2
) -> torch.Tensor:
    """
    Compute the spectral bandwidth (spread of the spectrum around centroid).

    Args:
        S: Magnitude spectrogram, shape (..., n_freqs, n_frames)
        freqs: Frequency values in Hz, shape (n_freqs,)
        centroid: Pre-computed spectral centroid (optional)
        p: Order of the bandwidth (default: 2)
        dim: Dimension along which frequencies are arranged

    Returns:
        Spectral bandwidth for each frame
    """
    if centroid is None:
        centroid = spectral_centroid(S, freqs, dim=dim)

    # Ensure freqs is broadcastable
    freqs = freqs.view(-1, 1) if freqs.dim() == 1 else freqs
    centroid = centroid.unsqueeze(dim) if centroid.dim() < S.dim() else centroid

    # Normalize spectrogram
    S_norm = S / (S.sum(dim=dim, keepdim=True) + 1e-10)

    # Compute weighted deviation
    deviation = torch.abs(freqs - centroid) ** p
    bandwidth = ((deviation * S_norm).sum(dim=dim)) ** (1.0 / p)

    return bandwidth


def spectral_rolloff(
    S: torch.Tensor,
    freqs: torch.Tensor,
    roll_percent: float = 0.85,
    dim: int = -2
) -> torch.Tensor:
    """
    Compute the spectral rolloff frequency.

    The rolloff frequency is the frequency below which roll_percent
    of the total spectral energy is contained.

    Args:
        S: Magnitude spectrogram, shape (..., n_freqs, n_frames)
        freqs: Frequency values in Hz
        roll_percent: Cumulative energy threshold (0-1)
        dim: Dimension along which frequencies are arranged

    Returns:
        Rolloff frequency for each frame in Hz
    """
    # Compute power spectrum
    power = S ** 2

    # Cumulative sum along frequency axis
    cumsum = torch.cumsum(power, dim=dim)
    total = power.sum(dim=dim, keepdim=True)

    # Find threshold
    threshold = roll_percent * total

    # Find index where cumsum exceeds threshold
    # This is a bit tricky in torch - we use argmax on a boolean mask
    mask = cumsum >= threshold

    # Get first True index along frequency dimension
    # Add small offset to handle edge cases
    indices = mask.float().argmax(dim=dim)

    # Map indices to frequencies
    freqs = freqs.to(S.device)
    rolloff = freqs[indices.clamp(0, len(freqs) - 1)]

    return rolloff


def spectral_flatness(
    S: torch.Tensor,
    dim: int = -2,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute the spectral flatness (tonality coefficient).

    Flatness is the ratio of geometric mean to arithmetic mean.
    High values (~1) indicate noise-like signal, low values indicate tonal signal.

    Args:
        S: Magnitude spectrogram, shape (..., n_freqs, n_frames)
        dim: Dimension along which frequencies are arranged
        eps: Small constant for numerical stability

    Returns:
        Spectral flatness for each frame (0-1)
    """
    # Ensure positive values
    S_pos = S.clamp(min=eps)

    # Geometric mean = exp(mean(log(x)))
    log_S = torch.log(S_pos)
    geometric_mean = torch.exp(log_S.mean(dim=dim))

    # Arithmetic mean
    arithmetic_mean = S_pos.mean(dim=dim)

    # Flatness
    flatness = geometric_mean / (arithmetic_mean + eps)

    return flatness


def spectral_flux(
    S: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute the spectral flux (rate of change of spectrum).

    Args:
        S: Magnitude spectrogram, shape (..., n_freqs, n_frames)
        dim: Time dimension

    Returns:
        Spectral flux for each frame (first frame is 0)
    """
    # Compute difference between consecutive frames
    if dim == -1 or dim == S.dim() - 1:
        diff = S[..., 1:] - S[..., :-1]
        # Pad first frame with zeros
        pad_shape = list(S.shape)
        pad_shape[dim] = 1
        pad = torch.zeros(pad_shape, device=S.device, dtype=S.dtype)
        diff = torch.cat([pad, diff], dim=dim)
    else:
        diff = torch.diff(S, dim=dim)
        pad_shape = list(S.shape)
        pad_shape[dim] = 1
        pad = torch.zeros(pad_shape, device=S.device, dtype=S.dtype)
        diff = torch.cat([pad, diff], dim=dim)

    # L2 norm of positive differences (onset detection)
    diff_pos = torch.relu(diff)
    flux = torch.sqrt((diff_pos ** 2).sum(dim=-2))

    return flux


def spectral_contrast(
    S: torch.Tensor,
    freqs: torch.Tensor,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    dim: int = -2
) -> torch.Tensor:
    """
    Compute spectral contrast.

    Spectral contrast measures the difference between peaks and valleys
    in the spectrum.

    Args:
        S: Magnitude spectrogram
        freqs: Frequency values
        n_bands: Number of frequency bands
        fmin: Minimum frequency
        quantile: Quantile for valley detection
        dim: Frequency dimension

    Returns:
        Spectral contrast, shape (n_bands + 1, n_frames)
    """
    device = S.device
    dtype = S.dtype

    # Define octave bands
    fmax = freqs[-1].item()
    octa = torch.zeros(n_bands + 2, device=device, dtype=dtype)
    octa[0] = fmin
    for i in range(1, n_bands + 2):
        octa[i] = octa[i-1] * 2
    octa = octa.clamp(max=fmax)

    contrast = []
    for i in range(n_bands + 1):
        # Find frequency bins in this band
        if i < n_bands:
            mask = (freqs >= octa[i]) & (freqs < octa[i + 1])
        else:
            mask = freqs >= octa[i]

        if mask.sum() == 0:
            # Empty band
            contrast.append(torch.zeros(S.shape[-1], device=device, dtype=dtype))
            continue

        # Get band spectrum
        band_S = S[mask, :]

        # Compute peak and valley
        k = max(1, int(quantile * band_S.shape[0]))
        sorted_S, _ = torch.sort(band_S, dim=0)

        peak = sorted_S[-k:, :].mean(dim=0)
        valley = sorted_S[:k, :].mean(dim=0)

        # Contrast is log difference
        c = torch.log(peak + 1e-10) - torch.log(valley + 1e-10)
        contrast.append(c)

    return torch.stack(contrast, dim=0)


def zero_crossing_rate(
    y: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True
) -> torch.Tensor:
    """
    Compute the zero-crossing rate of an audio signal.

    Args:
        y: Audio time series
        frame_length: Length of analysis frames
        hop_length: Hop between frames
        center: If True, pad signal for centered frames

    Returns:
        Zero-crossing rate for each frame
    """
    if center:
        pad_amount = frame_length // 2
        y = torch.nn.functional.pad(y, (pad_amount, pad_amount))

    # Compute sign changes
    sign_changes = torch.abs(torch.diff(torch.sign(y)))

    # Frame the signal
    num_samples = len(y)
    num_frames = 1 + (num_samples - frame_length) // hop_length

    zcr = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length - 1  # -1 because diff reduces length by 1
        frame_zcr = sign_changes[start:end].sum() / (2 * (frame_length - 1))
        zcr.append(frame_zcr)

    return torch.stack(zcr)


def rms_energy(
    y: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True
) -> torch.Tensor:
    """
    Compute root-mean-square (RMS) energy for each frame.

    Args:
        y: Audio time series
        frame_length: Length of analysis frames
        hop_length: Hop between frames
        center: If True, pad signal for centered frames

    Returns:
        RMS energy for each frame
    """
    if center:
        pad_amount = frame_length // 2
        y = torch.nn.functional.pad(y, (pad_amount, pad_amount))

    # Frame the signal
    num_samples = len(y)
    num_frames = 1 + (num_samples - frame_length) // hop_length

    rms = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame_rms = torch.sqrt(torch.mean(y[start:end] ** 2))
        rms.append(frame_rms)

    return torch.stack(rms)


def extract_spectral_features(
    S: torch.Tensor,
    freqs: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    sr: int = 22050,
    hop_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Extract all spectral features from a spectrogram.

    Args:
        S: Magnitude spectrogram, shape (n_freqs, n_frames)
        freqs: Frequency values in Hz
        y: Original audio signal (optional, for ZCR and RMS)
        sr: Sample rate
        hop_length: Hop length used for STFT

    Returns:
        Dictionary of feature tensors
    """
    features = {}

    # Spectral features
    features['spectral_centroid'] = spectral_centroid(S, freqs)
    features['spectral_bandwidth'] = spectral_bandwidth(
        S, freqs, centroid=features['spectral_centroid']
    )
    features['spectral_rolloff'] = spectral_rolloff(S, freqs)
    features['spectral_flatness'] = spectral_flatness(S)
    features['spectral_flux'] = spectral_flux(S)

    # Time-domain features (if audio is provided)
    if y is not None:
        frame_length = (S.shape[0] - 1) * 2  # Infer from spectrogram size
        features['zcr'] = zero_crossing_rate(y, frame_length, hop_length)
        features['rms'] = rms_energy(y, frame_length, hop_length)

        # Ensure same length as spectrogram
        n_frames = S.shape[1]
        for key in ['zcr', 'rms']:
            if features[key].shape[0] > n_frames:
                features[key] = features[key][:n_frames]
            elif features[key].shape[0] < n_frames:
                pad = torch.zeros(
                    n_frames - features[key].shape[0],
                    device=features[key].device,
                    dtype=features[key].dtype
                )
                features[key] = torch.cat([features[key], pad])

    return features
