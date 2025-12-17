"""
Delta (derivative) feature computation using PyTorch.

Delta features capture the temporal dynamics of audio features.
"""

import torch
from typing import Optional


def compute_delta(
    data: torch.Tensor,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = 'replicate'
) -> torch.Tensor:
    """
    Compute delta features (local estimate of the derivative).

    Uses a Savitzky-Golay style filter for smooth derivative estimation.

    Args:
        data: Input feature matrix, shape (..., n_features, n_frames) or (..., n_frames)
        width: Filter width (must be odd, typically 9)
        order: Derivative order (1 for delta, 2 for delta-delta)
        axis: Axis along which to compute deltas (typically -1 for time axis)
        mode: Padding mode ('replicate', 'reflect', 'constant')

    Returns:
        Delta features with same shape as input
    """
    if width % 2 == 0:
        width += 1  # Ensure odd width

    half_width = width // 2

    if order == 2:
        return compute_delta_delta(data, width=width, axis=axis, mode=mode)

    if order != 1:
        raise ValueError(f"Order must be 1 or 2, got {order}")

    # Create finite difference weights (regression slope)
    n = torch.arange(-half_width, half_width + 1, dtype=data.dtype, device=data.device)
    weights = n / (n ** 2).sum()

    # Reshape weights for convolution
    weights = weights.view(1, 1, -1)

    # Handle different input shapes
    original_shape = data.shape
    original_ndim = data.ndim

    # Reshape to (batch, channels, time) for conv1d
    if axis == -1 or axis == data.ndim - 1:
        time_axis = -1
    else:
        # Move time axis to last position
        data = data.transpose(axis, -1)
        time_axis = -1

    if data.ndim == 1:
        data = data.view(1, 1, -1)
    elif data.ndim == 2:
        data = data.unsqueeze(0)
    elif data.ndim > 3:
        # Flatten batch dimensions
        batch_shape = data.shape[:-2]
        data = data.view(-1, data.shape[-2], data.shape[-1])

    # Pad the input
    if mode == 'replicate':
        data_padded = torch.nn.functional.pad(data, (half_width, half_width), mode='replicate')
    elif mode == 'reflect':
        data_padded = torch.nn.functional.pad(data, (half_width, half_width), mode='reflect')
    elif mode == 'constant':
        data_padded = torch.nn.functional.pad(data, (half_width, half_width), mode='constant', value=0)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    # Apply convolution for each feature channel
    batch_size, n_features, n_time = data_padded.shape[:3]
    delta = []

    for i in range(n_features):
        channel = data_padded[:, i:i+1, :]
        d = torch.nn.functional.conv1d(channel, weights)
        delta.append(d)

    delta = torch.cat(delta, dim=1)

    # Restore original shape
    if original_ndim == 1:
        delta = delta.squeeze(0).squeeze(0)
    elif original_ndim == 2:
        delta = delta.squeeze(0)
    elif original_ndim > 3:
        delta = delta.view(*batch_shape, delta.shape[-2], delta.shape[-1])

    if axis != -1 and axis != original_ndim - 1:
        delta = delta.transpose(axis, -1)

    return delta


def compute_delta_delta(
    data: torch.Tensor,
    width: int = 9,
    axis: int = -1,
    mode: str = 'replicate'
) -> torch.Tensor:
    """
    Compute delta-delta (acceleration) features.

    This is equivalent to applying delta twice.

    Args:
        data: Input feature matrix
        width: Filter width
        axis: Time axis
        mode: Padding mode

    Returns:
        Delta-delta features with same shape as input
    """
    delta = compute_delta(data, width=width, order=1, axis=axis, mode=mode)
    delta_delta = compute_delta(delta, width=width, order=1, axis=axis, mode=mode)
    return delta_delta


def add_deltas(
    features: torch.Tensor,
    width: int = 9,
    axis: int = -1,
    include_delta: bool = True,
    include_delta_delta: bool = True
) -> torch.Tensor:
    """
    Add delta and delta-delta features to input features.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        width: Filter width for delta computation
        axis: Time axis
        include_delta: If True, include first derivative
        include_delta_delta: If True, include second derivative

    Returns:
        Concatenated features with deltas, shape (..., n_features * k, n_frames)
        where k is 1 + include_delta + include_delta_delta
    """
    result = [features]

    if include_delta:
        delta = compute_delta(features, width=width, axis=axis)
        result.append(delta)

    if include_delta_delta:
        delta_delta = compute_delta_delta(features, width=width, axis=axis)
        result.append(delta_delta)

    # Concatenate along feature dimension (second to last)
    return torch.cat(result, dim=-2)
