"""
Global pooling operations for converting frame-level features to utterance-level.

All operations use PyTorch for GPU acceleration.
"""

import torch
from typing import Union, List, Optional


def global_mean_pool(
    features: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute global mean over time dimension.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over (default: -1, time axis)
        keepdim: If True, keep the pooled dimension

    Returns:
        Pooled features, shape (..., n_features) or (..., n_features, 1)
    """
    return torch.mean(features, dim=dim, keepdim=keepdim)


def global_std_pool(
    features: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True
) -> torch.Tensor:
    """
    Compute global standard deviation over time dimension.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over
        keepdim: If True, keep the pooled dimension
        unbiased: If True, use Bessel's correction

    Returns:
        Pooled features
    """
    # Guard against 1-frame input which causes NaN with unbiased=True
    n_frames = features.shape[dim]
    if n_frames <= 1 and unbiased:
        # For single frame, return zeros (no variance)
        shape = list(features.shape)
        if not keepdim:
            shape.pop(dim)
        return torch.zeros(shape, dtype=features.dtype, device=features.device)
    return torch.std(features, dim=dim, keepdim=keepdim, unbiased=unbiased)


def global_max_pool(
    features: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute global max over time dimension.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over
        keepdim: If True, keep the pooled dimension

    Returns:
        Pooled features (values only, not indices)
    """
    return torch.max(features, dim=dim, keepdim=keepdim).values


def global_min_pool(
    features: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute global min over time dimension.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over
        keepdim: If True, keep the pooled dimension

    Returns:
        Pooled features (values only, not indices)
    """
    return torch.min(features, dim=dim, keepdim=keepdim).values


def global_median_pool(
    features: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute global median over time dimension.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over
        keepdim: If True, keep the pooled dimension

    Returns:
        Pooled features
    """
    return torch.median(features, dim=dim, keepdim=keepdim).values


def mean_std_pool(
    features: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Concatenate mean and std pooling.

    This is a common baseline for audio retrieval.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over (default: -1, time axis)

    Returns:
        Concatenated [mean, std], shape (..., 2 * n_features)
    """
    mean = global_mean_pool(features, dim=dim)
    std = global_std_pool(features, dim=dim)
    return torch.cat([mean, std], dim=-1)


def statistics_pool(
    features: torch.Tensor,
    dim: int = -1,
    stats: List[str] = None
) -> torch.Tensor:
    """
    Compute multiple statistics and concatenate.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        dim: Dimension to pool over
        stats: List of statistics to compute. Options:
               'mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis'
               Default: ['mean', 'std']

    Returns:
        Concatenated statistics, shape (..., len(stats) * n_features)
    """
    if stats is None:
        stats = ['mean', 'std']

    results = []

    for stat in stats:
        if stat == 'mean':
            results.append(global_mean_pool(features, dim=dim))
        elif stat == 'std':
            results.append(global_std_pool(features, dim=dim))
        elif stat == 'max':
            results.append(global_max_pool(features, dim=dim))
        elif stat == 'min':
            results.append(global_min_pool(features, dim=dim))
        elif stat == 'median':
            results.append(global_median_pool(features, dim=dim))
        elif stat == 'skew':
            results.append(_compute_skewness(features, dim=dim))
        elif stat == 'kurtosis':
            results.append(_compute_kurtosis(features, dim=dim))
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return torch.cat(results, dim=-1)


def _compute_skewness(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute skewness (third standardized moment)."""
    mean = features.mean(dim=dim, keepdim=True)
    std = features.std(dim=dim, keepdim=True) + 1e-10
    normalized = (features - mean) / std
    skew = (normalized ** 3).mean(dim=dim)
    return skew


def _compute_kurtosis(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute kurtosis (fourth standardized moment, excess kurtosis)."""
    mean = features.mean(dim=dim, keepdim=True)
    std = features.std(dim=dim, keepdim=True) + 1e-10
    normalized = (features - mean) / std
    kurt = (normalized ** 4).mean(dim=dim) - 3  # Excess kurtosis
    return kurt


def temporal_pyramid_pool(
    features: torch.Tensor,
    levels: List[int] = None,
    dim: int = -1
) -> torch.Tensor:
    """
    Temporal pyramid pooling - pool at multiple temporal scales.

    Args:
        features: Input features, shape (..., n_features, n_frames)
        levels: List of number of segments at each level
                Default: [1, 2, 4] (1 global + 2 halves + 4 quarters)
        dim: Time dimension

    Returns:
        Concatenated pyramid features
    """
    if levels is None:
        levels = [1, 2, 4]

    n_frames = features.shape[dim]
    results = []

    for n_segments in levels:
        # Skip levels that would create empty segments
        if n_frames < n_segments:
            # Fall back to global pooling repeated n_segments times
            global_mean = global_mean_pool(features, dim=dim)
            for _ in range(n_segments):
                results.append(global_mean)
            continue

        segment_size = n_frames // n_segments
        segment_results = []

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else n_frames

            if dim == -1:
                segment = features[..., start:end]
            else:
                # Handle other dimensions
                indices = [slice(None)] * features.ndim
                indices[dim] = slice(start, end)
                segment = features[tuple(indices)]

            # Pool each segment
            segment_mean = global_mean_pool(segment, dim=dim)
            segment_results.append(segment_mean)

        results.extend(segment_results)

    return torch.cat(results, dim=-1)


def attention_pool(
    features: torch.Tensor,
    query: Optional[torch.Tensor] = None,
    dim: int = -1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Attention-weighted pooling.

    If query is None, uses self-attention (attention based on feature norms).

    Args:
        features: Input features, shape (..., n_features, n_frames)
        query: Optional query vector for attention
        dim: Time dimension
        temperature: Softmax temperature

    Returns:
        Attention-pooled features, shape (..., n_features)
    """
    if query is None:
        # Self-attention based on L2 norm of each frame
        attention_scores = torch.norm(features, dim=-2, keepdim=True)
    else:
        # Dot product attention
        attention_scores = torch.sum(features * query.unsqueeze(-1), dim=-2, keepdim=True)

    # Apply softmax
    attention_weights = torch.softmax(attention_scores / temperature, dim=dim)

    # Weighted sum
    pooled = (features * attention_weights).sum(dim=dim)

    return pooled
