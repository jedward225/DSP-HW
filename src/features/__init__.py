"""
Feature extraction module.

All feature extraction is implemented using torch/numpy only.
For librosa/scipy operations, use dsp_core module.
"""

from .spectral import (
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    spectral_flux,
    spectral_contrast,
    zero_crossing_rate,
    rms_energy,
    extract_spectral_features,
)

from .delta import (
    compute_delta,
    compute_delta_delta,
    add_deltas,
)

from .pooling import (
    global_mean_pool,
    global_std_pool,
    global_max_pool,
    global_min_pool,
    mean_std_pool,
    statistics_pool,
)

__all__ = [
    # Spectral features
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'spectral_flatness',
    'spectral_flux',
    'spectral_contrast',
    'zero_crossing_rate',
    'rms_energy',
    'extract_spectral_features',
    # Delta features
    'compute_delta',
    'compute_delta_delta',
    'add_deltas',
    # Pooling
    'global_mean_pool',
    'global_std_pool',
    'global_max_pool',
    'global_min_pool',
    'mean_std_pool',
    'statistics_pool',
]
