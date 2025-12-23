"""
Bootstrap confidence interval calculation for evaluation metrics.

Provides robust confidence intervals for 5-fold cross-validation results.

Note on Architecture Convention:
    This module is an exception to the "only dsp_core imports librosa/scipy" rule.
    The BCa bootstrap method imports scipy.stats for normal distribution functions.
    This exception is documented because metrics is a utility module separate from
    the core audio processing pipeline.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: List of metric values (e.g., from 5 folds)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    # Use local RNG to avoid modifying global state
    rng = np.random.default_rng(random_state)

    values = np.array(values)
    n = len(values)

    # Generate bootstrap samples
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_means[i] = values[sample_idx].mean()

    # Compute percentiles
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    mean = values.mean()

    return mean, ci_lower, ci_upper


def bootstrap_ci_percentile(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap percentile confidence interval.

    This is the basic percentile method, suitable for most cases.
    """
    return bootstrap_ci(values, n_bootstrap, confidence, random_state)


def bootstrap_ci_bca(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa provides better coverage for skewed distributions.

    Args:
        values: List of metric values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    from scipy import stats

    # Use local RNG to avoid modifying global state
    rng = np.random.default_rng(random_state)

    values = np.array(values)
    n = len(values)
    mean = values.mean()

    # Generate bootstrap samples
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_means[i] = values[sample_idx].mean()

    # Bias correction factor
    # Clamp proportion to avoid inf from norm.ppf(0) or norm.ppf(1)
    prop = (bootstrap_means < mean).mean()
    prop = np.clip(prop, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop)

    # Acceleration factor (jackknife estimate)
    jackknife_means = np.zeros(n)
    for i in range(n):
        jackknife_means[i] = np.delete(values, i).mean()
    jack_mean = jackknife_means.mean()
    num = ((jack_mean - jackknife_means) ** 3).sum()
    denom = 6 * (((jack_mean - jackknife_means) ** 2).sum() ** 1.5)
    a = num / denom if denom != 0 else 0

    # Adjust percentiles
    alpha = 1 - confidence
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa adjustment
    def adjust_percentile(z_alpha):
        return stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

    p_lower = adjust_percentile(z_alpha_lower) * 100
    p_upper = adjust_percentile(z_alpha_upper) * 100

    # Clip to valid range
    p_lower = max(0, min(100, p_lower))
    p_upper = max(0, min(100, p_upper))

    ci_lower = np.percentile(bootstrap_means, p_lower)
    ci_upper = np.percentile(bootstrap_means, p_upper)

    return mean, ci_lower, ci_upper


def aggregate_metrics_with_ci(
    all_metrics: List[Dict[str, float]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    method: str = 'percentile',
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple folds with bootstrap confidence intervals.

    Args:
        all_metrics: List of metric dictionaries from each fold
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        method: 'percentile' or 'bca'
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with mean, ci_lower, ci_upper for each metric
    """
    if not all_metrics:
        return {}

    # Get all metric names
    metric_names = list(all_metrics[0].keys())

    # Select bootstrap method
    if method == 'bca':
        ci_func = bootstrap_ci_bca
    else:
        ci_func = bootstrap_ci_percentile

    # Compute CI for each metric
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean, ci_lower, ci_upper = ci_func(
            values, n_bootstrap, confidence, random_state=random_state
        )
        aggregated[name] = {
            'mean': mean,
            'std': np.std(values),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
        }

    return aggregated


def format_ci_string(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format confidence interval as string for reporting."""
    return f"{mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"


def compute_fold_ci(
    fold_results: Dict[str, float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, str]:
    """
    Compute and format CIs for a set of fold results.

    Args:
        fold_results: Dict mapping fold names to metric values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict with formatted CI strings
    """
    values = list(fold_results.values())
    mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap, confidence, random_state=random_state)

    return {
        'mean': mean,
        'std': np.std(values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'formatted': format_ci_string(mean, ci_lower, ci_upper),
    }
