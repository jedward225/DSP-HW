"""
Evaluation metrics for retrieval systems.
"""

from .retrieval_metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    map_at_k,
    ndcg_at_k,
    RetrievalMetrics,
    compute_all_metrics,
    aggregate_metrics,
)

from .bootstrap import (
    bootstrap_ci,
    bootstrap_ci_percentile,
    bootstrap_ci_bca,
    aggregate_metrics_with_ci,
    format_ci_string,
    compute_fold_ci,
)

__all__ = [
    # Retrieval metrics
    'hit_at_k',
    'precision_at_k',
    'recall_at_k',
    'mrr_at_k',
    'map_at_k',
    'ndcg_at_k',
    'RetrievalMetrics',
    'compute_all_metrics',
    'aggregate_metrics',
    # Bootstrap confidence intervals
    'bootstrap_ci',
    'bootstrap_ci_percentile',
    'bootstrap_ci_bca',
    'aggregate_metrics_with_ci',
    'format_ci_string',
    'compute_fold_ci',
]
