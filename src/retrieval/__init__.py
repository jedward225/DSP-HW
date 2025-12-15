"""
Sound Retrieval Module

This module provides tools for audio-based retrieval experiments:
- metrics: Evaluation metrics (MRR, NDCG, Recall, Precision)
- distances: Distance/similarity measures (Cosine, Euclidean, DTW)
- features: Feature extraction wrappers
- retrieval: Main retrieval logic
"""

from .metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    precision_at_k,
    evaluate_retrieval
)

from .distances import (
    cosine_similarity,
    euclidean_distance,
    dtw_distance,
    compute_similarity_matrix
)

from .features import (
    extract_features,
    FeatureExtractor,
    aggregate_features
)

from .retrieval import (
    SoundRetrieval,
    run_grid_search
)

__all__ = [
    # Metrics
    'mean_reciprocal_rank',
    'ndcg_at_k',
    'recall_at_k',
    'precision_at_k',
    'evaluate_retrieval',
    # Distances
    'cosine_similarity',
    'euclidean_distance',
    'dtw_distance',
    'compute_similarity_matrix',
    # Features
    'extract_features',
    'FeatureExtractor',
    'aggregate_features',
    # Retrieval
    'SoundRetrieval',
    'run_grid_search'
]
