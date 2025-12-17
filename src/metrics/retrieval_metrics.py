"""
Retrieval evaluation metrics.

All metrics are computed using torch/numpy for GPU acceleration.
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


def hit_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10
) -> float:
    """
    Compute Hit@K (whether any of top-K results is relevant).

    Args:
        retrieved_labels: Labels of retrieved items, shape (n_retrieved,)
        query_label: Label of the query item
        k: Number of top results to consider

    Returns:
        1.0 if at least one relevant item in top-K, else 0.0
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    top_k = retrieved_labels[:k]
    return 1.0 if (top_k == query_label).any().item() else 0.0


def precision_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10
) -> float:
    """
    Compute Precision@K (fraction of top-K that are relevant).

    Args:
        retrieved_labels: Labels of retrieved items, shape (n_retrieved,)
        query_label: Label of the query item
        k: Number of top results to consider

    Returns:
        Precision value between 0 and 1
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    top_k = retrieved_labels[:k]
    relevant = (top_k == query_label).sum().item()
    return relevant / k


def recall_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10,
    num_relevant: Optional[int] = None
) -> float:
    """
    Compute Recall@K (fraction of relevant items retrieved in top-K).

    Args:
        retrieved_labels: Labels of retrieved items
        query_label: Label of the query item
        k: Number of top results to consider
        num_relevant: Total number of relevant items in gallery (if known)

    Returns:
        Recall value between 0 and 1
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    if num_relevant is None:
        num_relevant = (retrieved_labels == query_label).sum().item()

    if num_relevant == 0:
        return 0.0

    top_k = retrieved_labels[:k]
    relevant_retrieved = (top_k == query_label).sum().item()
    return relevant_retrieved / num_relevant


def mrr_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10
) -> float:
    """
    Compute Mean Reciprocal Rank at K.

    The reciprocal rank is 1/rank of the first relevant result.

    Args:
        retrieved_labels: Labels of retrieved items
        query_label: Label of the query item
        k: Maximum rank to consider

    Returns:
        Reciprocal rank (0 if no relevant item in top-K)
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    top_k = retrieved_labels[:k]
    relevant_mask = (top_k == query_label)

    if not relevant_mask.any():
        return 0.0

    # Find first relevant position (1-indexed)
    first_relevant = relevant_mask.nonzero(as_tuple=True)[0][0].item() + 1
    return 1.0 / first_relevant


def average_precision_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 20
) -> float:
    """
    Compute Average Precision at K.

    AP@K = (1/min(k, R)) * sum_{i=1}^{k} (P@i * rel_i)
    where R is total relevant items and rel_i indicates if item i is relevant.

    Args:
        retrieved_labels: Labels of retrieved items
        query_label: Label of the query item
        k: Maximum rank to consider

    Returns:
        Average precision value
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    top_k = retrieved_labels[:k]
    relevant_mask = (top_k == query_label)

    if not relevant_mask.any():
        return 0.0

    # Compute precision at each relevant position
    precisions = []
    relevant_count = 0

    for i in range(len(top_k)):
        if relevant_mask[i]:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precisions.append(precision)

    # Total relevant items (in the full gallery, not just top-k)
    total_relevant = (retrieved_labels == query_label).sum().item()

    # AP = sum(precisions) / min(k, total_relevant)
    return sum(precisions) / min(k, total_relevant)


def map_at_k(
    all_retrieved_labels: List[torch.Tensor],
    all_query_labels: List[Union[int, torch.Tensor]],
    k: int = 20
) -> float:
    """
    Compute Mean Average Precision at K.

    Args:
        all_retrieved_labels: List of retrieved label tensors for each query
        all_query_labels: List of query labels
        k: Maximum rank to consider

    Returns:
        Mean AP across all queries
    """
    aps = []
    for retrieved, query_label in zip(all_retrieved_labels, all_query_labels):
        ap = average_precision_at_k(retrieved, query_label, k)
        aps.append(ap)
    return np.mean(aps)


def dcg_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10
) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    DCG@K = sum_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        retrieved_labels: Labels of retrieved items
        query_label: Label of the query item
        k: Maximum rank to consider

    Returns:
        DCG value
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    top_k = retrieved_labels[:k]
    relevant_mask = (top_k == query_label).float()

    # Discount factors: 1/log2(i+2) for i in [0, k-1]
    positions = torch.arange(1, len(top_k) + 1, dtype=torch.float32, device=top_k.device)
    discounts = torch.log2(positions + 1)

    dcg = (relevant_mask / discounts).sum().item()
    return dcg


def ndcg_at_k(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    k: int = 10,
    num_relevant: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K
    where IDCG is the ideal DCG (all relevant items at top).

    Args:
        retrieved_labels: Labels of retrieved items
        query_label: Label of the query item
        k: Maximum rank to consider
        num_relevant: Total relevant items (for computing ideal DCG)

    Returns:
        NDCG value between 0 and 1
    """
    if isinstance(query_label, torch.Tensor):
        query_label = query_label.item()

    dcg = dcg_at_k(retrieved_labels, query_label, k)

    if dcg == 0:
        return 0.0

    # Compute IDCG (ideal DCG)
    if num_relevant is None:
        num_relevant = (retrieved_labels == query_label).sum().item()

    # Ideal: all relevant items at the top
    n_ideal = min(k, num_relevant)
    positions = torch.arange(1, n_ideal + 1, dtype=torch.float32)
    discounts = torch.log2(positions + 1)
    idcg = (1.0 / discounts).sum().item()

    if idcg == 0:
        return 0.0

    return dcg / idcg


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    hit_at_20: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    mrr_at_10: float = 0.0
    mrr_at_20: float = 0.0
    map_at_10: float = 0.0
    map_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'hit@1': self.hit_at_1,
            'hit@5': self.hit_at_5,
            'hit@10': self.hit_at_10,
            'hit@20': self.hit_at_20,
            'precision@1': self.precision_at_1,
            'precision@5': self.precision_at_5,
            'precision@10': self.precision_at_10,
            'precision@20': self.precision_at_20,
            'mrr@10': self.mrr_at_10,
            'mrr@20': self.mrr_at_20,
            'map@10': self.map_at_10,
            'map@20': self.map_at_20,
            'ndcg@10': self.ndcg_at_10,
            'ndcg@20': self.ndcg_at_20,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'RetrievalMetrics':
        """Create from dictionary."""
        return cls(
            hit_at_1=d.get('hit@1', 0.0),
            hit_at_5=d.get('hit@5', 0.0),
            hit_at_10=d.get('hit@10', 0.0),
            hit_at_20=d.get('hit@20', 0.0),
            precision_at_1=d.get('precision@1', 0.0),
            precision_at_5=d.get('precision@5', 0.0),
            precision_at_10=d.get('precision@10', 0.0),
            precision_at_20=d.get('precision@20', 0.0),
            mrr_at_10=d.get('mrr@10', 0.0),
            mrr_at_20=d.get('mrr@20', 0.0),
            map_at_10=d.get('map@10', 0.0),
            map_at_20=d.get('map@20', 0.0),
            ndcg_at_10=d.get('ndcg@10', 0.0),
            ndcg_at_20=d.get('ndcg@20', 0.0),
        )


def compute_all_metrics(
    retrieved_labels: torch.Tensor,
    query_label: Union[int, torch.Tensor],
    num_relevant: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_labels: Labels of retrieved items (sorted by similarity)
        query_label: Label of the query item
        num_relevant: Total number of relevant items in gallery

    Returns:
        Dictionary of all metrics
    """
    if num_relevant is None:
        num_relevant = (retrieved_labels == query_label).sum().item()

    return {
        'hit@1': hit_at_k(retrieved_labels, query_label, k=1),
        'hit@5': hit_at_k(retrieved_labels, query_label, k=5),
        'hit@10': hit_at_k(retrieved_labels, query_label, k=10),
        'hit@20': hit_at_k(retrieved_labels, query_label, k=20),
        'precision@1': precision_at_k(retrieved_labels, query_label, k=1),
        'precision@5': precision_at_k(retrieved_labels, query_label, k=5),
        'precision@10': precision_at_k(retrieved_labels, query_label, k=10),
        'precision@20': precision_at_k(retrieved_labels, query_label, k=20),
        'mrr@10': mrr_at_k(retrieved_labels, query_label, k=10),
        'mrr@20': mrr_at_k(retrieved_labels, query_label, k=20),
        'ap@10': average_precision_at_k(retrieved_labels, query_label, k=10),
        'ap@20': average_precision_at_k(retrieved_labels, query_label, k=20),
        'ndcg@10': ndcg_at_k(retrieved_labels, query_label, k=10, num_relevant=num_relevant),
        'ndcg@20': ndcg_at_k(retrieved_labels, query_label, k=20, num_relevant=num_relevant),
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_metrics: List of metric dictionaries for each query

    Returns:
        Dictionary with mean and std for each metric
    """
    if len(all_metrics) == 0:
        return {}

    metric_names = all_metrics[0].keys()
    result = {}

    for name in metric_names:
        values = [m[name] for m in all_metrics]
        result[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    # Also compute mAP from individual APs
    if 'ap@10' in metric_names:
        result['map@10'] = {
            'mean': np.mean([m['ap@10'] for m in all_metrics]),
            'std': np.std([m['ap@10'] for m in all_metrics]),
        }
    if 'ap@20' in metric_names:
        result['map@20'] = {
            'mean': np.mean([m['ap@20'] for m in all_metrics]),
            'std': np.std([m['ap@20'] for m in all_metrics]),
        }

    return result
