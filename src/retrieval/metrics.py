"""
Retrieval Evaluation Metrics

This module implements standard retrieval evaluation metrics for the sound retrieval task.

Metrics:
    - MRR@K (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant item
    - NDCG@K (Normalized Discounted Cumulative Gain): Measures ranking quality
    - Recall@K (Binary): Whether at least one relevant item is in top-K
    - Precision@K (Proportion): Proportion of relevant items in top-K

Usage:
    >>> from src.retrieval.metrics import evaluate_retrieval
    >>> results = evaluate_retrieval(query_labels, retrieved_labels, k_values=[10, 20])
"""

import numpy as np
from typing import List, Dict, Union, Optional


def mean_reciprocal_rank(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute Mean Reciprocal Rank at K (MRR@K).

    MRR@K = (1/Q) * Σ (1/rank_q)

    where rank_q is the rank of the first relevant item for query q.
    If no relevant item is found in top-K, reciprocal rank is 0.

    Parameters
    ----------
    query_labels : np.ndarray
        Ground truth labels for queries, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items, shape (n_queries, n_retrieved)
        Items should be sorted by similarity (most similar first)
    k : int
        Number of top results to consider

    Returns
    -------
    float
        MRR@K score in range [0, 1]

    Examples
    --------
    >>> query_labels = np.array([0, 1, 2])
    >>> retrieved_labels = np.array([
    ...     [1, 0, 2, 3, 4],  # Query 0: first match at rank 2
    ...     [1, 1, 0, 2, 3],  # Query 1: first match at rank 1
    ...     [0, 1, 2, 2, 3],  # Query 2: first match at rank 3
    ... ])
    >>> mrr = mean_reciprocal_rank(query_labels, retrieved_labels, k=5)
    >>> print(f"MRR@5: {mrr:.4f}")  # (1/2 + 1/1 + 1/3) / 3 = 0.6111
    """
    n_queries = len(query_labels)
    reciprocal_ranks = []

    for i in range(n_queries):
        query_label = query_labels[i]
        top_k_labels = retrieved_labels[i, :k]

        # Find rank of first relevant item (1-indexed)
        matches = np.where(top_k_labels == query_label)[0]

        if len(matches) > 0:
            # Rank is 1-indexed, so add 1
            first_rank = matches[0] + 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            # No relevant item in top-K
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def ndcg_at_k(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    NDCG@K = DCG@K / IDCG@K

    where:
        DCG@K = Σ (rel_i / log2(i + 1)) for i = 1 to K
        IDCG@K = ideal DCG (all relevant items ranked first)
        rel_i = 1 if item i is relevant (same class), 0 otherwise

    Parameters
    ----------
    query_labels : np.ndarray
        Ground truth labels for queries, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items, shape (n_queries, n_retrieved)
    k : int
        Number of top results to consider

    Returns
    -------
    float
        NDCG@K score in range [0, 1]
    """
    n_queries = len(query_labels)
    ndcg_scores = []

    # Precompute discount factors: 1/log2(i+1) for i=1,2,...,k
    # Using positions 1 to k (1-indexed)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)

    for i in range(n_queries):
        query_label = query_labels[i]
        top_k_labels = retrieved_labels[i, :k]

        # Binary relevance: 1 if same class, 0 otherwise
        relevances = (top_k_labels == query_label).astype(float)

        # DCG@K
        dcg = np.sum(relevances * discounts[:len(relevances)])

        # IDCG@K: ideal case where all relevant items are ranked first
        n_relevant = int(np.sum(relevances))
        if n_relevant == 0:
            ndcg_scores.append(0.0)
        else:
            # Ideal relevances: [1, 1, ..., 1, 0, 0, ..., 0]
            ideal_relevances = np.zeros(k)
            ideal_relevances[:n_relevant] = 1.0
            idcg = np.sum(ideal_relevances * discounts)

            ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores)


def recall_at_k(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k: int = 10,
    binary: bool = True
) -> float:
    """
    Compute Recall at K (Recall@K).

    Binary mode (default):
        Recall@K = 1 if at least one relevant item in top-K, else 0
        Final score is the proportion of queries with at least one hit.

    Proportion mode:
        Recall@K = (# relevant in top-K) / (total # relevant in database)
        Note: This requires knowing total relevant items per query.

    Parameters
    ----------
    query_labels : np.ndarray
        Ground truth labels for queries, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items, shape (n_queries, n_retrieved)
    k : int
        Number of top results to consider
    binary : bool
        If True, use binary recall (hit or miss)
        If False, use proportion recall

    Returns
    -------
    float
        Recall@K score in range [0, 1]

    Notes
    -----
    For ESC-50 with fold 5 as query (400 samples) and fold 1-4 as database (1600 samples):
    - Each class has 32 samples in the database (8 samples/fold * 4 folds)
    - Binary recall is recommended as it's more intuitive for single-query retrieval
    """
    n_queries = len(query_labels)

    if binary:
        # Binary recall: at least one relevant item in top-K
        hits = 0
        for i in range(n_queries):
            query_label = query_labels[i]
            top_k_labels = retrieved_labels[i, :k]

            if np.any(top_k_labels == query_label):
                hits += 1

        return hits / n_queries
    else:
        # Proportion recall: requires knowing total relevant per query
        # For ESC-50: each class has 32 items in database (fold 1-4)
        n_relevant_per_class = 32  # 8 samples/fold * 4 folds

        recall_scores = []
        for i in range(n_queries):
            query_label = query_labels[i]
            top_k_labels = retrieved_labels[i, :k]

            n_relevant_retrieved = np.sum(top_k_labels == query_label)
            recall_scores.append(n_relevant_retrieved / n_relevant_per_class)

        return np.mean(recall_scores)


def precision_at_k(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute Precision at K (Precision@K).

    Precision@K = (# relevant items in top-K) / K

    This measures the proportion of retrieved items that are relevant.

    Parameters
    ----------
    query_labels : np.ndarray
        Ground truth labels for queries, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items, shape (n_queries, n_retrieved)
    k : int
        Number of top results to consider

    Returns
    -------
    float
        Precision@K score in range [0, 1]

    Examples
    --------
    >>> query_labels = np.array([0, 1])
    >>> retrieved_labels = np.array([
    ...     [0, 0, 1, 2, 3],  # 2 relevant in top-5
    ...     [1, 2, 1, 1, 0],  # 3 relevant in top-5
    ... ])
    >>> prec = precision_at_k(query_labels, retrieved_labels, k=5)
    >>> print(f"Precision@5: {prec:.4f}")  # (2/5 + 3/5) / 2 = 0.5
    """
    n_queries = len(query_labels)
    precision_scores = []

    for i in range(n_queries):
        query_label = query_labels[i]
        top_k_labels = retrieved_labels[i, :k]

        n_relevant = np.sum(top_k_labels == query_label)
        precision_scores.append(n_relevant / k)

    return np.mean(precision_scores)


def evaluate_retrieval(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k_values: List[int] = [10, 20],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute all retrieval metrics for given k values.

    Parameters
    ----------
    query_labels : np.ndarray
        Ground truth labels for queries, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items, shape (n_queries, n_retrieved)
        Items should be sorted by similarity (most similar first)
    k_values : List[int]
        List of K values to evaluate
    verbose : bool
        If True, print results in a formatted table

    Returns
    -------
    Dict[str, float]
        Dictionary with metric names as keys:
        - 'MRR@K': Mean Reciprocal Rank
        - 'NDCG@K': Normalized Discounted Cumulative Gain
        - 'Recall@K': Binary Recall
        - 'Precision@K': Precision (proportion)

    Examples
    --------
    >>> query_labels = np.array([0, 1, 2, 3, 4])  # 5 queries
    >>> # Simulate retrieval results (most similar first)
    >>> retrieved_labels = np.random.randint(0, 50, (5, 100))
    >>> results = evaluate_retrieval(query_labels, retrieved_labels, k_values=[10, 20])
    """
    results = {}

    for k in k_values:
        results[f'MRR@{k}'] = mean_reciprocal_rank(query_labels, retrieved_labels, k)
        results[f'NDCG@{k}'] = ndcg_at_k(query_labels, retrieved_labels, k)
        results[f'Recall@{k}'] = recall_at_k(query_labels, retrieved_labels, k, binary=True)
        results[f'Precision@{k}'] = precision_at_k(query_labels, retrieved_labels, k)

    if verbose:
        print("\n" + "=" * 60)
        print("Retrieval Evaluation Results")
        print("=" * 60)
        print(f"{'Metric':<15} | " + " | ".join([f'K={k:<4}' for k in k_values]))
        print("-" * 60)

        for metric_name in ['MRR', 'NDCG', 'Recall', 'Precision']:
            values = [results[f'{metric_name}@{k}'] for k in k_values]
            formatted_values = " | ".join([f'{v:.4f}' for v in values])
            print(f"{metric_name:<15} | {formatted_values}")

        print("=" * 60)

    return results


def compute_retrieval_labels(
    query_indices: np.ndarray,
    database_indices: np.ndarray,
    similarity_matrix: np.ndarray,
    all_labels: np.ndarray,
    k: int = None
) -> tuple:
    """
    Convert similarity matrix to retrieved labels for evaluation.

    Parameters
    ----------
    query_indices : np.ndarray
        Indices of query samples in the full dataset
    database_indices : np.ndarray
        Indices of database samples in the full dataset
    similarity_matrix : np.ndarray
        Similarity scores, shape (n_queries, n_database)
        Higher values = more similar
    all_labels : np.ndarray
        Labels for all samples in the dataset
    k : int, optional
        Only return top-K results. If None, return all.

    Returns
    -------
    query_labels : np.ndarray
        Labels of query samples, shape (n_queries,)
    retrieved_labels : np.ndarray
        Labels of retrieved items sorted by similarity, shape (n_queries, n_database or k)
    """
    n_queries = len(query_indices)
    n_database = len(database_indices)

    # Get query labels
    query_labels = all_labels[query_indices]

    # Sort database items by similarity (descending)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)

    # Get labels of sorted items
    if k is not None:
        sorted_indices = sorted_indices[:, :k]

    # Map sorted indices back to database labels
    retrieved_labels = np.zeros_like(sorted_indices)
    for i in range(n_queries):
        for j in range(sorted_indices.shape[1]):
            db_idx = sorted_indices[i, j]
            retrieved_labels[i, j] = all_labels[database_indices[db_idx]]

    return query_labels, retrieved_labels


if __name__ == "__main__":
    # Unit tests for metrics
    print("=" * 70)
    print("Retrieval Metrics - Unit Tests")
    print("=" * 70)

    # Test 1: MRR with known values
    print("\n[Test 1] MRR@K")
    query_labels = np.array([0, 1, 2])
    retrieved_labels = np.array([
        [1, 0, 2, 3, 4],  # Query 0: first match at rank 2 -> RR = 1/2
        [1, 1, 0, 2, 3],  # Query 1: first match at rank 1 -> RR = 1
        [0, 1, 2, 2, 3],  # Query 2: first match at rank 3 -> RR = 1/3
    ])
    mrr = mean_reciprocal_rank(query_labels, retrieved_labels, k=5)
    expected_mrr = (0.5 + 1.0 + 1/3) / 3
    print(f"  Computed MRR@5: {mrr:.6f}")
    print(f"  Expected MRR@5: {expected_mrr:.6f}")
    print(f"  ✓ PASS" if abs(mrr - expected_mrr) < 1e-10 else "  ✗ FAIL")

    # Test 2: Recall@K (Binary)
    print("\n[Test 2] Recall@K (Binary)")
    query_labels = np.array([0, 1, 2, 3])
    retrieved_labels = np.array([
        [0, 1, 2, 3, 4],  # Hit at rank 1
        [2, 3, 4, 5, 1],  # Hit at rank 5
        [0, 1, 3, 4, 5],  # No hit in top-5
        [4, 5, 6, 3, 7],  # Hit at rank 4
    ])
    recall = recall_at_k(query_labels, retrieved_labels, k=5, binary=True)
    expected_recall = 3 / 4  # 3 queries have at least one hit
    print(f"  Computed Recall@5: {recall:.6f}")
    print(f"  Expected Recall@5: {expected_recall:.6f}")
    print(f"  ✓ PASS" if abs(recall - expected_recall) < 1e-10 else "  ✗ FAIL")

    # Test 3: Precision@K
    print("\n[Test 3] Precision@K")
    query_labels = np.array([0, 1])
    retrieved_labels = np.array([
        [0, 0, 1, 2, 3],  # 2 relevant in top-5 -> P = 2/5
        [1, 2, 1, 1, 0],  # 3 relevant in top-5 -> P = 3/5
    ])
    prec = precision_at_k(query_labels, retrieved_labels, k=5)
    expected_prec = (2/5 + 3/5) / 2
    print(f"  Computed Precision@5: {prec:.6f}")
    print(f"  Expected Precision@5: {expected_prec:.6f}")
    print(f"  ✓ PASS" if abs(prec - expected_prec) < 1e-10 else "  ✗ FAIL")

    # Test 4: NDCG@K
    print("\n[Test 4] NDCG@K")
    query_labels = np.array([0])
    retrieved_labels = np.array([
        [1, 0, 0, 1, 1],  # Relevance: [0, 1, 1, 0, 0]
    ])
    ndcg = ndcg_at_k(query_labels, retrieved_labels, k=5)
    # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
    #     = 0 + 0.6309 + 0.5 + 0 + 0 = 1.1309
    # IDCG (2 relevant items first) = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    # NDCG = 1.1309 / 1.6309 = 0.6934
    print(f"  Computed NDCG@5: {ndcg:.6f}")
    print(f"  Expected ~0.6934")
    print(f"  ✓ PASS" if 0.69 < ndcg < 0.70 else "  ✗ FAIL")

    # Test 5: Full evaluation
    print("\n[Test 5] Full Evaluation")
    np.random.seed(42)
    n_queries = 100
    n_database = 500
    n_classes = 50

    # Simulate retrieval: query labels and retrieved results
    query_labels = np.random.randint(0, n_classes, n_queries)

    # Create somewhat realistic retrieved labels
    # First few items have higher chance of being correct class
    retrieved_labels = np.zeros((n_queries, n_database), dtype=int)
    for i in range(n_queries):
        for j in range(n_database):
            if j < 10 and np.random.rand() < 0.3:
                retrieved_labels[i, j] = query_labels[i]
            else:
                retrieved_labels[i, j] = np.random.randint(0, n_classes)

    results = evaluate_retrieval(query_labels, retrieved_labels, k_values=[10, 20])

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
