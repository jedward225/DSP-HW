"""
Distance and Similarity Measures for Sound Retrieval

This module implements various distance/similarity measures for comparing
audio features in the retrieval task.

Methods:
    - Cosine Similarity: Fast, works well with high-dimensional features
    - Euclidean Distance: Standard L2 distance
    - DTW (Dynamic Time Warping): Handles temporal misalignment, slower but often better

Usage:
    >>> from src.retrieval.distances import cosine_similarity, dtw_distance
    >>> sim = cosine_similarity(query_features, database_features)
    >>> dist = dtw_distance(query_mfcc, database_mfcc)
"""

import numpy as np
from typing import Union, Optional, Callable
import numba
from numba import jit, prange


def cosine_similarity(
    query: np.ndarray,
    database: np.ndarray,
    flatten: bool = True
) -> np.ndarray:
    """
    Compute cosine similarity between query and database items.

    Cosine similarity = (A · B) / (||A|| × ||B||)

    Parameters
    ----------
    query : np.ndarray
        Query features, shape (n_features,) or (n_queries, n_features)
        For 2D features like MFCC (n_mfcc, n_frames), will be flattened if flatten=True
    database : np.ndarray
        Database features, shape (n_database, n_features) or (n_database, ...)
    flatten : bool
        If True, flatten features beyond first dimension before computing similarity

    Returns
    -------
    np.ndarray
        Similarity scores, shape (n_queries, n_database) or (n_database,)
        Higher values indicate more similar items

    Examples
    --------
    >>> query = np.random.randn(13, 44)  # Single MFCC query
    >>> database = np.random.randn(100, 13, 44)  # 100 database items
    >>> sim = cosine_similarity(query, database)
    >>> sim.shape  # (100,)
    """
    # Handle single query
    single_query = query.ndim == 1 or (query.ndim > 1 and flatten and query.ndim == database.ndim - 1)

    if flatten:
        # Flatten all dimensions except the first (batch) dimension
        if query.ndim > 1:
            if single_query:
                query = query.flatten()
            else:
                query = query.reshape(query.shape[0], -1)

        if database.ndim > 2:
            database = database.reshape(database.shape[0], -1)

    # Ensure 2D arrays
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Normalize vectors
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
    database_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-10)

    # Compute similarity matrix
    similarity = np.dot(query_norm, database_norm.T)

    if single_query:
        return similarity.squeeze(0)
    return similarity


def euclidean_distance(
    query: np.ndarray,
    database: np.ndarray,
    flatten: bool = True
) -> np.ndarray:
    """
    Compute Euclidean distance between query and database items.

    Euclidean distance = ||A - B||_2

    Parameters
    ----------
    query : np.ndarray
        Query features
    database : np.ndarray
        Database features
    flatten : bool
        If True, flatten features before computing distance

    Returns
    -------
    np.ndarray
        Distance scores (lower = more similar)
    """
    single_query = query.ndim == 1 or (query.ndim > 1 and flatten and query.ndim == database.ndim - 1)

    if flatten:
        if query.ndim > 1:
            if single_query:
                query = query.flatten()
            else:
                query = query.reshape(query.shape[0], -1)

        if database.ndim > 2:
            database = database.reshape(database.shape[0], -1)

    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Compute pairwise Euclidean distances efficiently
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a · b
    query_sq = np.sum(query ** 2, axis=1, keepdims=True)
    database_sq = np.sum(database ** 2, axis=1, keepdims=True)
    cross = np.dot(query, database.T)

    distances = np.sqrt(np.maximum(query_sq + database_sq.T - 2 * cross, 0))

    if single_query:
        return distances.squeeze(0)
    return distances


@jit(nopython=True)
def _dtw_distance_single(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute DTW distance between two sequences using Numba JIT.

    Parameters
    ----------
    x : np.ndarray
        First sequence, shape (n_features, n_frames_x)
    y : np.ndarray
        Second sequence, shape (n_features, n_frames_y)

    Returns
    -------
    float
        DTW distance (lower = more similar)
    """
    n_features, n_x = x.shape
    _, n_y = y.shape

    # Initialize cost matrix with infinity
    # Add 1 to dimensions for easier boundary handling
    dtw_matrix = np.full((n_x + 1, n_y + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the DTW matrix
    for i in range(1, n_x + 1):
        for j in range(1, n_y + 1):
            # Euclidean distance between frames
            cost = 0.0
            for k in range(n_features):
                diff = x[k, i - 1] - y[k, j - 1]
                cost += diff * diff
            cost = np.sqrt(cost)

            # DTW recurrence: min of three possible predecessors
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return dtw_matrix[n_x, n_y]


@jit(nopython=True)
def _dtw_distance_single_with_window(
    x: np.ndarray,
    y: np.ndarray,
    window: int
) -> float:
    """
    Compute DTW distance with Sakoe-Chiba band constraint for speedup.

    Parameters
    ----------
    x : np.ndarray
        First sequence, shape (n_features, n_frames_x)
    y : np.ndarray
        Second sequence, shape (n_features, n_frames_y)
    window : int
        Sakoe-Chiba band width (constraint on warping path)

    Returns
    -------
    float
        DTW distance
    """
    n_features, n_x = x.shape
    _, n_y = y.shape

    # Adjust window based on sequence length difference
    window = max(window, abs(n_x - n_y))

    # Initialize cost matrix
    dtw_matrix = np.full((n_x + 1, n_y + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the DTW matrix with window constraint
    for i in range(1, n_x + 1):
        j_start = max(1, i - window)
        j_end = min(n_y + 1, i + window + 1)

        for j in range(j_start, j_end):
            # Euclidean distance between frames
            cost = 0.0
            for k in range(n_features):
                diff = x[k, i - 1] - y[k, j - 1]
                cost += diff * diff
            cost = np.sqrt(cost)

            # DTW recurrence
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n_x, n_y]


def dtw_distance(
    query: np.ndarray,
    database: np.ndarray,
    window: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute DTW distance between query and database items.

    DTW aligns two sequences by warping them non-linearly in time,
    which is useful when audio clips have different speeds or durations.

    Parameters
    ----------
    query : np.ndarray
        Query features, shape (n_features, n_frames) for single query
        or (n_queries, n_features, n_frames) for batch
    database : np.ndarray
        Database features, shape (n_database, n_features, n_frames)
        Note: n_frames can vary across samples
    window : int, optional
        Sakoe-Chiba band width for speedup. If None, no constraint.
        Typical value: 10-20% of sequence length
    normalize : bool
        If True, normalize distance by path length

    Returns
    -------
    np.ndarray
        DTW distances, shape (n_database,) for single query
        or (n_queries, n_database) for batch

    Notes
    -----
    DTW has O(m*n) time complexity where m, n are sequence lengths.
    Using Sakoe-Chiba band reduces this to O(m*w) where w is window size.

    For ESC-50 with ~44 frames per clip and 1600 database items,
    full DTW can be slow. Consider:
    1. Using window constraint
    2. Pre-filtering with fast method (cosine) then DTW on top candidates
    """
    single_query = query.ndim == 2

    if single_query:
        query = query[np.newaxis, ...]

    n_queries = query.shape[0]
    n_database = database.shape[0]

    distances = np.zeros((n_queries, n_database))

    for i in range(n_queries):
        q = query[i].astype(np.float64)

        for j in range(n_database):
            d = database[j].astype(np.float64)

            if window is not None:
                dist = _dtw_distance_single_with_window(q, d, window)
            else:
                dist = _dtw_distance_single(q, d)

            if normalize:
                # Normalize by path length (approximated by sum of sequence lengths)
                path_length = q.shape[1] + d.shape[1]
                dist = dist / path_length

            distances[i, j] = dist

    if single_query:
        return distances.squeeze(0)
    return distances


def dtw_distance_batch(
    query: np.ndarray,
    database: np.ndarray,
    window: Optional[int] = None,
    normalize: bool = True,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Batch DTW computation with optional parallelization.

    This is a wrapper that can use multiprocessing for large-scale DTW.
    For small datasets, use dtw_distance directly.

    Parameters
    ----------
    query : np.ndarray
        Query features
    database : np.ndarray
        Database features
    window : int, optional
        Sakoe-Chiba band width
    normalize : bool
        Normalize by path length
    n_jobs : int
        Number of parallel jobs (1 = no parallelization)

    Returns
    -------
    np.ndarray
        DTW distances
    """
    if n_jobs == 1:
        return dtw_distance(query, database, window, normalize)

    # Parallel implementation using joblib or multiprocessing
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import os

    single_query = query.ndim == 2
    if single_query:
        query = query[np.newaxis, ...]

    n_queries = query.shape[0]
    n_database = database.shape[0]
    distances = np.zeros((n_queries, n_database))

    def compute_dtw_for_query(i):
        q = query[i].astype(np.float64)
        dists = np.zeros(n_database)
        for j in range(n_database):
            d = database[j].astype(np.float64)
            if window is not None:
                dist = _dtw_distance_single_with_window(q, d, window)
            else:
                dist = _dtw_distance_single(q, d)
            if normalize:
                path_length = q.shape[1] + d.shape[1]
                dist = dist / path_length
            dists[j] = dist
        return i, dists

    # Use ProcessPoolExecutor for CPU-bound DTW
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(compute_dtw_for_query, range(n_queries)))

    for i, dists in results:
        distances[i] = dists

    if single_query:
        return distances.squeeze(0)
    return distances


def compute_similarity_matrix(
    query_features: np.ndarray,
    database_features: np.ndarray,
    method: str = 'cosine',
    **kwargs
) -> np.ndarray:
    """
    Compute similarity matrix using specified method.

    Parameters
    ----------
    query_features : np.ndarray
        Query features
    database_features : np.ndarray
        Database features
    method : str
        Distance/similarity method: 'cosine', 'euclidean', 'dtw'
    **kwargs
        Additional arguments for specific methods (e.g., window for DTW)

    Returns
    -------
    np.ndarray
        Similarity matrix, shape (n_queries, n_database)
        Higher values indicate more similar items
        (distances are converted to similarities)
    """
    method = method.lower()

    if method == 'cosine':
        return cosine_similarity(query_features, database_features, **kwargs)

    elif method == 'euclidean':
        distances = euclidean_distance(query_features, database_features, **kwargs)
        # Convert distance to similarity: sim = 1 / (1 + dist)
        return 1.0 / (1.0 + distances)

    elif method == 'dtw':
        distances = dtw_distance(query_features, database_features, **kwargs)
        # Convert distance to similarity
        return 1.0 / (1.0 + distances)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'cosine', 'euclidean', 'dtw'")


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("Distance/Similarity Measures - Unit Tests")
    print("=" * 70)

    # Test 1: Cosine Similarity
    print("\n[Test 1] Cosine Similarity")
    np.random.seed(42)

    # Create test data
    query = np.random.randn(13, 44)  # Single MFCC-like query
    database = np.random.randn(100, 13, 44)  # 100 database items

    sim = cosine_similarity(query, database)
    print(f"  Query shape: {query.shape}")
    print(f"  Database shape: {database.shape}")
    print(f"  Similarity shape: {sim.shape}")
    print(f"  Similarity range: [{sim.min():.4f}, {sim.max():.4f}]")

    # Test: same vector should have similarity 1
    test_sim = cosine_similarity(query, query[np.newaxis, ...])
    print(f"  Self-similarity: {test_sim[0]:.6f} (should be 1.0)")
    print(f"  ✓ PASS" if abs(test_sim[0] - 1.0) < 1e-6 else "  ✗ FAIL")

    # Test 2: Euclidean Distance
    print("\n[Test 2] Euclidean Distance")
    dist = euclidean_distance(query, database)
    print(f"  Distance shape: {dist.shape}")
    print(f"  Distance range: [{dist.min():.4f}, {dist.max():.4f}]")

    # Test: same vector should have distance 0
    test_dist = euclidean_distance(query, query[np.newaxis, ...])
    print(f"  Self-distance: {test_dist[0]:.6f} (should be 0.0)")
    print(f"  ✓ PASS" if abs(test_dist[0]) < 1e-6 else "  ✗ FAIL")

    # Test 3: DTW Distance (small test)
    print("\n[Test 3] DTW Distance")

    # Small test case for verification
    query_small = np.random.randn(13, 20)
    database_small = np.random.randn(10, 13, 20)

    # Warm up JIT
    _ = dtw_distance(query_small, database_small[:2])

    start = time.time()
    dtw_dist = dtw_distance(query_small, database_small)
    elapsed = time.time() - start

    print(f"  Query shape: {query_small.shape}")
    print(f"  Database shape: {database_small.shape}")
    print(f"  DTW distance shape: {dtw_dist.shape}")
    print(f"  DTW distance range: [{dtw_dist.min():.4f}, {dtw_dist.max():.4f}]")
    print(f"  Time for 10 items: {elapsed*1000:.2f} ms")

    # Test: same sequence should have DTW distance ~0
    test_dtw = dtw_distance(query_small, query_small[np.newaxis, ...])
    print(f"  Self DTW distance: {test_dtw[0]:.6f} (should be ~0)")
    print(f"  ✓ PASS" if test_dtw[0] < 0.1 else "  ✗ FAIL")

    # Test 4: DTW with window constraint
    print("\n[Test 4] DTW with Sakoe-Chiba Window")
    start = time.time()
    dtw_dist_window = dtw_distance(query_small, database_small, window=5)
    elapsed_window = time.time() - start

    print(f"  DTW (window=5) time: {elapsed_window*1000:.2f} ms")
    print(f"  Speedup: {elapsed/elapsed_window:.2f}x")

    # Test 5: Compute similarity matrix
    print("\n[Test 5] Similarity Matrix")
    for method in ['cosine', 'euclidean', 'dtw']:
        if method == 'dtw':
            sim_matrix = compute_similarity_matrix(
                query_small, database_small, method=method, window=5
            )
        else:
            sim_matrix = compute_similarity_matrix(
                query, database, method=method
            )
        print(f"  {method:>10s}: shape={sim_matrix.shape}, range=[{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")

    # Test 6: Batch query
    print("\n[Test 6] Batch Query")
    batch_query = np.random.randn(5, 13, 44)  # 5 queries
    batch_sim = cosine_similarity(batch_query, database)
    print(f"  Batch query shape: {batch_query.shape}")
    print(f"  Batch similarity shape: {batch_sim.shape}")
    print(f"  ✓ PASS" if batch_sim.shape == (5, 100) else "  ✗ FAIL")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
