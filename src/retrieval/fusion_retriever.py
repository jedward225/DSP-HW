"""
Multi-Feature Fusion Retrieval Methods.

This module provides:
  1. LateFusionRetriever: Weighted combination of distances
  2. RankFusionRetriever: Reciprocal Rank Fusion (RRF)
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple

from src.retrieval.base import BaseRetriever


class LateFusionRetriever(BaseRetriever):
    """
    Late Fusion Retriever.

    Combines multiple retrievers by weighted combination of their distances:
        d_fusion = α*d_1 + β*d_2 + γ*d_3 + ...

    The weights should sum to 1.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: List[float] = None,
        name: str = "LateFusionRetriever",
        device: str = 'cpu',
        sr: int = 22050,
    ):
        """
        Initialize late fusion retriever.

        Args:
            retrievers: List of base retrievers to combine
            weights: Weights for each retriever (default: equal weights)
            name: Method name
            device: Computation device
            sr: Sample rate
        """
        super().__init__(name=name, device=device, sr=sr)

        self.retrievers = retrievers
        self.n_retrievers = len(retrievers)

        if weights is None:
            # Equal weights
            self.weights = [1.0 / self.n_retrievers] * self.n_retrievers
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Not used directly - each sub-retriever extracts its own features."""
        raise NotImplementedError("LateFusionRetriever uses sub-retriever features")

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """Build gallery for all sub-retrievers."""
        self._gallery_labels = []
        self._gallery_indices = []

        for retriever in self.retrievers:
            retriever.build_gallery(gallery_samples, show_progress=False)

        # Store labels from first retriever
        if self.retrievers:
            self._gallery_labels = self.retrievers[0]._gallery_labels.to(self.device)
            self._gallery_indices = self.retrievers[0]._gallery_indices

    def clear_gallery(self):
        """Clear gallery for all sub-retrievers."""
        for retriever in self.retrievers:
            retriever.clear_gallery()
        self._gallery_labels = None
        self._gallery_indices = None

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        query_sr: int = None,
    ) -> torch.Tensor:
        """
        Retrieve using weighted distance fusion.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return
            return_distances: If True, also return fused distances

        Returns:
            Indices of top-k results (and optionally distances)
        """
        if self._gallery_labels is None:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        n_gallery = len(self._gallery_labels)
        fusion_device = torch.device(self.device)
        fusion_dtype = torch.float32
        all_distances = []

        for retriever in self.retrievers:
            # Get retriever's sorted indices and distances
            sorted_indices, sorted_distances = retriever.retrieve(
                query_waveform, k=None, return_distances=True, query_sr=query_sr
            )

            sorted_indices = sorted_indices.to(fusion_device)
            sorted_distances = sorted_distances.to(device=fusion_device, dtype=fusion_dtype)

            # Reconstruct full distance array indexed by gallery position
            # (sorted_distances are in rank order, we need gallery order)
            full_dist = torch.zeros(n_gallery, dtype=fusion_dtype, device=fusion_device)
            full_dist[sorted_indices] = sorted_distances
            all_distances.append(full_dist)

        # Normalize each distance array to [0, 1]
        normalized_distances = []
        for dist in all_distances:
            d_min = dist.min()
            d_max = dist.max()
            if d_max - d_min > 1e-10:
                d_norm = (dist - d_min) / (d_max - d_min)
            else:
                d_norm = torch.zeros_like(dist)
            normalized_distances.append(d_norm)

        # Weighted combination
        fused_distance = torch.zeros_like(normalized_distances[0])
        for w, d in zip(self.weights, normalized_distances):
            fused_distance += w * d

        # Sort by fused distance
        sorted_indices = torch.argsort(fused_distance)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            sorted_distances = fused_distance[sorted_indices]
            return sorted_indices, sorted_distances

        return sorted_indices

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """Not used directly."""
        raise NotImplementedError("Use retrieve() method instead")


class RankFusionRetriever(BaseRetriever):
    """
    Rank Fusion Retriever (Reciprocal Rank Fusion - RRF).

    Combines rankings from multiple retrievers using RRF:
        score(d) = Σ 1/(k + rank_i(d)) for each retriever i

    Higher score = better (closer to query).
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        rrf_k: int = 60,
        name: str = "RankFusionRetriever",
        device: str = 'cpu',
        sr: int = 22050,
    ):
        """
        Initialize rank fusion retriever.

        Args:
            retrievers: List of base retrievers to combine
            rrf_k: RRF constant (higher = more weight to lower ranks)
            name: Method name
            device: Computation device
            sr: Sample rate
        """
        super().__init__(name=name, device=device, sr=sr)

        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Not used directly."""
        raise NotImplementedError("RankFusionRetriever uses sub-retriever features")

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """Build gallery for all sub-retrievers."""
        for retriever in self.retrievers:
            retriever.build_gallery(gallery_samples, show_progress=False)

        if self.retrievers:
            self._gallery_labels = self.retrievers[0]._gallery_labels.to(self.device)
            self._gallery_indices = self.retrievers[0]._gallery_indices

    def clear_gallery(self):
        """Clear gallery for all sub-retrievers."""
        for retriever in self.retrievers:
            retriever.clear_gallery()
        self._gallery_labels = None
        self._gallery_indices = None

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        query_sr: int = None,
    ) -> torch.Tensor:
        """
        Retrieve using reciprocal rank fusion.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return
            return_distances: If True, return negative RRF scores as "distances"

        Returns:
            Indices of top-k results
        """
        if self._gallery_labels is None:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        n_gallery = len(self._gallery_labels)
        fusion_device = torch.device(self.device)

        # Get rankings from each retriever
        all_ranks = []
        for retriever in self.retrievers:
            # Get full ranking
            indices = retriever.retrieve(query_waveform, k=None, return_distances=False, query_sr=query_sr)
            indices = indices.to(fusion_device)
            # Convert to rank array
            ranks = torch.empty(n_gallery, dtype=torch.long, device=fusion_device)
            ranks[indices] = torch.arange(n_gallery, device=fusion_device)
            all_ranks.append(ranks)

        # Compute RRF scores
        rrf_scores = torch.zeros(n_gallery, dtype=torch.float32, device=fusion_device)
        for ranks in all_ranks:
            rrf_scores += 1.0 / (self.rrf_k + ranks.float())

        # Sort by RRF score (descending - higher is better)
        sorted_indices = torch.argsort(rrf_scores, descending=True)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            # Return negative RRF scores as "distances" (lower is better)
            sorted_scores = -rrf_scores[sorted_indices]
            return sorted_indices, sorted_scores

        return sorted_indices

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """Not used directly."""
        raise NotImplementedError("Use retrieve() method instead")


def create_late_fusion(
    retrievers: List[BaseRetriever],
    weights: List[float] = None,
    **kwargs
) -> LateFusionRetriever:
    """
    Create a late fusion retriever.

    Args:
        retrievers: List of base retrievers
        weights: Optional weights for each retriever

    Returns:
        LateFusionRetriever instance
    """
    return LateFusionRetriever(
        retrievers=retrievers,
        weights=weights,
        name="LateFusion",
        **kwargs
    )


def create_rank_fusion(
    retrievers: List[BaseRetriever],
    rrf_k: int = 60,
    **kwargs
) -> RankFusionRetriever:
    """
    Create a rank fusion retriever.

    Args:
        retrievers: List of base retrievers
        rrf_k: RRF constant

    Returns:
        RankFusionRetriever instance
    """
    return RankFusionRetriever(
        retrievers=retrievers,
        rrf_k=rrf_k,
        name="RankFusion",
        **kwargs
    )
