"""
Two-Stage Retrieval Method.

Stage 1: Fast coarse recall using global embeddings (M1/M3) → Top-N candidates
Stage 2: Fine re-ranking using DTW or other expensive methods → Top-K results
"""

import torch
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.base import BaseRetriever


class TwoStageRetriever(BaseRetriever):
    """
    Two-stage retrieval: Coarse recall + Fine re-ranking.

    This approach balances accuracy and efficiency:
    - Stage 1: Fast method (M1/M3) retrieves top-N candidates
    - Stage 2: Expensive method (M5/DTW) re-ranks candidates
    """

    def __init__(
        self,
        coarse_retriever: BaseRetriever,
        fine_retriever: BaseRetriever,
        top_n: int = 100,
        name: str = "TwoStageRetriever",
        device: str = 'cpu',
        sr: int = 22050,
    ):
        """
        Initialize two-stage retriever.

        Args:
            coarse_retriever: Fast retriever for stage 1 (e.g., M1, M3)
            fine_retriever: Accurate retriever for stage 2 (e.g., M5)
            top_n: Number of candidates from stage 1
            name: Method name
            device: Computation device
            sr: Sample rate
        """
        super().__init__(name=name, device=device, sr=sr)

        self.coarse_retriever = coarse_retriever
        self.fine_retriever = fine_retriever
        self.top_n = top_n

        # Store gallery samples for fine retriever access
        self._gallery_samples: Optional[List[dict]] = None

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Extract features using coarse retriever."""
        return self.coarse_retriever.extract_features(waveform, sr)

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """
        Build gallery for both retrievers.

        Note: For efficiency, fine retriever builds gallery lazily
        only for candidates during retrieval.
        """
        # Store samples for later access
        self._gallery_samples = gallery_samples

        # Build coarse gallery (always needed)
        self.coarse_retriever.build_gallery(gallery_samples, show_progress=show_progress)

        # Store labels and indices from coarse retriever
        self._gallery_labels = self.coarse_retriever._gallery_labels
        self._gallery_indices = self.coarse_retriever._gallery_indices

    def clear_gallery(self):
        """Clear galleries for both retrievers."""
        self.coarse_retriever.clear_gallery()
        self.fine_retriever.clear_gallery()
        self._gallery_samples = None
        self._gallery_labels = None
        self._gallery_indices = None

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False
    ) -> torch.Tensor:
        """
        Two-stage retrieval.

        Stage 1: Get top-N candidates from coarse retriever
        Stage 2: Re-rank candidates using fine retriever

        Args:
            query_waveform: Query audio waveform
            k: Final number of results to return
            return_distances: If True, return fine stage distances

        Returns:
            Indices of top-k results (after re-ranking)
        """
        if self._gallery_samples is None:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        # Stage 1: Coarse recall
        coarse_indices = self.coarse_retriever.retrieve(
            query_waveform, k=self.top_n, return_distances=False
        )

        # If top_n >= gallery size, just use coarse results
        if self.top_n >= len(self._gallery_samples):
            if k is not None:
                coarse_indices = coarse_indices[:k]
            if return_distances:
                _, coarse_dist = self.coarse_retriever.retrieve(
                    query_waveform, k=len(coarse_indices), return_distances=True
                )
                return coarse_indices, coarse_dist
            return coarse_indices

        # Stage 2: Build mini-gallery for fine retriever
        candidate_samples = [self._gallery_samples[idx] for idx in coarse_indices.cpu().numpy()]

        # Build fine gallery with candidates only
        self.fine_retriever.clear_gallery()
        self.fine_retriever.build_gallery(candidate_samples, show_progress=False)

        # Re-rank using fine retriever
        if return_distances:
            fine_indices, fine_distances = self.fine_retriever.retrieve(
                query_waveform, k=k, return_distances=True
            )
        else:
            fine_indices = self.fine_retriever.retrieve(
                query_waveform, k=k, return_distances=False
            )

        # Map back to original gallery indices
        original_indices = coarse_indices[fine_indices]

        if return_distances:
            return original_indices, fine_distances

        return original_indices

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """Not used directly - delegated to sub-retrievers."""
        raise NotImplementedError("Use retrieve() method instead")


def create_twostage_retriever(
    coarse_retriever: BaseRetriever,
    fine_retriever: BaseRetriever,
    top_n: int = 100,
    **kwargs
) -> TwoStageRetriever:
    """
    Create a two-stage retriever.

    Args:
        coarse_retriever: Fast retriever for stage 1
        fine_retriever: Accurate retriever for stage 2
        top_n: Number of candidates from stage 1

    Returns:
        TwoStageRetriever instance
    """
    return TwoStageRetriever(
        coarse_retriever=coarse_retriever,
        fine_retriever=fine_retriever,
        top_n=top_n,
        name=f"TwoStage_N{top_n}",
        device=coarse_retriever.device,
        sr=coarse_retriever.sr,
    )
