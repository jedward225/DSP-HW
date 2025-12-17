"""
Hybrid CLAP+MFCC retrieval method (M9).

Combines deep learning embeddings (CLAP) with traditional features (MFCC)
through late fusion for robust retrieval.
"""

import torch
import numpy as np
from typing import Optional, List, Dict

from src.retrieval.base import BaseRetriever
from src.retrieval.clap_retriever import CLAPRetriever
from src.retrieval.pool_retriever import create_method_m1


class HybridRetriever(BaseRetriever):
    """
    M9: Hybrid CLAP + MFCC retrieval.

    Combines CLAP embeddings (semantic, deep learning) with MFCC features
    (acoustic, traditional) through late fusion:
        d_hybrid = α * d_CLAP + β * d_MFCC

    This hybrid approach leverages:
    - CLAP: Semantic understanding from large-scale audio-text pretraining
    - MFCC: Fine-grained acoustic characteristics

    The combination often outperforms either method alone.
    """

    def __init__(
        self,
        name: str = "M9_Hybrid",
        device: Optional[str] = None,
        sr: int = 22050,
        clap_weight: float = 0.7,
        mfcc_weight: float = 0.3,
        clap_checkpoint: str = None,
        enable_fusion: bool = False,
        clap_amodel: str = 'HTSAT-base',
        n_mfcc: int = 20,
        n_mels: int = 128,
    ):
        """
        Initialize hybrid retriever.

        Args:
            name: Method name
            device: Device for computation
            sr: Sample rate for MFCC extraction (CLAP resamples to 48kHz internally)
            clap_weight: Weight for CLAP distance (α)
            mfcc_weight: Weight for MFCC distance (β)
            clap_checkpoint: Path to CLAP checkpoint
            enable_fusion: Enable fusion CLAP model
            clap_amodel: CLAP audio model architecture
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel filter banks
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(name=name, device=device, sr=sr)

        # Normalize weights
        total = clap_weight + mfcc_weight
        self.clap_weight = clap_weight / total
        self.mfcc_weight = mfcc_weight / total

        # Create CLAP retriever
        # Note: We pass sr=sr (dataset SR, e.g. 22050), and CLAPRetriever
        # will handle resampling to its internal 48kHz requirement
        self.clap_retriever = CLAPRetriever(
            name="Hybrid_CLAP",
            device=device,
            sr=sr,  # Pass dataset SR, let CLAP resample internally
            checkpoint_path=clap_checkpoint,
            enable_fusion=enable_fusion,
            amodel=clap_amodel,
        )

        # Create MFCC retriever (M1)
        self.mfcc_retriever = create_method_m1(
            device=device,
            sr=sr,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
        )

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Not used directly - each sub-retriever extracts its own features."""
        raise NotImplementedError("HybridRetriever uses sub-retriever features")

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """Build gallery for both CLAP and MFCC retrievers."""
        self._gallery_labels = []
        self._gallery_indices = []

        # Build CLAP gallery
        self.clap_retriever.build_gallery(gallery_samples, show_progress=show_progress)

        # Build MFCC gallery (no progress bar to avoid duplicate)
        self.mfcc_retriever.build_gallery(gallery_samples, show_progress=False)

        # Store labels from MFCC retriever (same as CLAP)
        self._gallery_labels = self.mfcc_retriever._gallery_labels
        self._gallery_indices = self.mfcc_retriever._gallery_indices

    def clear_gallery(self):
        """Clear gallery for both retrievers."""
        self.clap_retriever.clear_gallery()
        self.mfcc_retriever.clear_gallery()
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
        Retrieve using weighted late fusion of CLAP and MFCC distances.

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

        # Get CLAP sorted indices and distances
        clap_sorted_indices, clap_sorted_distances = self.clap_retriever.retrieve(
            query_waveform, k=None, return_distances=True, query_sr=query_sr
        )
        clap_sorted_indices = clap_sorted_indices.to(fusion_device)
        clap_sorted_distances = clap_sorted_distances.to(device=fusion_device, dtype=fusion_dtype)
        # Reconstruct full distance array indexed by gallery position
        clap_distances = torch.zeros(n_gallery, dtype=fusion_dtype, device=fusion_device)
        clap_distances[clap_sorted_indices] = clap_sorted_distances

        # Get MFCC sorted indices and distances
        mfcc_sorted_indices, mfcc_sorted_distances = self.mfcc_retriever.retrieve(
            query_waveform, k=None, return_distances=True, query_sr=query_sr
        )
        mfcc_sorted_indices = mfcc_sorted_indices.to(fusion_device)
        mfcc_sorted_distances = mfcc_sorted_distances.to(device=fusion_device, dtype=fusion_dtype)
        # Reconstruct full distance array indexed by gallery position
        mfcc_distances = torch.zeros(n_gallery, dtype=fusion_dtype, device=fusion_device)
        mfcc_distances[mfcc_sorted_indices] = mfcc_sorted_distances

        # Normalize distances to [0, 1]
        def normalize(dist):
            d_min, d_max = dist.min(), dist.max()
            if d_max - d_min > 1e-10:
                return (dist - d_min) / (d_max - d_min)
            return torch.zeros_like(dist)

        clap_norm = normalize(clap_distances)
        mfcc_norm = normalize(mfcc_distances)

        # Weighted fusion
        fused_distance = self.clap_weight * clap_norm + self.mfcc_weight * mfcc_norm

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

    def to(self, device: str) -> 'HybridRetriever':
        """Move retriever to device."""
        self.device = device
        self.clap_retriever.to(device)
        self.mfcc_retriever.to(device)
        if self._gallery_labels is not None:
            self._gallery_labels = self._gallery_labels.to(device)
        return self

    @property
    def gallery_size(self) -> int:
        """Return the number of items in the gallery."""
        return self.mfcc_retriever.gallery_size


def create_method_m9(
    device: Optional[str] = None,
    sr: int = 22050,
    clap_weight: float = 0.7,
    mfcc_weight: float = 0.3,
    clap_checkpoint: str = None,
    **kwargs
) -> HybridRetriever:
    """
    Create M9: Hybrid CLAP + MFCC retrieval.

    Combines semantic CLAP embeddings with acoustic MFCC features.

    Args:
        device: Computation device
        sr: Sample rate for MFCC extraction
        clap_weight: Weight for CLAP distance (default: 0.7)
        mfcc_weight: Weight for MFCC distance (default: 0.3)
        clap_checkpoint: Path to CLAP checkpoint (uses default if None)
        **kwargs: Additional arguments

    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        name="M9_Hybrid_CLAP_MFCC",
        device=device,
        sr=sr,
        clap_weight=clap_weight,
        mfcc_weight=mfcc_weight,
        clap_checkpoint=clap_checkpoint,
    )
