"""
Multi-resolution retrieval method (M7).

This method extracts features at multiple time-frequency resolutions
and combines them through late fusion.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict

from src.retrieval.base import BaseRetriever
from src.retrieval.pool_retriever import PoolRetriever
from src.dsp_core import mfcc, log_melspectrogram
from src.features.pooling import mean_std_pool


class MultiResRetriever(BaseRetriever):
    """
    Multi-resolution retriever combining short and long window analysis.

    Different window sizes capture different aspects:
    - Short windows (25ms): High time resolution, captures transients
    - Long windows (80ms): High frequency resolution, captures tonal content
    """

    def __init__(
        self,
        name: str = "MultiResRetriever",
        device: str = 'cpu',
        sr: int = 22050,
        # Short window parameters
        short_n_fft: int = 551,
        short_hop_length: int = 220,
        # Long window parameters
        long_n_fft: int = 1764,
        long_hop_length: int = 441,
        # Feature parameters
        n_mfcc: int = 20,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = None,
        # Fusion parameters
        fusion_weights: Tuple[float, float] = (0.5, 0.5),
        distance: str = 'cosine',
    ):
        """
        Initialize multi-resolution retriever.

        Args:
            name: Method name
            device: Computation device
            sr: Sample rate
            short_n_fft: FFT size for short window
            short_hop_length: Hop length for short window
            long_n_fft: FFT size for long window
            long_hop_length: Hop length for long window
            n_mfcc: Number of MFCCs per resolution
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            fusion_weights: Weights for (short, long) features
            distance: Distance metric
        """
        super().__init__(name=name, device=device, sr=sr)

        self.short_n_fft = short_n_fft
        self.short_hop_length = short_hop_length
        self.long_n_fft = long_n_fft
        self.long_hop_length = long_hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.fusion_weights = fusion_weights
        self.distance_type = distance

        # Store features for each resolution separately
        self._short_features: Optional[torch.Tensor] = None
        self._long_features: Optional[torch.Tensor] = None

    def _extract_resolution_features(
        self,
        waveform: np.ndarray,
        sr: int,
        n_fft: int,
        hop_length: int
    ) -> np.ndarray:
        """Extract MFCC features at a specific resolution."""
        features = mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        return features

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract multi-resolution features from waveform.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Concatenated feature vector from both resolutions
        """
        sr = sr or self.sr

        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Extract short window features
        short_mfcc = self._extract_resolution_features(
            waveform_np, sr, self.short_n_fft, self.short_hop_length
        )
        short_mfcc = torch.from_numpy(short_mfcc).float().to(self.device)
        short_pooled = mean_std_pool(short_mfcc, dim=-1)

        # Extract long window features
        long_mfcc = self._extract_resolution_features(
            waveform_np, sr, self.long_n_fft, self.long_hop_length
        )
        long_mfcc = torch.from_numpy(long_mfcc).float().to(self.device)
        long_pooled = mean_std_pool(long_mfcc, dim=-1)

        # Concatenate both resolutions
        combined = torch.cat([short_pooled, long_pooled], dim=-1)

        return combined

    def extract_features_separate(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features separately for each resolution.

        Used for late fusion with weighted distances.
        """
        sr = sr or self.sr

        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Short window
        short_mfcc = self._extract_resolution_features(
            waveform_np, sr, self.short_n_fft, self.short_hop_length
        )
        short_mfcc = torch.from_numpy(short_mfcc).float().to(self.device)
        short_pooled = mean_std_pool(short_mfcc, dim=-1)

        # Long window
        long_mfcc = self._extract_resolution_features(
            waveform_np, sr, self.long_n_fft, self.long_hop_length
        )
        long_mfcc = torch.from_numpy(long_mfcc).float().to(self.device)
        long_pooled = mean_std_pool(long_mfcc, dim=-1)

        return short_pooled, long_pooled

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """
        Build gallery storing features for each resolution separately.
        """
        short_features_list = []
        long_features_list = []
        labels_list = []
        indices_list = []

        iterator = gallery_samples
        if show_progress:
            from rich.progress import track
            iterator = track(gallery_samples, description="Building multi-res gallery...")

        for sample in iterator:
            waveform = sample['waveform']
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float()
            waveform = waveform.to(self.device)

            # Use sample's actual SR if available, otherwise fall back to self.sr
            sample_sr = sample.get('sr', self.sr)
            short_feat, long_feat = self.extract_features_separate(waveform, sample_sr)
            short_features_list.append(short_feat)
            long_features_list.append(long_feat)
            labels_list.append(sample['target'])
            indices_list.append(sample.get('idx', len(indices_list)))

        # Store both feature sets
        self._short_features = torch.stack(short_features_list, dim=0)
        self._long_features = torch.stack(long_features_list, dim=0)
        self._gallery_labels = torch.tensor(labels_list, device=self.device)
        self._gallery_indices = indices_list

        # Combined features for compatibility
        self._gallery_features = torch.cat([self._short_features, self._long_features], dim=1)

    def clear_gallery(self):
        """Clear gallery and multi-resolution specific feature storage."""
        super().clear_gallery()
        self._short_features = None
        self._long_features = None

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fused distances.

        Uses late fusion: weighted combination of distances from each resolution.
        """
        # Split query features
        n_short = self.n_mfcc * 2  # mean + std
        query_short = query_features[:n_short]
        query_long = query_features[n_short:]

        # Compute distances for each resolution
        short_dist = self._single_distance(query_short, self._short_features)
        long_dist = self._single_distance(query_long, self._long_features)

        # Weighted fusion
        w_short, w_long = self.fusion_weights
        combined_dist = w_short * short_dist + w_long * long_dist

        return combined_dist

    def _single_distance(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance using specified metric."""
        if self.distance_type == 'cosine':
            query_norm = query / (query.norm() + 1e-10)
            gallery_norm = gallery / (gallery.norm(dim=1, keepdim=True) + 1e-10)
            similarity = torch.matmul(gallery_norm, query_norm)
            return 1 - similarity

        elif self.distance_type == 'euclidean':
            diff = gallery - query.unsqueeze(0)
            return torch.sqrt((diff ** 2).sum(dim=1))

        else:
            raise ValueError(f"Unknown distance: {self.distance_type}")

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        query_sr: int = None,
    ) -> torch.Tensor:
        """Retrieve with multi-resolution features."""
        if self._short_features is None:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        if not isinstance(query_waveform, torch.Tensor):
            query_waveform = torch.from_numpy(query_waveform).float()
        query_waveform = query_waveform.to(self.device)

        # Extract combined features
        sr = query_sr if query_sr is not None else self.sr
        query_features = self.extract_features(query_waveform, sr)

        # Compute fused distances
        distances = self.compute_distance(query_features, self._gallery_features)

        sorted_indices = torch.argsort(distances)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            sorted_distances = distances[sorted_indices]
            return sorted_indices, sorted_distances

        return sorted_indices


def create_method_m7(
    device: str = 'cpu',
    sr: int = 22050,
    short_n_fft: int = 551,
    short_hop_length: int = 220,
    long_n_fft: int = 1764,
    long_hop_length: int = 441,
    n_mfcc: int = 20,
    n_mels: int = 128,
    fusion_weights: Tuple[float, float] = (0.5, 0.5),
    **kwargs
) -> MultiResRetriever:
    """
    Create M7: Multi-resolution MFCC + Late Fusion.

    Combines short window (high time resolution) and long window
    (high frequency resolution) features.
    """
    return MultiResRetriever(
        name="M7_MultiRes_Fusion",
        device=device,
        sr=sr,
        short_n_fft=short_n_fft,
        short_hop_length=short_hop_length,
        long_n_fft=long_n_fft,
        long_hop_length=long_hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        fusion_weights=fusion_weights,
        distance='cosine',
        **kwargs
    )
