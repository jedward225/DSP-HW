"""
Partial Query Retrieval Method.

Enables retrieval using short query clips (0.5s, 1s, 2s)
against full-length gallery audios using sliding window matching.

This is a bonus feature (加分项) as specified in proposal Section 5.4.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Literal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.base import BaseRetriever


class PartialQueryRetriever(BaseRetriever):
    """
    Retrieval with partial (short) query clips.

    Uses sliding window to match short queries against full gallery items.
    For each gallery item, finds the best-matching window position.

    Protocol:
    - Query: Short clip (e.g., 0.5s, 1s, 2s)
    - Gallery: Full-length audios
    - Matching: Sliding window over gallery, find minimum distance window
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        query_duration_s: float = 1.0,
        stride_s: float = 0.5,
        aggregation: Literal['min', 'mean', 'max'] = 'min',
        name: str = "PartialQueryRetriever",
        device: str = 'cpu',
        sr: int = 22050,
    ):
        """
        Initialize partial query retriever.

        Args:
            base_retriever: Underlying retriever for feature extraction and distance
            query_duration_s: Duration of query clip in seconds
            stride_s: Stride for sliding window in seconds
            aggregation: How to aggregate window distances ('min', 'mean', 'max')
            name: Method name
            device: Computation device
            sr: Sample rate
        """
        super().__init__(name=name, device=device, sr=sr)

        self.base_retriever = base_retriever
        self.query_duration_s = query_duration_s
        self.stride_s = stride_s
        self.aggregation = aggregation

        # Compute lengths in samples
        self.query_length = int(query_duration_s * sr)
        self.stride_length = int(stride_s * sr)

        # Gallery storage
        self._gallery_samples: Optional[List[dict]] = None
        self._gallery_window_features: List[List[torch.Tensor]] = []

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Extract features using base retriever."""
        return self.base_retriever.extract_features(waveform, sr)

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """
        Build gallery with sliding window features.

        For each gallery audio, extract features from multiple windows
        to enable partial query matching.
        """
        self._gallery_samples = gallery_samples
        self._gallery_labels = torch.tensor(
            [s['target'] for s in gallery_samples],
            device=self.device
        )
        self._gallery_indices = list(range(len(gallery_samples)))
        self._gallery_window_features = []

        iterator = gallery_samples
        if show_progress:
            from rich.progress import track
            iterator = track(gallery_samples, description="Building gallery (partial)...")

        for sample in iterator:
            waveform = sample['waveform']
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()

            # Extract features from sliding windows
            window_features = []
            total_samples = len(waveform)

            # Slide window across audio
            for start in range(0, total_samples - self.query_length + 1, self.stride_length):
                window = waveform[start:start + self.query_length]
                window_tensor = torch.from_numpy(window).float().to(self.device)
                feat = self.base_retriever.extract_features(window_tensor, self.sr)
                window_features.append(feat)

            if window_features:
                self._gallery_window_features.append(window_features)
            else:
                # Audio shorter than query length - use full audio
                full_tensor = torch.from_numpy(waveform).float().to(self.device)
                feat = self.base_retriever.extract_features(full_tensor, self.sr)
                self._gallery_window_features.append([feat])

        # For compatibility with base class
        self._gallery_features = None

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute minimum distance across all windows.

        For each gallery item, find the window with minimum distance to query.
        """
        distances = []

        for window_feats in self._gallery_window_features:
            # Compute distance to each window
            window_dists = []
            for wf in window_feats:
                # Use base retriever's distance computation
                # Need to handle the case where it expects batch dimension
                if len(wf.shape) == 1:
                    wf = wf.unsqueeze(0)
                dist = self.base_retriever.compute_distance(query_features, wf)
                window_dists.append(dist.item())

            # Aggregate across windows
            if self.aggregation == 'min':
                distances.append(min(window_dists))
            elif self.aggregation == 'mean':
                distances.append(sum(window_dists) / len(window_dists))
            elif self.aggregation == 'max':
                distances.append(max(window_dists))

        return torch.tensor(distances, device=query_features.device, dtype=query_features.dtype)

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        crop_mode: str = 'center'
    ):
        """
        Retrieve using partial query matching.

        Args:
            query_waveform: Query audio (will be cropped to query_duration_s)
            k: Number of results to return
            return_distances: If True, also return distances
            crop_mode: How to crop query ('center', 'start', 'random')

        Returns:
            Indices of retrieved items (sorted by distance)
        """
        if self._gallery_window_features is None or len(self._gallery_window_features) == 0:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        # Convert to tensor if needed
        if not isinstance(query_waveform, torch.Tensor):
            query_waveform = torch.from_numpy(query_waveform).float()
        query_waveform = query_waveform.to(self.device)

        # Crop query to specified duration
        if len(query_waveform) > self.query_length:
            if crop_mode == 'center':
                start = (len(query_waveform) - self.query_length) // 2
            elif crop_mode == 'start':
                start = 0
            elif crop_mode == 'random':
                start = np.random.randint(0, len(query_waveform) - self.query_length + 1)
            else:
                start = (len(query_waveform) - self.query_length) // 2

            query_waveform = query_waveform[start:start + self.query_length]

        # Extract query features
        query_features = self.extract_features(query_waveform, self.sr)

        # Compute distances
        distances = self.compute_distance(query_features, None)

        # Sort by distance
        sorted_indices = torch.argsort(distances)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            return sorted_indices, distances[sorted_indices]

        return sorted_indices

    def retrieve_with_localization(
        self,
        query_waveform: torch.Tensor,
        gallery_idx: int,
        k_windows: int = 5
    ) -> Dict:
        """
        Find best matching locations within a gallery item.

        Returns top-k matching window positions as a localization heatmap.

        Args:
            query_waveform: Query audio
            gallery_idx: Index of gallery item to localize within
            k_windows: Number of top windows to return

        Returns:
            Dictionary with:
            - 'positions': List of (start_s, end_s) tuples for top windows
            - 'distances': Distances for each window
            - 'heatmap': Full distance array for all windows
        """
        if not isinstance(query_waveform, torch.Tensor):
            query_waveform = torch.from_numpy(query_waveform).float()
        query_waveform = query_waveform.to(self.device)

        # Crop query
        if len(query_waveform) > self.query_length:
            start = (len(query_waveform) - self.query_length) // 2
            query_waveform = query_waveform[start:start + self.query_length]

        # Extract query features
        query_features = self.extract_features(query_waveform, self.sr)

        # Get window features for specified gallery item
        window_feats = self._gallery_window_features[gallery_idx]

        # Compute distance to each window
        window_dists = []
        for wf in window_feats:
            if len(wf.shape) == 1:
                wf = wf.unsqueeze(0)
            dist = self.base_retriever.compute_distance(query_features, wf)
            window_dists.append(dist.item())

        window_dists = np.array(window_dists)

        # Get top-k windows
        top_k_indices = np.argsort(window_dists)[:k_windows]

        # Convert to time positions
        positions = []
        for idx in top_k_indices:
            start_sample = idx * self.stride_length
            end_sample = start_sample + self.query_length
            start_s = start_sample / self.sr
            end_s = end_sample / self.sr
            positions.append((start_s, end_s))

        return {
            'positions': positions,
            'distances': window_dists[top_k_indices].tolist(),
            'heatmap': window_dists.tolist(),
        }

    def clear_gallery(self):
        """Clear the gallery cache."""
        super().clear_gallery()
        self._gallery_samples = None
        self._gallery_window_features = []

    @property
    def gallery_size(self) -> int:
        """Return number of items in gallery."""
        return len(self._gallery_window_features)


def create_partial_retriever(
    base_retriever: BaseRetriever,
    query_duration_s: float = 1.0,
    stride_s: float = 0.5,
    aggregation: str = 'min',
    **kwargs
) -> PartialQueryRetriever:
    """
    Create a partial query retriever.

    Args:
        base_retriever: Underlying retriever for features/distance
        query_duration_s: Query clip duration in seconds
        stride_s: Sliding window stride in seconds
        aggregation: Window aggregation method ('min', 'mean', 'max')

    Returns:
        PartialQueryRetriever instance
    """
    return PartialQueryRetriever(
        base_retriever=base_retriever,
        query_duration_s=query_duration_s,
        stride_s=stride_s,
        aggregation=aggregation,
        name=f"Partial_{query_duration_s}s",
        device=base_retriever.device,
        sr=base_retriever.sr,
    )
