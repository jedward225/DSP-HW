"""
Base class for audio retrieval methods.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union

from src.metrics.retrieval_metrics import compute_all_metrics, aggregate_metrics


def _stable_argsort(tensor: torch.Tensor, descending: bool = False) -> torch.Tensor:
    """
    Stable argsort with compatibility for older PyTorch versions.

    torch.argsort(..., stable=True) requires PyTorch >= 1.9.
    Falls back to standard argsort if stable is not supported.
    """
    try:
        return torch.argsort(tensor, descending=descending, stable=True)
    except TypeError:
        # PyTorch < 1.9 doesn't support stable parameter
        return torch.argsort(tensor, descending=descending)


class BaseRetriever(ABC):
    """
    Base class for audio retrieval methods.

    All retrievers must implement:
    - extract_features(): Extract features from audio
    - compute_distance(): Compute distance/similarity between features

    The retrieve() method is provided by the base class.
    """

    def __init__(
        self,
        name: str = "BaseRetriever",
        device: str = 'cpu',
        sr: int = 22050
    ):
        """
        Initialize retriever.

        Args:
            name: Name of the retrieval method
            device: Device for torch operations ('cpu' or 'cuda')
            sr: Expected sample rate of audio
        """
        self.name = name
        self.device = device
        self.sr = sr

        # Cache for gallery features
        self._gallery_features: Optional[torch.Tensor] = None
        self._gallery_labels: Optional[torch.Tensor] = None
        self._gallery_indices: Optional[List[int]] = None

    @abstractmethod
    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract features from audio waveform.

        Args:
            waveform: Audio waveform tensor
            sr: Sample rate (uses self.sr if not provided)

        Returns:
            Feature tensor (shape depends on method)
        """
        pass

    @abstractmethod
    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query and gallery features.

        Args:
            query_features: Features of query item
            gallery_features: Features of gallery items

        Returns:
            Distance tensor of shape (n_gallery,)
            Lower values = more similar
        """
        pass

    def extract_features_batch(
        self,
        waveforms: List[torch.Tensor],
        sr: int = None
    ) -> List[torch.Tensor]:
        """
        Extract features from multiple waveforms.

        Args:
            waveforms: List of audio waveform tensors
            sr: Sample rate

        Returns:
            List of feature tensors
        """
        return [self.extract_features(w, sr) for w in waveforms]

    def build_gallery(
        self,
        gallery_samples: List[Dict],
        show_progress: bool = False
    ):
        """
        Build the gallery index from samples.

        Args:
            gallery_samples: List of sample dictionaries with 'waveform' and 'target'
            show_progress: If True, show progress bar
        """
        features_list = []
        labels_list = []
        indices_list = []

        iterator = gallery_samples
        if show_progress:
            from rich.progress import track
            iterator = track(gallery_samples, description="Building gallery...")

        for sample in iterator:
            waveform = sample['waveform']
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float()
            waveform = waveform.to(self.device)

            # Use sample's actual SR if available, otherwise fall back to self.sr
            sample_sr = sample.get('sr', self.sr)
            features = self.extract_features(waveform, sample_sr)
            features_list.append(features)
            labels_list.append(sample['target'])
            indices_list.append(sample.get('idx', len(indices_list)))

        # Store gallery
        self._gallery_features = self._stack_features(features_list)
        self._gallery_labels = torch.tensor(labels_list, device=self.device)
        self._gallery_indices = indices_list

    def _stack_features(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack features into a single tensor.

        Override this method for non-standard feature shapes (e.g., sequences).
        """
        return torch.stack(features_list, dim=0)

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        query_sr: int = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve similar items from gallery for a query.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return (None = all)
            return_distances: If True, also return distances
            query_sr: Sample rate of query (uses self.sr if not provided)

        Returns:
            Indices of retrieved items (sorted by similarity)
            If return_distances=True, also returns distance values
        """
        if self._gallery_features is None:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        # Ensure query is on correct device
        if not isinstance(query_waveform, torch.Tensor):
            query_waveform = torch.from_numpy(query_waveform).float()
        query_waveform = query_waveform.to(self.device)

        # Extract query features using provided SR or default
        sr = query_sr if query_sr is not None else self.sr
        query_features = self.extract_features(query_waveform, sr)

        # Compute distances to all gallery items
        distances = self.compute_distance(query_features, self._gallery_features)

        # Sort by distance (ascending = most similar first)
        sorted_indices = _stable_argsort(distances, descending=False)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            sorted_distances = distances[sorted_indices]
            return sorted_indices, sorted_distances

        return sorted_indices

    def retrieve_with_labels(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        query_sr: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve and return labels of retrieved items.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return
            query_sr: Sample rate of query (uses self.sr if not provided)

        Returns:
            Tuple of (retrieved_indices, retrieved_labels)
        """
        indices = self.retrieve(query_waveform, k=k, query_sr=query_sr)
        labels = self._gallery_labels[indices]
        return indices, labels

    def evaluate_query(
        self,
        query_sample: Dict,
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a single query.

        Args:
            query_sample: Query sample dictionary
            k_values: List of K values for metrics (default: [1, 5, 10, 20])

        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        query_waveform = query_sample['waveform']
        query_label = query_sample['target']
        query_sr = query_sample.get('sr', None)  # Use sample SR if provided

        # Retrieve
        _, retrieved_labels = self.retrieve_with_labels(query_waveform, query_sr=query_sr)

        # Compute metrics
        metrics = compute_all_metrics(
            retrieved_labels,
            query_label,
            num_relevant=(self._gallery_labels == query_label).sum().item(),
            k_values=k_values
        )

        return metrics

    def evaluate(
        self,
        query_samples: List[Dict],
        show_progress: bool = False,
        k_values: List[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval performance on a set of queries.

        Args:
            query_samples: List of query sample dictionaries
            show_progress: If True, show progress bar
            k_values: List of K values for metrics (default: [1, 5, 10, 20])

        Returns:
            Aggregated metrics (mean and std for each metric)
        """
        all_metrics = []

        iterator = query_samples
        if show_progress:
            from rich.progress import track
            iterator = track(query_samples, description=f"Evaluating {self.name}...")

        for sample in iterator:
            metrics = self.evaluate_query(sample, k_values=k_values)
            all_metrics.append(metrics)

        return aggregate_metrics(all_metrics)

    def clear_gallery(self):
        """Clear the gallery cache."""
        self._gallery_features = None
        self._gallery_labels = None
        self._gallery_indices = None

    @property
    def gallery_size(self) -> int:
        """Return the number of items in the gallery."""
        if self._gallery_labels is None:
            return 0
        return len(self._gallery_labels)

    def to(self, device: str) -> 'BaseRetriever':
        """Move retriever to device."""
        self.device = device
        if self._gallery_features is not None:
            self._gallery_features = self._gallery_features.to(device)
        if self._gallery_labels is not None:
            self._gallery_labels = self._gallery_labels.to(device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, device={self.device}, gallery_size={self.gallery_size})"
