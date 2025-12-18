"""
Partial Query Retrieval Method.

Enables retrieval using short query clips (0.5s, 1s, 2s)
against full-length gallery audios using sliding window matching.

This is a bonus feature (加分项) as specified in proposal Section 5.4.

Compatibility Notes:
    - Works with: PoolRetriever (M1, M2, M3, M4), BoAWRetriever (M6)
    - NOT compatible with: DTWRetriever (M5) - DTW uses sequences, not fixed features
    - Caution with Mahalanobis distance: Single-window "galleries" may cause
      covariance estimation issues (singular matrix)
    - Memory/Performance: Caches per-window features for all gallery items,
      which can be memory-intensive for large galleries
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Literal, Tuple

from src.retrieval.base import BaseRetriever
from src.dsp_core import mfcc


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
        fast_windowing: bool = False,
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
        self.fast_windowing = fast_windowing

        # Compute lengths in samples
        self.query_length = int(query_duration_s * sr)
        self.stride_length = int(stride_s * sr)

        # Gallery storage
        self._gallery_samples: Optional[List[dict]] = None
        # Per-gallery-item window features (list of tensors shaped (n_windows, D)).
        self._gallery_window_features: List[torch.Tensor] = []

        # Cache window features by (sample_idx, sample_sr) so repeated builds across
        # folds don't recompute MFCCs for the same audio.
        self._window_feature_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Flattened view for fast batched distance computation.
        self._flat_window_features: Optional[torch.Tensor] = None  # (total_windows, D)
        self._flat_window_owner: Optional[torch.Tensor] = None  # (total_windows,)

    def _is_fast_pool_mfcc(self) -> bool:
        """Return True if we can use a fast MFCC windowing implementation."""
        return bool(
            self.fast_windowing
            and getattr(self.base_retriever, 'feature_type', None) == 'mfcc'
            and getattr(self.base_retriever, 'pooling', None) == 'mean_std'
            and hasattr(self.base_retriever, 'n_mfcc')
            and hasattr(self.base_retriever, 'n_fft')
            and hasattr(self.base_retriever, 'hop_length')
            and hasattr(self.base_retriever, 'n_mels')
        )

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
        # Ensure BoAW codebook is fitted before extracting features
        # This is needed because BoAW.extract_features() requires a fitted codebook
        if hasattr(self.base_retriever, 'fit_codebook') and hasattr(self.base_retriever, '_codebook_fitted'):
            if not self.base_retriever._codebook_fitted:
                self.base_retriever.fit_codebook(gallery_samples, show_progress=show_progress)

        if getattr(self.base_retriever, 'distance', None) == 'mahalanobis':
            self.base_retriever.clear_gallery()
            self.base_retriever.build_gallery(gallery_samples, show_progress=False)

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

            sample_sr = sample.get('sr', self.sr)
            query_length = max(1, int(self.query_duration_s * sample_sr))
            stride_length = max(1, int(self.stride_s * sample_sr))

            sample_idx = sample.get('idx', None)
            cache_key: Optional[Tuple[int, int]] = None
            if sample_idx is not None:
                cache_key = (int(sample_idx), int(sample_sr))

            if cache_key is not None and cache_key in self._window_feature_cache:
                self._gallery_window_features.append(self._window_feature_cache[cache_key])
                continue

            # Extract features from sliding windows
            window_features: List[torch.Tensor] = []
            total_samples = len(waveform)

            if self._is_fast_pool_mfcc():
                n_mfcc = int(getattr(self.base_retriever, 'n_mfcc'))
                n_fft = int(getattr(self.base_retriever, 'n_fft'))
                hop_length = int(getattr(self.base_retriever, 'hop_length'))
                n_mels = int(getattr(self.base_retriever, 'n_mels'))
                fmin = float(getattr(self.base_retriever, 'fmin', 0.0))
                fmax = getattr(self.base_retriever, 'fmax', None)
                fmax = None if fmax is None else float(fmax)
                window = str(getattr(self.base_retriever, 'window', 'hann'))

                # Compute MFCC sequence once for the full audio.
                # Cache on the base retriever so multiple PartialQueryRetriever instances
                # (different durations/strides) can reuse it.
                mfcc_seq = None
                if sample_idx is not None:
                    mfcc_cache = getattr(self.base_retriever, '_partial_mfcc_seq_cache', None)
                    if mfcc_cache is None:
                        mfcc_cache = {}
                        setattr(self.base_retriever, '_partial_mfcc_seq_cache', mfcc_cache)

                    mfcc_key = (
                        int(sample_idx),
                        int(sample_sr),
                        n_mfcc,
                        n_fft,
                        hop_length,
                        n_mels,
                        float(fmin),
                        None if fmax is None else float(fmax),
                        window,
                    )

                    mfcc_seq = mfcc_cache.get(mfcc_key)
                    if mfcc_seq is None:
                        mfcc_seq = mfcc(
                            y=waveform,
                            sr=sample_sr,
                            n_mfcc=n_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            n_mels=n_mels,
                            fmin=fmin,
                            fmax=fmax,
                            window=window,
                        ).astype(np.float32, copy=False)
                        mfcc_cache[mfcc_key] = mfcc_seq

                if mfcc_seq is None:
                    mfcc_seq = mfcc(
                        y=waveform,
                        sr=sample_sr,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                        fmin=fmin,
                        fmax=fmax,
                        window=window,
                    ).astype(np.float32, copy=False)

                n_frames = int(mfcc_seq.shape[1])
                pad = n_fft // 2

                starts = np.arange(0, max(1, total_samples - query_length + 1), stride_length)
                ends = starts + query_length
                ends = np.minimum(ends, total_samples)

                start_frames = (starts + pad + hop_length - 1) // hop_length
                end_frames = (ends + pad + hop_length - 1) // hop_length
                start_frames = np.clip(start_frames, 0, n_frames)
                end_frames = np.clip(end_frames, 0, n_frames)
                frame_counts = np.maximum(1, end_frames - start_frames)

                # Cumulative sums for fast window stats
                cumsum = np.cumsum(mfcc_seq, axis=1)
                cumsum_sq = np.cumsum(mfcc_seq ** 2, axis=1)
                cumsum = np.concatenate([np.zeros((n_mfcc, 1), dtype=mfcc_seq.dtype), cumsum], axis=1)
                cumsum_sq = np.concatenate([np.zeros((n_mfcc, 1), dtype=mfcc_seq.dtype), cumsum_sq], axis=1)

                sums = cumsum[:, end_frames] - cumsum[:, start_frames]
                sums_sq = cumsum_sq[:, end_frames] - cumsum_sq[:, start_frames]

                means = sums / frame_counts
                var_num = sums_sq - (sums ** 2) / frame_counts
                denom = np.maximum(1, frame_counts - 1)
                vars_unbiased = var_num / denom
                vars_unbiased = np.where(frame_counts > 1, vars_unbiased, 0.0)
                stds = np.sqrt(np.maximum(vars_unbiased, 0.0))

                feats = np.concatenate([means, stds], axis=0).T  # (n_windows, 2*n_mfcc)
                window_tensor = torch.from_numpy(feats).float().to(self.device)
                self._gallery_window_features.append(window_tensor)

                if cache_key is not None:
                    self._window_feature_cache[cache_key] = window_tensor

                continue

            # Slide window across audio
            for start in range(0, total_samples - query_length + 1, stride_length):
                window = waveform[start:start + query_length]
                window_tensor = torch.from_numpy(window).float().to(self.device)
                feat = self.base_retriever.extract_features(window_tensor, sample_sr)
                window_features.append(feat)

            if not window_features:
                # Audio shorter than query length - use full audio
                full_tensor = torch.from_numpy(waveform).float().to(self.device)
                feat = self.base_retriever.extract_features(full_tensor, sample_sr)
                window_features = [feat]

            # Stack to (n_windows, D)
            window_tensor = torch.stack(window_features, dim=0)
            self._gallery_window_features.append(window_tensor)

            if cache_key is not None:
                self._window_feature_cache[cache_key] = window_tensor

        # For compatibility with base class
        self._gallery_features = None

        # Build flattened view for fast distance computation.
        flat_feats: List[torch.Tensor] = []
        flat_owner: List[torch.Tensor] = []
        for gallery_idx, window_tensor in enumerate(self._gallery_window_features):
            if window_tensor.dim() == 1:
                window_tensor = window_tensor.unsqueeze(0)
            flat_feats.append(window_tensor)
            flat_owner.append(
                torch.full(
                    (window_tensor.shape[0],),
                    gallery_idx,
                    device=window_tensor.device,
                    dtype=torch.long,
                )
            )

        if flat_feats:
            self._flat_window_features = torch.cat(flat_feats, dim=0)
            self._flat_window_owner = torch.cat(flat_owner, dim=0)
        else:
            self._flat_window_features = None
            self._flat_window_owner = None

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute minimum distance across all windows.

        For each gallery item, find the window with minimum distance to query.
        """
        if self._flat_window_features is not None and self._flat_window_owner is not None:
            window_dists = self.base_retriever.compute_distance(query_features, self._flat_window_features)
            window_dists = window_dists.flatten()

            n_gallery = len(self._gallery_window_features)
            owners = self._flat_window_owner

            if self.aggregation == 'min':
                distances = torch.full(
                    (n_gallery,),
                    float('inf'),
                    device=window_dists.device,
                    dtype=window_dists.dtype,
                )
                distances.scatter_reduce_(0, owners, window_dists, reduce='amin', include_self=True)
                return distances

            if self.aggregation == 'max':
                distances = torch.full(
                    (n_gallery,),
                    -float('inf'),
                    device=window_dists.device,
                    dtype=window_dists.dtype,
                )
                distances.scatter_reduce_(0, owners, window_dists, reduce='amax', include_self=True)
                return distances

            if self.aggregation == 'mean':
                sums = torch.zeros(
                    (n_gallery,),
                    device=window_dists.device,
                    dtype=window_dists.dtype,
                )
                sums.scatter_add_(0, owners, window_dists)
                counts = torch.zeros(
                    (n_gallery,),
                    device=window_dists.device,
                    dtype=window_dists.dtype,
                )
                counts.scatter_add_(0, owners, torch.ones_like(window_dists))
                return sums / counts.clamp_min(1.0)

        # Fallback (should be rare): compute per-gallery-item.
        distances: List[torch.Tensor] = []
        for window_tensor in self._gallery_window_features:
            if window_tensor.dim() == 1:
                window_tensor = window_tensor.unsqueeze(0)
            window_dists = self.base_retriever.compute_distance(query_features, window_tensor).flatten()

            if self.aggregation == 'min':
                distances.append(window_dists.min())
            elif self.aggregation == 'mean':
                distances.append(window_dists.mean())
            elif self.aggregation == 'max':
                distances.append(window_dists.max())
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return torch.stack(distances, dim=0).to(device=query_features.device, dtype=query_features.dtype)

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        crop_mode: str = 'center',
        query_sr: int = None,
        query_idx: Optional[int] = None,
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

        sr = query_sr if query_sr is not None else self.sr
        query_length = max(1, int(self.query_duration_s * sr))

        crop_start = 0

        # Crop query to specified duration
        if len(query_waveform) > query_length:
            if crop_mode == 'center':
                crop_start = (len(query_waveform) - query_length) // 2
            elif crop_mode == 'start':
                crop_start = 0
            elif crop_mode == 'random':
                crop_start = int(np.random.randint(0, len(query_waveform) - query_length + 1))
            else:
                crop_start = (len(query_waveform) - query_length) // 2

            query_waveform = query_waveform[crop_start:crop_start + query_length]

        # Extract query features
        query_features = None
        if self._is_fast_pool_mfcc() and query_idx is not None:
            mfcc_cache = getattr(self.base_retriever, '_partial_mfcc_seq_cache', None)
            if mfcc_cache is not None:
                n_mfcc = int(getattr(self.base_retriever, 'n_mfcc'))
                n_fft = int(getattr(self.base_retriever, 'n_fft'))
                hop_length = int(getattr(self.base_retriever, 'hop_length'))
                n_mels = int(getattr(self.base_retriever, 'n_mels'))
                fmin = float(getattr(self.base_retriever, 'fmin', 0.0))
                fmax = getattr(self.base_retriever, 'fmax', None)
                fmax = None if fmax is None else float(fmax)
                window = str(getattr(self.base_retriever, 'window', 'hann'))

                mfcc_key = (
                    int(query_idx),
                    int(sr),
                    n_mfcc,
                    n_fft,
                    hop_length,
                    n_mels,
                    float(fmin),
                    None if fmax is None else float(fmax),
                    window,
                )

                mfcc_seq = mfcc_cache.get(mfcc_key)
                if mfcc_seq is not None:
                    n_frames = int(mfcc_seq.shape[1])
                    pad = n_fft // 2
                    start_frame = int((crop_start + pad + hop_length - 1) // hop_length)
                    end_frame = int((crop_start + query_length + pad + hop_length - 1) // hop_length)
                    start_frame = int(np.clip(start_frame, 0, n_frames))
                    end_frame = int(np.clip(end_frame, 0, n_frames))

                    if end_frame <= start_frame:
                        end_frame = min(n_frames, start_frame + 1)

                    seg = mfcc_seq[:, start_frame:end_frame]
                    mean = seg.mean(axis=1)
                    if seg.shape[1] > 1:
                        std = seg.std(axis=1, ddof=1)
                    else:
                        std = np.zeros_like(mean)

                    pooled = np.concatenate([mean, std], axis=0)
                    query_features = torch.from_numpy(pooled).float().to(self.device)

        if query_features is None:
            query_features = self.extract_features(query_waveform, sr)

        # Compute distances
        distances = self.compute_distance(query_features, None)

        # Sort by distance
        sorted_indices = torch.argsort(distances)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            return sorted_indices, distances[sorted_indices]

        return sorted_indices

    def retrieve_with_labels(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        crop_mode: str = 'center',
        query_sr: int = None,
        query_idx: Optional[int] = None,
    ):
        """
        Retrieve and return labels of retrieved items.

        Overrides base method to pass crop_mode through.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return
            crop_mode: How to crop query ('center', 'start', 'random')

        Returns:
            Tuple of (retrieved_indices, retrieved_labels)
        """
        indices = self.retrieve(
            query_waveform,
            k=k,
            crop_mode=crop_mode,
            query_sr=query_sr,
            query_idx=query_idx,
        )
        labels = self._gallery_labels[indices]
        return indices, labels

    def retrieve_with_localization(
        self,
        query_waveform: torch.Tensor,
        gallery_idx: int,
        k_windows: int = 5,
        query_sr: int = None,
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

        sr = query_sr if query_sr is not None else self.sr
        query_length = max(1, int(self.query_duration_s * sr))

        # Crop query
        if len(query_waveform) > query_length:
            start = (len(query_waveform) - query_length) // 2
            query_waveform = query_waveform[start:start + query_length]

        # Extract query features
        query_features = self.extract_features(query_waveform, sr)

        # Get window features for specified gallery item
        window_tensor = self._gallery_window_features[gallery_idx]
        if window_tensor.dim() == 1:
            window_tensor = window_tensor.unsqueeze(0)

        # Compute distance to each window (batched)
        window_dists = self.base_retriever.compute_distance(query_features, window_tensor).detach().flatten()
        window_dists = window_dists.cpu().numpy()

        # Get top-k windows
        top_k_indices = np.argsort(window_dists)[:k_windows]

        gallery_sr = self._gallery_samples[gallery_idx].get('sr', self.sr) if self._gallery_samples else self.sr
        gallery_query_length = max(1, int(self.query_duration_s * gallery_sr))
        gallery_stride_length = max(1, int(self.stride_s * gallery_sr))

        # Convert to time positions
        positions = []
        for idx in top_k_indices:
            start_sample = idx * gallery_stride_length
            end_sample = start_sample + gallery_query_length
            start_s = start_sample / gallery_sr
            end_s = end_sample / gallery_sr
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
        self._flat_window_features = None
        self._flat_window_owner = None

    @property
    def gallery_size(self) -> int:
        """Return number of items in gallery."""
        return len(self._gallery_window_features)


def create_partial_retriever(
    base_retriever: BaseRetriever,
    query_duration_s: float = 1.0,
    stride_s: float = 0.5,
    aggregation: str = 'min',
    fast_windowing: bool = False,
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
        fast_windowing=fast_windowing,
        name=f"Partial_{query_duration_s}s",
        device=base_retriever.device,
        sr=base_retriever.sr,
    )
