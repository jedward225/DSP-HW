"""
DTW-based retrieval method (M5).

Uses Dynamic Time Warping to compare sequences directly.
Numba JIT compilation is used for acceleration.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, TYPE_CHECKING

from src.retrieval.base import BaseRetriever
from src.dsp_core import mfcc

# Numba JIT compilation for DTW (required dependency)
from numba import jit, prange


@jit(nopython=True, cache=True)
def _dtw_distance_numba(
    x: np.ndarray,
    y: np.ndarray,
    sakoe_chiba_radius: int = -1
) -> float:
    """
    Compute DTW distance between two sequences using Numba JIT.

    Args:
        x: First sequence, shape (n_frames_x, n_features)
        y: Second sequence, shape (n_frames_y, n_features)
        sakoe_chiba_radius: Band radius for Sakoe-Chiba constraint (-1 = no constraint)

    Returns:
        DTW distance
    """
    n, m = x.shape[0], y.shape[0]

    # Initialize cost matrix with infinity
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0

    # Compute DTW with optional Sakoe-Chiba band
    for i in range(1, n + 1):
        if sakoe_chiba_radius > 0:
            j_start = max(1, i - sakoe_chiba_radius)
            j_end = min(m + 1, i + sakoe_chiba_radius + 1)
        else:
            j_start = 1
            j_end = m + 1

        for j in range(j_start, j_end):
            # Euclidean distance between frames
            cost = 0.0
            for k in range(x.shape[1]):
                diff = x[i-1, k] - y[j-1, k]
                cost += diff * diff
            cost = np.sqrt(cost)

            # DTW recurrence
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[n, m]


@jit(nopython=True, cache=True)
def _downsample_sequence(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample sequence by averaging consecutive frames.

    Args:
        x: Sequence of shape (n_frames, n_features)
        factor: Downsampling factor

    Returns:
        Downsampled sequence
    """
    n_frames = x.shape[0]
    n_features = x.shape[1]
    new_n_frames = n_frames // factor

    result = np.zeros((new_n_frames, n_features))
    for i in range(new_n_frames):
        for j in range(n_features):
            total = 0.0
            for k in range(factor):
                total += x[i * factor + k, j]
            result[i, j] = total / factor

    return result


@jit(nopython=True, cache=True)
def _dtw_with_path(x: np.ndarray, y: np.ndarray, radius: int) -> Tuple[float, np.ndarray]:
    """
    DTW with path tracking and radius constraint.

    Args:
        x: First sequence
        y: Second sequence
        radius: Sakoe-Chiba radius

    Returns:
        Tuple of (distance, path) where path is (n_path, 2) array
    """
    n, m = x.shape[0], y.shape[0]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m + 1, i + radius + 1)

        for j in range(j_start, j_end):
            cost = 0.0
            for k in range(x.shape[1]):
                diff = x[i-1, k] - y[j-1, k]
                cost += diff * diff
            cost = np.sqrt(cost)

            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Backtrack to find path
    path = np.zeros((n + m, 2), dtype=np.int64)
    path_len = 0
    i, j = n, m

    while i > 0 or j > 0:
        path[path_len, 0] = i - 1
        path[path_len, 1] = j - 1
        path_len += 1

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            if D[i-1, j-1] == min_val:
                i -= 1
                j -= 1
            elif D[i-1, j] == min_val:
                i -= 1
            else:
                j -= 1

    return D[n, m], path[:path_len][::-1]


@jit(nopython=True, cache=True)
def _fastdtw_recursive(x: np.ndarray, y: np.ndarray, radius: int) -> float:
    """
    FastDTW: O(N) approximate DTW using coarse-to-fine approach.

    Algorithm:
    1. If sequences are short enough, use exact DTW
    2. Otherwise:
       a. Downsample both sequences by factor 2
       b. Recursively compute DTW on downsampled sequences
       c. Project the path to full resolution
       d. Compute DTW within projected band (radius constraint)

    Args:
        x: First sequence, shape (n_frames, n_features)
        y: Second sequence, shape (n_frames, n_features)
        radius: Sakoe-Chiba radius for path projection

    Returns:
        Approximate DTW distance
    """
    min_size = radius + 2

    # Base case: use exact DTW for small sequences
    if x.shape[0] < min_size or y.shape[0] < min_size:
        return _dtw_distance_numba(x, y, -1)

    # Downsample
    x_small = _downsample_sequence(x, 2)
    y_small = _downsample_sequence(y, 2)

    # Recursive call
    _, low_res_path = _dtw_with_path(x_small, y_small, max(radius, 1))

    # Project path to full resolution and expand window
    n, m = x.shape[0], y.shape[0]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    # First pass: mark valid cells based on projected path
    # Using a separate array to track which cells are in the band
    valid = np.zeros((n + 1, m + 1), dtype=np.bool_)
    valid[0, 0] = True

    for p in range(low_res_path.shape[0]):
        i_low = low_res_path[p, 0]
        j_low = low_res_path[p, 1]

        # Project to full resolution with radius expansion
        i_start = max(1, i_low * 2 - radius + 1)
        i_end = min(n + 1, i_low * 2 + 2 + radius + 1)
        j_start = max(1, j_low * 2 - radius + 1)
        j_end = min(m + 1, j_low * 2 + 2 + radius + 1)

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                valid[i, j] = True

    # Second pass: fill DP matrix in row-major order
    # This ensures dependencies D[i-1,j], D[i,j-1], D[i-1,j-1] are computed first
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if valid[i, j]:
                # Euclidean distance between frames
                cost = 0.0
                for k in range(x.shape[1]):
                    diff = x[i-1, k] - y[j-1, k]
                    cost += diff * diff
                cost = np.sqrt(cost)

                # DTW recurrence - only consider valid predecessors
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[n, m] if D[n, m] != np.inf else 1e10


@jit(nopython=True, cache=True)
def _dtw_distance_itakura(x: np.ndarray, y: np.ndarray) -> float:
    """
    DTW with Itakura parallelogram constraint.

    Itakura constraint restricts the warping path to lie within
    a parallelogram, enforcing more symmetric alignments.
    Constraint: 0.5 <= (i/n) / (j/m) <= 2

    This prevents extreme time stretching/compression.

    Args:
        x: First sequence, shape (n_frames_x, n_features)
        y: Second sequence, shape (n_frames_y, n_features)

    Returns:
        DTW distance with Itakura constraint
    """
    n, m = x.shape[0], y.shape[0]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Itakura parallelogram constraint
            # Check if (i, j) is within the valid region
            # Constraint: 0.5 <= (i/n) / (j/m) <= 2
            # Rearranged: 0.5 * j * n <= i * m <= 2 * j * n
            ratio_check = i * m
            lower_bound = 0.5 * j * n
            upper_bound = 2.0 * j * n

            if lower_bound <= ratio_check <= upper_bound:
                # Euclidean distance between frames
                cost = 0.0
                for k in range(x.shape[1]):
                    diff = x[i-1, k] - y[j-1, k]
                    cost += diff * diff
                cost = np.sqrt(cost)

                # DTW recurrence
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[n, m] if D[n, m] != np.inf else 1e10


@jit(nopython=True, parallel=True, cache=True)
def _dtw_distance_batch_numba(
    query: np.ndarray,
    gallery: List[np.ndarray],
    sakoe_chiba_radius: int = -1
) -> np.ndarray:
    """
    Compute DTW distances between query and all gallery items.

    This function is parallelized using Numba's prange.

    Args:
        query: Query sequence, shape (n_frames, n_features)
        gallery: List of gallery sequences
        sakoe_chiba_radius: Band radius for Sakoe-Chiba constraint

    Returns:
        Distance array of shape (n_gallery,)
    """
    n_gallery = len(gallery)
    distances = np.zeros(n_gallery)

    for i in prange(n_gallery):
        distances[i] = _dtw_distance_numba(query, gallery[i], sakoe_chiba_radius)

    return distances


class DTWRetriever(BaseRetriever):
    """
    Retriever using Dynamic Time Warping distance.

    DTW allows comparison of sequences with different lengths
    by finding the optimal alignment.
    """

    def __init__(
        self,
        name: str = "DTWRetriever",
        device: str = 'cpu',  # DTW runs on CPU with Numba
        sr: int = 22050,
        # Feature extraction parameters
        n_mfcc: int = 13,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0.0,
        fmax: float = None,
        # DTW parameters
        sakoe_chiba_radius: int = -1,  # -1 = no constraint
        constraint: str = 'none',  # 'none', 'sakoe_chiba', 'itakura', 'fastdtw'
        fastdtw_radius: int = 10,  # radius for FastDTW
        use_delta: bool = False,
    ):
        """
        Initialize DTW-based retriever.

        Args:
            name: Method name
            device: Device (DTW always uses CPU with Numba)
            sr: Sample rate
            n_mfcc: Number of MFCCs
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency
            sakoe_chiba_radius: Sakoe-Chiba band radius (-1 = full DTW)
            constraint: DTW constraint type ('none', 'sakoe_chiba', 'itakura', 'fastdtw')
            fastdtw_radius: Radius for FastDTW path expansion (default: 10)
            use_delta: If True, append delta features
        """
        # DTW uses CPU because Numba doesn't support CUDA directly
        super().__init__(name=name, device='cpu', sr=sr)

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.constraint = constraint
        self.fastdtw_radius = fastdtw_radius
        self.use_delta = use_delta

        # Gallery stores sequences, not stacked tensor
        self._gallery_sequences: List[np.ndarray] = []

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract MFCC sequence from waveform.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            MFCC sequence tensor of shape (n_frames, n_features)
        """
        sr = sr or self.sr

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Extract MFCCs
        features = mfcc(
            y=waveform_np,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Transpose to (n_frames, n_features) for DTW
        features = features.T

        if self.use_delta:
            from src.dsp_core import delta as compute_delta
            delta1 = compute_delta(features.T, width=9, order=1).T
            delta2 = compute_delta(features.T, width=9, order=2).T
            features = np.concatenate([features, delta1, delta2], axis=1)

        return torch.from_numpy(features).float()

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """
        Build gallery of sequences.

        For DTW, we store sequences separately instead of stacking.
        """
        self._gallery_sequences = []
        labels_list = []
        indices_list = []

        iterator = gallery_samples
        if show_progress:
            from rich.progress import track
            iterator = track(gallery_samples, description="Building gallery (DTW)...")

        for sample in iterator:
            waveform = sample['waveform']
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float()

            # Use sample's actual SR if available, otherwise fall back to self.sr
            sample_sr = sample.get('sr', self.sr)
            features = self.extract_features(waveform, sample_sr)
            self._gallery_sequences.append(features.numpy())
            labels_list.append(sample['target'])
            indices_list.append(sample.get('idx', len(indices_list)))

        self._gallery_labels = torch.tensor(labels_list)
        self._gallery_indices = indices_list

        # For compatibility
        self._gallery_features = None

    def _stack_features(self, features_list: List[torch.Tensor]) -> None:
        """Override: DTW doesn't stack features."""
        # Do nothing - we use _gallery_sequences instead
        pass

    def clear_gallery(self):
        """Clear gallery and DTW-specific sequence storage."""
        super().clear_gallery()
        self._gallery_sequences = []

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DTW distances between query and gallery.

        Args:
            query_features: Query sequence (n_frames, n_features)
            gallery_features: Not used (we use _gallery_sequences)

        Returns:
            Distance tensor (n_gallery,)
        """
        query_np = query_features.numpy()

        # Compute DTW distances based on constraint type
        distances = np.zeros(len(self._gallery_sequences))

        if self.constraint == 'itakura':
            # Use Itakura parallelogram constraint
            for i, gallery_seq in enumerate(self._gallery_sequences):
                distances[i] = _dtw_distance_itakura(query_np, gallery_seq)
        elif self.constraint == 'fastdtw':
            # Use FastDTW (O(N) approximate DTW)
            for i, gallery_seq in enumerate(self._gallery_sequences):
                distances[i] = _fastdtw_recursive(query_np, gallery_seq, self.fastdtw_radius)
        else:
            # Use Sakoe-Chiba band constraint or no constraint
            sakoe_radius = self.sakoe_chiba_radius if self.constraint == 'sakoe_chiba' else -1
            for i, gallery_seq in enumerate(self._gallery_sequences):
                distances[i] = _dtw_distance_numba(query_np, gallery_seq, sakoe_radius)

        return torch.from_numpy(distances).float()

    def retrieve(
        self,
        query_waveform: torch.Tensor,
        k: int = None,
        return_distances: bool = False,
        query_sr: int = None,
    ) -> torch.Tensor:
        """
        Retrieve similar items using DTW.

        Args:
            query_waveform: Query audio waveform
            k: Number of results to return
            return_distances: If True, also return distances

        Returns:
            Indices of retrieved items
        """
        if len(self._gallery_sequences) == 0:
            raise RuntimeError("Gallery not built. Call build_gallery() first.")

        # Convert query
        if not isinstance(query_waveform, torch.Tensor):
            query_waveform = torch.from_numpy(query_waveform).float()

        # Extract query features
        sr = query_sr if query_sr is not None else self.sr
        query_features = self.extract_features(query_waveform, sr)

        # Compute distances
        distances = self.compute_distance(query_features, None)

        # Sort by distance
        sorted_indices = torch.argsort(distances)

        if k is not None:
            sorted_indices = sorted_indices[:k]

        if return_distances:
            sorted_distances = distances[sorted_indices]
            return sorted_indices, sorted_distances

        return sorted_indices

    @property
    def gallery_size(self) -> int:
        """Return number of items in gallery."""
        return len(self._gallery_sequences)


def create_method_m5(
    device: str = 'cpu',
    sr: int = 22050,
    n_mfcc: int = 13,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
    sakoe_chiba_radius: int = -1,
    constraint: str = 'none',
    use_delta: bool = False,
    **kwargs
) -> DTWRetriever:
    """
    Create M5: MFCC + DTW Distance.

    DTW allows direct comparison of variable-length sequences.

    Args:
        device: Device (DTW uses CPU)
        sr: Sample rate
        n_mfcc: Number of MFCCs
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length
        sakoe_chiba_radius: Sakoe-Chiba band radius (used if constraint='sakoe_chiba')
        constraint: DTW constraint type ('none', 'sakoe_chiba', 'itakura')
        use_delta: If True, append delta features
    """
    return DTWRetriever(
        name="M5_MFCC_DTW",
        device=device,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        sakoe_chiba_radius=sakoe_chiba_radius,
        constraint=constraint,
        use_delta=use_delta,
        **kwargs
    )


def create_method_m5_fast(
    device: str = 'cpu',
    sr: int = 22050,
    n_mfcc: int = 13,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
    fastdtw_radius: int = 10,
    use_delta: bool = False,
    **kwargs
) -> DTWRetriever:
    """
    Create M5 variant with FastDTW: O(N) approximate DTW.

    FastDTW uses a coarse-to-fine approach to achieve linear time complexity
    while maintaining good approximation quality.

    Args:
        device: Device (DTW uses CPU)
        sr: Sample rate
        n_mfcc: Number of MFCCs
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length
        fastdtw_radius: Radius for path expansion (larger = more accurate but slower)
        use_delta: If True, append delta features
    """
    return DTWRetriever(
        name="M5_MFCC_FastDTW",
        device=device,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        constraint='fastdtw',
        fastdtw_radius=fastdtw_radius,
        use_delta=use_delta,
        **kwargs
    )
