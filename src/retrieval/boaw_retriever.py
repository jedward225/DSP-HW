"""
Bag-of-Audio-Words retrieval method (M6).

This method quantizes frame-level features into a codebook
and represents each audio as a histogram of codeword occurrences.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple

from src.retrieval.base import BaseRetriever
from src.dsp_core import mfcc


class KMeansClusterer:
    """
    K-means clustering implemented in PyTorch.

    Used to learn the audio codebook.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        device: str = 'cpu'
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = device

        self.cluster_centers_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> 'KMeansClusterer':
        """
        Fit K-means to data.

        Args:
            X: Data tensor of shape (n_samples, n_features)

        Returns:
            self
        """
        torch.manual_seed(self.random_state)

        X = X.to(self.device)
        n_samples, n_features = X.shape

        # Initialize centers using k-means++
        centers = self._kmeans_plusplus_init(X)

        for iteration in range(self.max_iter):
            # Assign points to nearest center
            distances = self._compute_distances(X, centers)
            assignments = torch.argmin(distances, dim=1)

            # Update centers
            new_centers = torch.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    # Empty cluster - reinitialize randomly
                    # Use .item() to get scalar index (avoids shape mismatch)
                    new_centers[k] = X[torch.randint(n_samples, (1,)).item()]

            # Check convergence
            center_shift = ((new_centers - centers) ** 2).sum()
            centers = new_centers

            if center_shift < self.tol:
                break

        self.cluster_centers_ = centers
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments for data.

        Args:
            X: Data tensor of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = X.to(self.device)
        distances = self._compute_distances(X, self.cluster_centers_)
        return torch.argmin(distances, dim=1)

    def _kmeans_plusplus_init(self, X: torch.Tensor) -> torch.Tensor:
        """Initialize centers using k-means++."""
        n_samples = X.shape[0]
        centers = []

        # First center is random
        idx = torch.randint(n_samples, (1,)).item()
        centers.append(X[idx])

        for _ in range(1, self.n_clusters):
            # Compute distances to nearest center
            centers_tensor = torch.stack(centers)
            distances = self._compute_distances(X, centers_tensor)
            min_distances = distances.min(dim=1).values

            # Sample proportional to squared distance
            probs = min_distances ** 2
            total = probs.sum()
            if total <= 0 or not torch.isfinite(total):
                idx = torch.randint(n_samples, (1,)).item()
            else:
                probs = probs / total
                idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])

        return torch.stack(centers)

    def _compute_distances(self, X: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between X and centers."""
        # (n_samples, n_features) vs (n_clusters, n_features)
        # Output: (n_samples, n_clusters)
        return torch.cdist(X, centers, p=2)


class BoAWRetriever(BaseRetriever):
    """
    Bag-of-Audio-Words retriever.

    Represents audio as histograms over a learned codebook.
    """

    def __init__(
        self,
        name: str = "BoAWRetriever",
        device: str = 'cpu',
        sr: int = 22050,
        # Feature extraction parameters
        n_mfcc: int = 13,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0.0,
        fmax: float = None,
        # BoAW parameters
        n_clusters: int = 128,
        normalize_hist: bool = True,
        distance: str = 'chi_squared',  # 'chi_squared', 'euclidean', 'cosine'
        random_state: int = 42,
    ):
        """
        Initialize BoAW retriever.

        Args:
            name: Method name
            device: Computation device
            sr: Sample rate
            n_mfcc: Number of MFCCs
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency
            n_clusters: Number of codewords in codebook
            normalize_hist: If True, normalize histograms to sum to 1
            distance: Distance metric for histograms
            random_state: Random seed for reproducibility
        """
        super().__init__(name=name, device=device, sr=sr)

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.n_clusters = n_clusters
        self.normalize_hist = normalize_hist
        self.distance_type = distance
        self.random_state = random_state

        # Codebook (learned from training data)
        self.codebook: Optional[KMeansClusterer] = None
        self._codebook_fitted = False

    def _extract_frame_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> np.ndarray:
        """Extract frame-level MFCC features."""
        sr = sr or self.sr

        fmax = self.fmax
        if fmax is None:
            fmax = sr / 2
        else:
            fmax = min(float(fmax), sr / 2)

        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        features = mfcc(
            y=waveform_np,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=fmax
        )

        # Transpose to (n_frames, n_features)
        return features.T

    def fit_codebook(
        self,
        samples: List[dict],
        max_frames: int = 100000,
        show_progress: bool = False
    ):
        """
        Learn codebook from training samples.

        Args:
            samples: List of sample dictionaries
            max_frames: Maximum number of frames to use for clustering
            show_progress: If True, show progress
        """
        all_frames = []

        iterator = samples
        if show_progress:
            from rich.progress import track
            iterator = track(samples, description="Extracting frames for codebook...")

        for sample in iterator:
            waveform = sample['waveform']
            sample_sr = sample.get('sr', self.sr)
            frames = self._extract_frame_features(waveform, sample_sr)
            all_frames.append(frames)

        # Concatenate all frames
        all_frames = np.vstack(all_frames)

        # Subsample if too many frames (use seed for reproducibility)
        if len(all_frames) > max_frames:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(all_frames), max_frames, replace=False)
            all_frames = all_frames[indices]

        # Fit k-means
        frames_tensor = torch.from_numpy(all_frames).float().to(self.device)
        self.codebook = KMeansClusterer(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            device=self.device
        )
        self.codebook.fit(frames_tensor)
        self._codebook_fitted = True

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract BoAW histogram from waveform.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Histogram tensor of shape (n_clusters,)
        """
        if not self._codebook_fitted:
            raise RuntimeError("Codebook not fitted. Call fit_codebook() first.")

        sr = sr or self.sr

        # Extract frame features
        frames = self._extract_frame_features(waveform, sr)
        frames_tensor = torch.from_numpy(frames).float().to(self.device)

        # Assign frames to codewords
        assignments = self.codebook.predict(frames_tensor)

        # Build histogram using vectorized bincount (much faster than Python loop)
        histogram = torch.bincount(
            assignments, minlength=self.n_clusters
        ).float().to(self.device)

        # Normalize
        if self.normalize_hist:
            histogram = histogram / (histogram.sum() + 1e-10)

        return histogram

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query and gallery histograms.

        Args:
            query_features: Query histogram (n_clusters,)
            gallery_features: Gallery histograms (n_gallery, n_clusters)

        Returns:
            Distance tensor (n_gallery,)
        """
        if self.distance_type == 'chi_squared':
            # Chi-squared distance
            # χ²(p, q) = 0.5 * Σ (p_i - q_i)² / (p_i + q_i)
            query = query_features.unsqueeze(0)  # (1, n_clusters)
            diff = (query - gallery_features) ** 2
            sum_features = query + gallery_features + 1e-10
            chi_sq = 0.5 * (diff / sum_features).sum(dim=1)
            return chi_sq

        elif self.distance_type == 'euclidean':
            diff = gallery_features - query_features.unsqueeze(0)
            return torch.sqrt((diff ** 2).sum(dim=1))

        elif self.distance_type == 'cosine':
            query_norm = query_features / (query_features.norm() + 1e-10)
            gallery_norm = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-10)
            similarity = torch.matmul(gallery_norm, query_norm)
            return 1 - similarity

        elif self.distance_type == 'js':
            # Jensen-Shannon divergence
            query = query_features.unsqueeze(0) + 1e-10
            gallery = gallery_features + 1e-10
            m = 0.5 * (query + gallery)

            kl_qm = (query * torch.log(query / m)).sum(dim=1)
            kl_gm = (gallery * torch.log(gallery / m)).sum(dim=1)

            return 0.5 * (kl_qm + kl_gm)

        elif self.distance_type == 'kl':
            # KL divergence: KL(Q||G) = Σ q_i * log(q_i / g_i)
            query = query_features.unsqueeze(0) + 1e-10
            gallery = gallery_features + 1e-10
            # Normalize to probability distributions
            query = query / query.sum(dim=1, keepdim=True)
            gallery = gallery / gallery.sum(dim=1, keepdim=True)
            kl = (query * torch.log(query / gallery)).sum(dim=1)
            return kl

        elif self.distance_type == 'emd':
            # Earth Mover's Distance (Wasserstein-1)
            # Requires the 'pot' (Python Optimal Transport) library
            try:
                import ot
            except ImportError:
                raise ImportError(
                    "EMD distance requires the 'pot' library. "
                    "Install it with: pip install POT"
                )

            query_np = query_features.cpu().numpy()
            gallery_np = gallery_features.cpu().numpy()

            # Normalize to probability distributions
            query_np = query_np / (query_np.sum() + 1e-10)

            # Build cost matrix from pairwise distances between codebook centers
            # This captures the semantic similarity between codewords
            if self.codebook is not None and self.codebook.cluster_centers_ is not None:
                centers = self.codebook.cluster_centers_.cpu().numpy()
                # Pairwise Euclidean distance between cluster centers
                M = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
                # Normalize cost matrix to [0, 1] range for numerical stability
                if M.max() > 0:
                    M = M / M.max()
            else:
                # Fallback to uniform cost if codebook not available
                n_bins = query_np.shape[0]
                M = np.ones((n_bins, n_bins)) - np.eye(n_bins)

            distances = []
            for g in gallery_np:
                g_norm = g / (g.sum() + 1e-10)
                # Compute EMD (Wasserstein-1 distance)
                emd_dist = ot.emd2(query_np, g_norm, M)
                distances.append(emd_dist)

            return torch.tensor(distances, device=query_features.device, dtype=query_features.dtype)

        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def build_gallery(
        self,
        gallery_samples: List[dict],
        show_progress: bool = False
    ):
        """
        Build gallery with optional codebook fitting.

        If codebook is not fitted, fits it first using gallery samples.
        """
        if not self._codebook_fitted:
            self.fit_codebook(gallery_samples, show_progress=show_progress)

        # Now build gallery using parent method
        super().build_gallery(gallery_samples, show_progress=show_progress)

    def clear_gallery(self):
        """Clear gallery and reset codebook for proper cross-validation."""
        super().clear_gallery()
        self.codebook = None
        self._codebook_fitted = False


def create_method_m6(
    device: str = 'cpu',
    sr: int = 22050,
    n_mfcc: int = 13,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_clusters: int = 128,
    **kwargs
) -> BoAWRetriever:
    """
    Create M6: MFCC + Bag-of-Audio-Words + Chi-squared Distance.

    Represents audio as histogram over learned codebook.
    """
    return BoAWRetriever(
        name="M6_BoAW_ChiSq",
        device=device,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        n_clusters=n_clusters,
        normalize_hist=True,
        distance='chi_squared',
        **kwargs
    )
