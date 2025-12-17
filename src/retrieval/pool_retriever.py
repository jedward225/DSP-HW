"""
Pool-based retrieval methods (M1, M2, M3, M4).

These methods extract frame-level features, apply global pooling,
and use vector distance metrics for retrieval.
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Literal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.base import BaseRetriever
from src.dsp_core import mfcc, melspectrogram, log_melspectrogram, delta, stft, power_to_db
from src.features.pooling import mean_std_pool, statistics_pool
from src.features.delta import add_deltas
from src.features.spectral import extract_spectral_features


class PoolRetriever(BaseRetriever):
    """
    Retriever using global pooling of frame-level features.

    Supports different feature types, pooling strategies, and distance metrics.
    """

    def __init__(
        self,
        name: str = "PoolRetriever",
        feature_type: Literal['mfcc', 'mfcc_delta', 'logmel', 'spectral'] = 'mfcc',
        pooling: Literal['mean_std', 'statistics'] = 'mean_std',
        distance: Literal['cosine', 'euclidean', 'l1', 'correlation', 'mahalanobis'] = 'cosine',
        device: str = 'cpu',
        sr: int = 22050,
        # Feature extraction parameters
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0.0,
        fmax: float = None,
        # Delta parameters
        delta_width: int = 9,
        # Pooling parameters
        pool_stats: List[str] = None,
    ):
        """
        Initialize pool-based retriever.

        Args:
            name: Method name
            feature_type: Type of features to extract
            pooling: Pooling strategy
            distance: Distance metric
            device: Computation device
            sr: Sample rate
            n_mfcc: Number of MFCCs
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency
            delta_width: Width for delta computation
            pool_stats: Statistics for 'statistics' pooling
        """
        super().__init__(name=name, device=device, sr=sr)

        self.feature_type = feature_type
        self.pooling = pooling
        self.distance = distance

        # Feature parameters
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2
        self.delta_width = delta_width
        self.pool_stats = pool_stats or ['mean', 'std']

        # For Mahalanobis distance
        self._inv_cov = None

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract pooled features from waveform.

        Args:
            waveform: Audio waveform tensor
            sr: Sample rate (uses self.sr if not provided)

        Returns:
            Fixed-length feature vector
        """
        sr = sr or self.sr

        # Convert to numpy for dsp_core functions
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Extract frame-level features
        if self.feature_type == 'mfcc':
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
            features = torch.from_numpy(features).float().to(self.device)

        elif self.feature_type == 'mfcc_delta':
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
            features = torch.from_numpy(features).float().to(self.device)
            # Add delta and delta-delta
            features = add_deltas(
                features,
                width=self.delta_width,
                include_delta=True,
                include_delta_delta=True
            )

        elif self.feature_type == 'logmel':
            features = log_melspectrogram(
                y=waveform_np,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            features = torch.from_numpy(features).float().to(self.device)

        elif self.feature_type == 'spectral':
            # Extract STFT first
            S = np.abs(stft(
                y=waveform_np,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            S = torch.from_numpy(S).float().to(self.device)

            # Get frequencies
            freqs = torch.linspace(0, sr / 2, S.shape[0], device=self.device)

            # Extract spectral features
            spectral_feats = extract_spectral_features(
                S, freqs,
                y=waveform if isinstance(waveform, torch.Tensor) else torch.from_numpy(waveform_np).to(self.device),
                sr=sr,
                hop_length=self.hop_length
            )

            # Stack all features
            features = torch.stack([
                spectral_feats['spectral_centroid'],
                spectral_feats['spectral_bandwidth'],
                spectral_feats['spectral_rolloff'],
                spectral_feats['spectral_flatness'],
                spectral_feats['spectral_flux'],
            ], dim=0)

            if 'zcr' in spectral_feats:
                features = torch.cat([
                    features,
                    spectral_feats['zcr'].unsqueeze(0),
                    spectral_feats['rms'].unsqueeze(0)
                ], dim=0)

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        # Apply pooling
        if self.pooling == 'mean_std':
            pooled = mean_std_pool(features, dim=-1)
        elif self.pooling == 'statistics':
            pooled = statistics_pool(features, dim=-1, stats=self.pool_stats)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query and gallery.

        Args:
            query_features: Query feature vector
            gallery_features: Gallery feature matrix (n_gallery, n_features)

        Returns:
            Distance tensor (n_gallery,)
        """
        if self.distance == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            query_norm = query_features / (query_features.norm() + 1e-10)
            gallery_norm = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-10)
            similarity = torch.matmul(gallery_norm, query_norm)
            return 1 - similarity

        elif self.distance == 'euclidean':
            # Euclidean distance
            diff = gallery_features - query_features.unsqueeze(0)
            return torch.sqrt((diff ** 2).sum(dim=1))

        elif self.distance == 'l1':
            # Manhattan distance
            diff = gallery_features - query_features.unsqueeze(0)
            return torch.abs(diff).sum(dim=1)

        elif self.distance == 'correlation':
            # Correlation distance = 1 - correlation
            query_centered = query_features - query_features.mean()
            gallery_centered = gallery_features - gallery_features.mean(dim=1, keepdim=True)

            query_norm = query_centered / (query_centered.norm() + 1e-10)
            gallery_norm = gallery_centered / (gallery_centered.norm(dim=1, keepdim=True) + 1e-10)

            correlation = torch.matmul(gallery_norm, query_norm)
            return 1 - correlation

        elif self.distance == 'mahalanobis':
            # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
            # Compute inverse covariance matrix from gallery if not cached
            if self._inv_cov is None:
                centered = gallery_features - gallery_features.mean(dim=0)
                cov = torch.mm(centered.T, centered) / (gallery_features.shape[0] - 1)
                # Add regularization for numerical stability
                cov = cov + 1e-5 * torch.eye(cov.shape[0], device=cov.device)
                self._inv_cov = torch.linalg.inv(cov)

            diff = gallery_features - query_features.unsqueeze(0)
            # Mahalanobis: sqrt((x-y)ᵀ Σ⁻¹ (x-y))
            mahal_sq = (torch.mm(diff, self._inv_cov) * diff).sum(dim=1)
            return torch.sqrt(torch.clamp(mahal_sq, min=0))

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def clear_gallery(self):
        """Clear the gallery cache and covariance matrix."""
        super().clear_gallery()
        self._inv_cov = None


def create_method_m1(
    device: str = 'cpu',
    sr: int = 22050,
    n_mfcc: int = 20,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs
) -> PoolRetriever:
    """
    Create M1: MFCC + Global Pooling + Cosine Distance.

    This is the basic baseline method.
    """
    return PoolRetriever(
        name="M1_MFCC_Pool_Cos",
        feature_type='mfcc',
        pooling='mean_std',
        distance='cosine',
        device=device,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        **kwargs
    )


def create_method_m2(
    device: str = 'cpu',
    sr: int = 22050,
    n_mfcc: int = 20,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    delta_width: int = 9,
    **kwargs
) -> PoolRetriever:
    """
    Create M2: MFCC + Delta + Delta-Delta + Global Pooling + Cosine Distance.

    Enhanced baseline with temporal dynamics.
    """
    return PoolRetriever(
        name="M2_MFCC_Delta_Pool",
        feature_type='mfcc_delta',
        pooling='mean_std',
        distance='cosine',
        device=device,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        delta_width=delta_width,
        **kwargs
    )


def create_method_m3(
    device: str = 'cpu',
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs
) -> PoolRetriever:
    """
    Create M3: Log-Mel Spectrogram + Global Pooling + Cosine Distance.

    Uses mel spectrogram instead of MFCC.
    """
    return PoolRetriever(
        name="M3_LogMel_Pool",
        feature_type='logmel',
        pooling='mean_std',
        distance='cosine',
        device=device,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        **kwargs
    )


def create_method_m4(
    device: str = 'cpu',
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    **kwargs
) -> PoolRetriever:
    """
    Create M4: Spectral Statistics + L2 Distance.

    Uses interpretable spectral features.
    """
    return PoolRetriever(
        name="M4_Spectral_Stat",
        feature_type='spectral',
        pooling='statistics',
        distance='euclidean',
        device=device,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        pool_stats=['mean', 'std', 'max', 'min'],
        **kwargs
    )
