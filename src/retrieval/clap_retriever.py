"""
CLAP-based audio retrieval method (M8).

Uses pretrained CLAP (Contrastive Language-Audio Pretraining) model
for audio embeddings with cosine distance.
"""

import torch
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.base import BaseRetriever


class CLAPRetriever(BaseRetriever):
    """
    M8: CLAP embedding retrieval.

    Uses pretrained CLAP model for audio embeddings with cosine distance.
    CLAP produces semantically meaningful embeddings trained on audio-text pairs.

    Note:
        - CLAP requires 48kHz audio input
        - Embeddings are 512-dimensional
        - Uses cosine distance (trained with contrastive learning)
    """

    def __init__(
        self,
        name: str = "M8_CLAP",
        device: str = 'cuda',
        sr: int = 48000,
        checkpoint_path: str = None,
        enable_fusion: bool = False,
        amodel: str = 'HTSAT-base',
    ):
        """
        Initialize CLAP retriever.

        Args:
            name: Method name
            device: Device for computation ('cpu' or 'cuda')
            sr: Sample rate (internally resamples to 48kHz)
            checkpoint_path: Path to CLAP checkpoint (required)
            enable_fusion: Enable fusion CLAP model
            amodel: Audio model architecture ('HTSAT-tiny' or 'HTSAT-base')
        """
        super().__init__(name=name, device=device, sr=sr)

        if checkpoint_path is None:
            # Default checkpoint path
            checkpoint_path = str(project_root / 'laion_clap' / 'music_audioset_epoch_15_esc_90.14.pt')

        self.checkpoint_path = checkpoint_path
        self.enable_fusion = enable_fusion
        self.amodel = amodel
        self.clap_sr = 48000  # CLAP requires 48kHz

        # Load CLAP model
        self._load_clap_model()

    def _load_clap_model(self):
        """Load the CLAP model from checkpoint."""
        # Add CLAP source to path
        clap_src_path = project_root / 'CLAP' / 'src'
        if str(clap_src_path) not in sys.path:
            sys.path.insert(0, str(clap_src_path))

        from laion_clap import CLAP_Module

        self.clap_model = CLAP_Module(
            enable_fusion=self.enable_fusion,
            device=self.device,
            amodel=self.amodel
        )
        self.clap_model.load_ckpt(ckpt=self.checkpoint_path, verbose=False)
        self.clap_model.eval()

    def _resample_audio(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to CLAP's required 48kHz."""
        if orig_sr == self.clap_sr:
            return waveform

        # Use librosa for resampling (allowed in this context since CLAP uses it)
        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.clap_sr)

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract CLAP embeddings from audio waveform.

        Args:
            waveform: Audio waveform tensor
            sr: Sample rate (uses self.sr if not provided)

        Returns:
            512-dimensional CLAP embedding
        """
        sr = sr or self.sr

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.flatten()

        # Resample to 48kHz if needed
        waveform_np = self._resample_audio(waveform_np, sr)

        # Get embedding from CLAP
        # CLAP expects batch of audio: (N, T)
        with torch.no_grad():
            embedding = self.clap_model.get_audio_embedding_from_data(
                [waveform_np],
                use_tensor=False
            )

        # Convert to tensor: (1, 512) -> (512,)
        embedding = torch.from_numpy(embedding[0]).float().to(self.device)

        return embedding

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine distance between query and gallery features.

        Args:
            query_features: Query embedding (512,)
            gallery_features: Gallery embeddings (N, 512)

        Returns:
            Cosine distances (N,)
        """
        # L2 normalize
        query_norm = query_features / (query_features.norm() + 1e-10)
        gallery_norm = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-10)

        # Cosine similarity -> distance
        similarity = torch.matmul(gallery_norm, query_norm)
        distance = 1 - similarity

        return distance

    def to(self, device: str) -> 'CLAPRetriever':
        """Move retriever to device."""
        self.device = device
        # Note: CLAP model handles device internally
        if self._gallery_features is not None:
            self._gallery_features = self._gallery_features.to(device)
        if self._gallery_labels is not None:
            self._gallery_labels = self._gallery_labels.to(device)
        return self


def create_method_m8(
    device: str = 'cuda',
    checkpoint_path: str = None,
    enable_fusion: bool = False,
    amodel: str = 'HTSAT-base',
    **kwargs
) -> CLAPRetriever:
    """
    Create M8: CLAP embedding retrieval.

    Uses pretrained CLAP model for semantically meaningful audio embeddings.

    Args:
        device: Computation device
        checkpoint_path: Path to CLAP checkpoint (uses default if None)
        enable_fusion: Enable fusion CLAP model
        amodel: Audio model architecture
        **kwargs: Additional arguments

    Returns:
        CLAPRetriever instance
    """
    return CLAPRetriever(
        name="M8_CLAP",
        device=device,
        sr=48000,  # CLAP requires 48kHz
        checkpoint_path=checkpoint_path,
        enable_fusion=enable_fusion,
        amodel=amodel,
    )
