"""
CLAP-based audio retrieval method (M8).

Uses pretrained CLAP (Contrastive Language-Audio Pretraining) model
for audio embeddings with cosine distance.

Note on Architecture Convention:
    This module is an exception to the "only dsp_core imports librosa/scipy" rule.
    CLAP requires resampling to 48kHz and uses librosa for this purpose.
    This exception is documented because deep learning retrievers have specialized
    preprocessing requirements that differ from the traditional DSP pipeline.
"""

import torch
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent

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
        device: Optional[str] = None,
        sr: int = 22050,
        checkpoint_path: str = None,
        enable_fusion: bool = False,
        amodel: str = 'HTSAT-base',
    ):
        """
        Initialize CLAP retriever.

        Args:
            name: Method name
            device: Device for computation ('cpu' or 'cuda')
            sr: Input sample rate (audio will be resampled to 48kHz internally)
            checkpoint_path: Path to CLAP checkpoint (required)
            enable_fusion: Enable fusion CLAP model
            amodel: Audio model architecture ('HTSAT-tiny' or 'HTSAT-base')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(name=name, device=device, sr=sr)

        if checkpoint_path is None:
            # Default checkpoint path
            checkpoint_path = str(project_root / 'laion_clap' / 'music_audioset_epoch_15_esc_90.14.pt')

        self.checkpoint_path = checkpoint_path
        self.enable_fusion = enable_fusion
        self.amodel = amodel
        self.clap_sr = 48000  # CLAP requires 48kHz

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"CLAP checkpoint not found: {self.checkpoint_path}")

        # Load CLAP model
        self._load_clap_model()

    def _load_clap_model(self):
        """Load the CLAP model from checkpoint."""
        # Add CLAP source to path
        clap_src_path = project_root / 'CLAP' / 'src'
        if not clap_src_path.exists():
            raise FileNotFoundError(f"CLAP source not found at: {clap_src_path}")
        if str(clap_src_path) not in sys.path:
            sys.path.insert(0, str(clap_src_path))

        from laion_clap import CLAP_Module

        self.clap_model = CLAP_Module(
            enable_fusion=self.enable_fusion,
            device=self.device,
            amodel=self.amodel
        )

        # Workaround for PyTorch 2.6+ weights_only=True default
        # CLAP checkpoints contain numpy scalars that need weights_only=False
        import functools
        original_load = torch.load
        torch.load = functools.partial(original_load, weights_only=False)
        try:
            self.clap_model.load_ckpt(ckpt=self.checkpoint_path, verbose=False)
        finally:
            torch.load = original_load

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
            query_features: Query embedding (512,) or (1, 512)
            gallery_features: Gallery embeddings (N, 512)

        Returns:
            Cosine distances (N,)
        """
        # Flatten to ensure 1D for query
        query_features = query_features.flatten()

        # Ensure gallery is 2D (N, D)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        elif gallery_features.dim() > 2:
            gallery_features = gallery_features.view(-1, gallery_features.shape[-1])

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
    device: Optional[str] = None,
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
        sr=22050,  # Input SR from dataset; internally resampled to 48kHz
        checkpoint_path=checkpoint_path,
        enable_fusion=enable_fusion,
        amodel=amodel,
    )
