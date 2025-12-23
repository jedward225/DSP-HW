"""
BEATs-based audio retrieval.

Uses pretrained BEATs (Audio Pre-Training with Acoustic Tokenizers) model
for audio embeddings with cosine distance.

Note on Architecture Convention:
    This module is an exception to the "only dsp_core imports librosa/scipy" rule.
    BEATs requires resampling to 16kHz and uses librosa for this purpose.
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


class BEATsRetriever(BaseRetriever):
    """
    BEATs embedding retrieval.

    Uses pretrained BEATs transformer model for audio embeddings.
    BEATs is a self-supervised audio representation model trained
    with acoustic tokenizers.

    Note:
        - BEATs requires 16kHz audio input
        - Embeddings are 768-dimensional (transformer hidden size)
        - Uses cosine distance
    """

    def __init__(
        self,
        name: str = "BEATs",
        device: Optional[str] = None,
        sr: int = 22050,
        checkpoint_path: str = None,
        layer: int = -1,
        pooling: str = 'mean',
    ):
        """
        Initialize BEATs retriever.

        Args:
            name: Method name
            device: Device for computation ('cpu' or 'cuda')
            sr: Input sample rate (audio will be resampled to 16kHz internally)
            checkpoint_path: Path to BEATs checkpoint (required)
            layer: Which transformer layer to use (-1 = last)
            pooling: Pooling method for temporal dimension ('mean', 'max', 'cls')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(name=name, device=device, sr=sr)

        if checkpoint_path is None:
            checkpoint_path = str(project_root / 'beats' / 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

        self.checkpoint_path = checkpoint_path
        # Note: layer parameter reserved for future intermediate layer extraction
        # Currently BEATs always uses final layer output
        self.layer = layer
        self.pooling = pooling
        self.beats_sr = 16000  # BEATs requires 16kHz

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {self.checkpoint_path}")

        # Load BEATs model
        self._load_beats_model()

    def _load_beats_model(self):
        """Load the BEATs model from checkpoint."""
        # Add BEATs source to path
        beats_src_path = project_root / 'unilm' / 'beats'
        if not beats_src_path.exists():
            raise FileNotFoundError(f"BEATs source not found at: {beats_src_path}")
        if str(beats_src_path) not in sys.path:
            sys.path.insert(0, str(beats_src_path))

        from BEATs import BEATs, BEATsConfig

        # Load checkpoint
        # Use weights_only=False for PyTorch 2.6+ compatibility
        # BEATs checkpoints contain custom objects that need unpickling
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.to(self.device)
        self.beats_model.eval()

        # For retrieval we want encoder embeddings. Some checkpoints are fine-tuned and
        # include a classification head; BEATs.extract_features() would then return class
        # probabilities instead of embeddings, so we drop the head.
        if getattr(self.beats_model, 'predictor', None) is not None:
            self.beats_model.predictor = None
            if hasattr(self.beats_model, 'predictor_dropout'):
                self.beats_model.predictor_dropout = None

    def _resample_audio(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to BEATs' required 16kHz."""
        if orig_sr == self.beats_sr:
            return waveform

        # Use librosa for resampling
        import librosa
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.beats_sr)

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract BEATs embeddings from audio waveform.

        Args:
            waveform: Audio waveform tensor
            sr: Sample rate (uses self.sr if not provided)

        Returns:
            768-dimensional BEATs embedding
        """
        sr = sr or self.sr

        # Convert to numpy for resampling
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Ensure 1D
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.flatten()

        # Resample to 16kHz if needed
        waveform_np = self._resample_audio(waveform_np, sr)

        # Convert to tensor
        # BEATs expects normalized audio in [-1, 1] range (NOT scaled by 2^15)
        # The model was pretrained with this normalization convention
        audio_tensor = torch.from_numpy(waveform_np).float().unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            # BEATs extract_features returns (features, padding_mask)
            # If the checkpoint is fine-tuned, BEATs returns class probabilities
            # with shape (batch, num_classes). Otherwise it returns embeddings with
            # shape (batch, time, dim).
            features, _ = self.beats_model.extract_features(audio_tensor)

            if features.dim() == 3:
                # Pool across time dimension
                if self.pooling == 'mean':
                    embedding = features.mean(dim=1)
                elif self.pooling == 'max':
                    embedding = features.max(dim=1)[0]
                elif self.pooling == 'cls':
                    # Use first token as CLS
                    embedding = features[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")
            elif features.dim() == 2:
                # Already pooled (e.g., fine-tuned checkpoints return class probabilities)
                embedding = features
            else:
                raise ValueError(f"Unexpected BEATs feature shape: {tuple(features.shape)}")

        # Return (D,) tensor
        return embedding.squeeze(0)

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine distance between query and gallery features.

        Args:
            query_features: Query embedding (768,) or (1, 768)
            gallery_features: Gallery embeddings (N, 768)

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

    def to(self, device: str) -> 'BEATsRetriever':
        """Move retriever to device."""
        self.device = device
        self.beats_model.to(device)
        if self._gallery_features is not None:
            self._gallery_features = self._gallery_features.to(device)
        if self._gallery_labels is not None:
            self._gallery_labels = self._gallery_labels.to(device)
        return self


def create_beats_retriever(
    device: Optional[str] = None,
    checkpoint_path: str = None,
    pooling: str = 'mean',
    **kwargs
) -> BEATsRetriever:
    """
    Create BEATs embedding retriever.

    Uses pretrained BEATs transformer for audio embeddings.

    Args:
        device: Computation device
        checkpoint_path: Path to BEATs checkpoint (uses default if None)
        pooling: Pooling method ('mean', 'max', 'cls')
        **kwargs: Additional arguments

    Returns:
        BEATsRetriever instance
    """
    return BEATsRetriever(
        name="BEATs",
        device=device,
        sr=22050,  # Input SR from dataset; internally resampled to 16kHz
        checkpoint_path=checkpoint_path,
        pooling=pooling,
    )
