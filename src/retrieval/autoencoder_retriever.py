"""
Autoencoder-based retrieval method.

Uses trained autoencoder latent space for audio retrieval.
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
from src.dsp_core import log_melspectrogram


class AutoencoderRetriever(BaseRetriever):
    """
    Retrieval using autoencoder latent space.

    The autoencoder is trained to reconstruct mel spectrograms.
    The latent vector captures a compressed representation of the audio.
    """

    def __init__(
        self,
        model_path: str,
        name: str = "Autoencoder",
        device: str = 'cuda',
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_frames: int = 216,
    ):
        """
        Initialize autoencoder retriever.

        Args:
            model_path: Path to trained autoencoder checkpoint
            name: Method name
            device: Device for computation
            sr: Sample rate
            n_mels: Number of mel frequency bins
            n_fft: FFT size
            hop_length: Hop length
            n_frames: Number of time frames (for padding/truncation)
        """
        super().__init__(name=name, device=device, sr=sr)

        self.model_path = model_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frames = n_frames

        # Load model
        self._load_model()

    def _load_model(self):
        """Load trained autoencoder model."""
        from src.models.autoencoder import MelAutoencoder

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Get model config from checkpoint if available
        config = checkpoint.get('config', {})
        n_mels = config.get('n_mels', self.n_mels)
        n_frames = config.get('n_frames', self.n_frames)
        latent_dim = config.get('latent_dim', 256)

        # Create and load model
        self.model = MelAutoencoder(
            n_mels=n_mels,
            n_frames=n_frames,
            latent_dim=latent_dim,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.latent_dim = latent_dim

    def _compute_mel(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        """Compute mel spectrogram from waveform."""
        # Compute log mel spectrogram
        mel = log_melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Pad or truncate to fixed length
        if mel.shape[1] < self.n_frames:
            pad_width = self.n_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel = mel[:, :self.n_frames]

        # Normalize to [0, 1]
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max - mel_min > 1e-10:
            mel = (mel - mel_min) / (mel_max - mel_min)

        return torch.from_numpy(mel).float()

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """
        Extract autoencoder latent features from waveform.

        Args:
            waveform: Audio waveform tensor
            sr: Sample rate

        Returns:
            Latent embedding
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

        # Compute mel spectrogram
        mel = self._compute_mel(waveform_np, sr)
        mel = mel.unsqueeze(0).to(self.device)  # Add batch dim

        # Extract latent embedding
        with torch.no_grad():
            embedding = self.model.get_embedding(mel)

        return embedding.squeeze(0)

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine distance."""
        # L2 normalize
        query_norm = query_features / (query_features.norm() + 1e-10)
        gallery_norm = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-10)

        # Cosine similarity -> distance
        similarity = torch.matmul(gallery_norm, query_norm)
        distance = 1 - similarity

        return distance

    def to(self, device: str) -> 'AutoencoderRetriever':
        """Move retriever to device."""
        self.device = device
        self.model.to(device)
        if self._gallery_features is not None:
            self._gallery_features = self._gallery_features.to(device)
        if self._gallery_labels is not None:
            self._gallery_labels = self._gallery_labels.to(device)
        return self


def create_autoencoder_retriever(
    model_path: str,
    device: str = 'cuda',
    **kwargs
) -> AutoencoderRetriever:
    """
    Create autoencoder retriever.

    Args:
        model_path: Path to trained autoencoder checkpoint
        device: Computation device
        **kwargs: Additional arguments

    Returns:
        AutoencoderRetriever instance
    """
    return AutoencoderRetriever(
        model_path=model_path,
        device=device,
        **kwargs
    )
