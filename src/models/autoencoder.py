"""
Mel Spectrogram Autoencoder for audio representation learning.

The autoencoder learns to compress mel spectrograms into a latent space,
which can be used for retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MelAutoencoder(nn.Module):
    """
    Convolutional autoencoder for mel spectrograms.

    Architecture:
        Encoder: Conv2d -> BN -> ReLU -> ... -> Flatten -> Linear
        Decoder: Linear -> Unflatten -> ConvTranspose2d -> ... -> Sigmoid

    The latent vector from the encoder is used for retrieval.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_frames: int = 256,  # ~6 seconds at 22050Hz with hop_length=512 (must be divisible by 16)
        latent_dim: int = 256,
        base_channels: int = 32,
    ):
        """
        Initialize autoencoder.

        Args:
            n_mels: Number of mel frequency bins
            n_frames: Number of time frames
            latent_dim: Dimension of latent space
            base_channels: Base number of channels (doubled each layer)
        """
        super().__init__()

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # (1, 128, 256) -> (32, 64, 128)
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # (32, 64, 128) -> (64, 32, 64)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # (64, 32, 64) -> (128, 16, 32)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # (128, 16, 32) -> (256, 8, 16)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size after encoder
        # After 4 conv layers with stride 2: (H, W) -> (H/16, W/16)
        self.flat_h = n_mels // 16
        self.flat_w = n_frames // 16
        self.flat_size = base_channels * 8 * self.flat_h * self.flat_w

        # Latent space projection
        self.fc_encode = nn.Linear(self.flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        # Decoder
        self.decoder = nn.Sequential(
            # (256, 8, 16) -> (128, 16, 32)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # (128, 16, 32) -> (64, 32, 64)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # (64, 32, 64) -> (32, 64, 128)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # (32, 64, 128) -> (1, 128, 256)
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.base_channels = base_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent vector.

        Args:
            x: Input mel spectrogram (batch, 1, n_mels, n_frames) or (batch, n_mels, n_frames)

        Returns:
            Latent vector (batch, latent_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to mel spectrogram.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstructed mel spectrogram (batch, 1, n_mels, n_frames)
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.base_channels * 8, self.flat_h, self.flat_w)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input mel spectrogram

        Returns:
            Tuple of (reconstructed, latent_vector)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent embedding for retrieval.

        Args:
            x: Input mel spectrogram

        Returns:
            Latent embedding (batch, latent_dim)
        """
        return self.encode(x)
