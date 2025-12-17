"""
CNN Audio Classifier for audio classification and retrieval.

The penultimate layer features are used for retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AudioCNN(nn.Module):
    """
    CNN classifier for audio (similar to VGGish/ResNet style).

    Architecture:
        Conv blocks with BatchNorm, ReLU, MaxPool
        Global average pooling
        FC layers: hidden_dim -> n_classes

    The penultimate layer (hidden_dim) is used for retrieval.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 50,
        hidden_dim: int = 512,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        """
        Initialize CNN classifier.

        Args:
            n_mels: Number of mel frequency bins
            n_classes: Number of output classes
            hidden_dim: Dimension of penultimate layer (embedding dim)
            base_channels: Base number of channels
            dropout: Dropout rate
        """
        super().__init__()

        self.n_mels = n_mels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: (1, 128, T) -> (64, 64, T/2)
            self._conv_block(1, base_channels, pool=True),

            # Block 2: (64, 64, T/2) -> (128, 32, T/4)
            self._conv_block(base_channels, base_channels * 2, pool=True),

            # Block 3: (128, 32, T/4) -> (256, 16, T/8)
            self._conv_block(base_channels * 2, base_channels * 4, pool=True),

            # Block 4: (256, 16, T/8) -> (512, 8, T/16)
            self._conv_block(base_channels * 4, base_channels * 8, pool=True),

            # Block 5: (512, 8, T/16) -> (512, 4, T/32)
            self._conv_block(base_channels * 8, base_channels * 8, pool=True),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head - structured so we can extract embeddings from fc1+relu
        self.fc1 = nn.Linear(base_channels * 8, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

        self.base_channels = base_channels

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True
    ) -> nn.Sequential:
        """Create a convolutional block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input mel spectrogram (batch, 1, n_mels, n_frames) or (batch, n_mels, n_frames)

        Returns:
            Class logits (batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional features
        x = self.conv_blocks(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification - fc1 -> relu -> dropout -> fc2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer features for retrieval.

        Args:
            x: Input mel spectrogram

        Returns:
            Embedding vector (batch, hidden_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional features
        x = self.conv_blocks(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Embedding (penultimate layer) - uses same fc1+relu as forward()
        # This ensures embeddings come from trained layers
        embedding = self.fc1(x)
        embedding = self.relu(embedding)
        return embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for extract_embedding."""
        return self.extract_embedding(x)

    def forward_with_embedding(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and embeddings.

        Args:
            x: Input mel spectrogram

        Returns:
            Tuple of (logits, embeddings)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional features
        x = self.conv_blocks(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Get embedding (fc1 + relu)
        embedding = self.fc1(x)
        embedding = self.relu(embedding)

        # Classification (using embedding as input)
        # Apply dropout and final layer
        x_drop = self.dropout(embedding)
        logits = self.fc2(x_drop)

        return logits, embedding
