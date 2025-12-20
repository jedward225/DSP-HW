"""
Deep learning models for audio retrieval.

This module provides trainable models:
- MelAutoencoder: Convolutional autoencoder for mel spectrogram reconstruction
- AudioCNN: CNN classifier for audio classification
- ContrastiveEncoder: Encoder for supervised contrastive learning
- SupConLoss: Supervised contrastive loss function
"""

from .autoencoder import MelAutoencoder
from .cnn_classifier import AudioCNN
from .contrastive import ContrastiveEncoder, SupConLoss

__all__ = [
    'MelAutoencoder',
    'AudioCNN',
    'ContrastiveEncoder',
    'SupConLoss',
]
