"""
Supervised Contrastive Learning for audio representation.

Based on: "Supervised Contrastive Learning" (Khosla et al., 2020)
https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Extends contrastive loss to leverage label information:
    - Positives: samples of the same class
    - Negatives: samples of different classes

    L = -sum_i (1/|P(i)|) * sum_{p in P(i)} log(exp(z_i . z_p / t) / sum_a exp(z_i . z_a / t))

    where P(i) is the set of positives for anchor i.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07,
    ):
        """
        Initialize SupCon loss.

        Args:
            temperature: Temperature for scaling similarities
            contrast_mode: 'one' or 'all' (use one or all samples as anchors)
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute SupCon loss.

        Args:
            features: L2-normalized features (batch_size, embed_dim) or
                      (batch_size, n_views, embed_dim) for multi-view
            labels: Ground truth labels (batch_size,)
            mask: Optional contrastive mask (batch_size, batch_size)

        Returns:
            Scalar loss value
        """
        device = features.device

        if features.dim() == 2:
            # Single view: (batch, embed_dim) -> (batch, 1, embed_dim)
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # Create mask: same class = 1, different class = 0
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Contrast features: (batch * n_views, embed_dim)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            # Use first view as anchor
            anchor_features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # Use all views as anchors
            anchor_features = contrast_features
            anchor_count = n_views
        else:
            raise ValueError(f'Unknown contrast mode: {self.contrast_mode}')

        # Compute similarity matrix
        # (batch * anchor_count, embed_dim) @ (embed_dim, batch * n_views)
        anchor_dot_contrast = torch.matmul(
            anchor_features, contrast_features.T
        ) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask for multi-view
        mask = mask.repeat(anchor_count, n_views)

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(
            anchor_count * batch_size, n_views * batch_size
        ).to(device)
        if anchor_count * batch_size == n_views * batch_size:
            logits_mask = torch.scatter(
                logits_mask,
                1,
                torch.arange(anchor_count * batch_size).view(-1, 1).to(device),
                0
            )

        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute mean of log-likelihood over positives
        # Avoid division by zero
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6,
            torch.ones_like(mask_pos_pairs),
            mask_pos_pairs
        )
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveEncoder(nn.Module):
    """
    Encoder network for supervised contrastive learning.

    Architecture:
        CNN encoder -> projection head

    The projection head maps features to a lower-dimensional space
    where contrastive learning is performed.
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 128,
        base_channels: int = 64,
        proj_dim: int = 128,
    ):
        """
        Initialize contrastive encoder.

        Args:
            n_mels: Number of mel frequency bins
            embed_dim: Embedding dimension before projection
            base_channels: Base number of channels
            proj_dim: Projection head output dimension
        """
        super().__init__()

        self.n_mels = n_mels
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # Encoder (CNN backbone)
        self.encoder = nn.Sequential(
            # Block 1: (1, 128, T) -> (64, 64, T/2)
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (64, 64, T/2) -> (128, 32, T/4)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (128, 32, T/4) -> (256, 16, T/8)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: (256, 16, T/8) -> (512, 8, T/16)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Embedding layer
        self.fc_embed = nn.Linear(base_channels * 8, embed_dim)

        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

        self.base_channels = base_channels

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input mel spectrogram (batch, 1, n_mels, n_frames) or (batch, n_mels, n_frames)
            return_embedding: If True, return pre-projection embedding

        Returns:
            L2-normalized projection (batch, proj_dim) or embedding (batch, embed_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        # Embedding
        embedding = self.fc_embed(h)

        if return_embedding:
            return embedding

        # Project and normalize
        projection = self.projection(embedding)
        projection = F.normalize(projection, p=2, dim=1)

        return projection

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for retrieval.

        Args:
            x: Input mel spectrogram

        Returns:
            L2-normalized embedding (batch, embed_dim)
        """
        embedding = self.forward(x, return_embedding=True)
        return F.normalize(embedding, p=2, dim=1)

    def get_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get projection for contrastive learning.

        Args:
            x: Input mel spectrogram

        Returns:
            L2-normalized projection (batch, proj_dim)
        """
        return self.forward(x, return_embedding=False)
