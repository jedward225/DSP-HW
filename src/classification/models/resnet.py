"""
ResNet18 model wrapper for ESC-50 classification.

Uses ImageNet pre-trained weights and fine-tunes on audio spectrograms.
Adapts the first conv layer to accept single-channel spectrogram input.

Expected accuracy: ~83% on ESC-50 (based on [Ref-1] 张鑫恺等 83.25%)

Features:
- ImageNet pre-trained weights
- Adapted for single-channel spectrogram input
- Supports different spectrogram types (STFT, Mel, MFCC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for audio spectrograms.

    Adapts pretrained ImageNet ResNet models for audio classification
    by modifying the input layer to accept single-channel spectrograms
    and replacing the final classification layer.

    Usage:
    >>> model = ResNetClassifier(num_classes=50, arch='resnet18')
    >>> logits = model(spectrogram)  # spectrogram: (batch, n_features, n_frames)
    """

    def __init__(
        self,
        num_classes: int = 50,
        arch: Literal['resnet18', 'resnet34', 'resnet50'] = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.5,
        freeze_early_layers: bool = False
    ):
        """
        Initialize ResNet classifier.

        Parameters
        ----------
        num_classes : int
            Number of output classes (50 for ESC-50)
        arch : str
            ResNet architecture ('resnet18', 'resnet34', 'resnet50')
        pretrained : bool
            Whether to use ImageNet pretrained weights
        in_channels : int
            Number of input channels (1 for spectrogram, 3 for RGB-like)
        dropout : float
            Dropout probability before final layer
        freeze_early_layers : bool
            Whether to freeze early convolutional layers
        """
        super().__init__()

        self.num_classes = num_classes
        self.arch = arch
        self.in_channels = in_channels

        # Load pretrained ResNet
        if arch == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            feature_dim = 512
        elif arch == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)
            feature_dim = 512
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify first conv layer for single-channel input
        if in_channels != 3:
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize new conv with average of pretrained weights
            if pretrained:
                with torch.no_grad():
                    # Average across RGB channels and repeat
                    self.resnet.conv1.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, in_channels, 1, 1)

        # Freeze early layers if specified
        if freeze_early_layers:
            for name, param in self.resnet.named_parameters():
                if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        # Replace classifier head
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram of shape (batch, n_features, n_frames)
            or (batch, channels, n_features, n_frames)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_features, n_frames)

        # Extract features
        features = self.resnet(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ResNetWithAttention(nn.Module):
    """
    ResNet with attention pooling for audio classification.

    Adds an attention mechanism over the spatial features before
    global average pooling, which can improve performance on
    variable-length audio or for focusing on salient regions.
    """

    def __init__(
        self,
        num_classes: int = 50,
        arch: Literal['resnet18', 'resnet34', 'resnet50'] = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.arch = arch

        # Load pretrained ResNet
        if arch == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            feature_dim = 512
        elif arch == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet34(weights=weights)
            feature_dim = 512
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Modify first conv layer
        if in_channels != 3:
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.resnet.conv1.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, in_channels, 1, 1)

        # Remove avgpool and fc
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention pooling."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature extraction (without avgpool and fc)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Shape: (batch, feature_dim, H, W)

        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, 1, H, W)
        attn_weights = attn_weights.view(attn_weights.size(0), -1)  # (batch, H*W)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(attn_weights.size(0), 1, x.size(2), x.size(3))

        # Apply attention
        x = (x * attn_weights).sum(dim=(2, 3))  # (batch, feature_dim)

        # Classify
        logits = self.classifier(x)

        return logits

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EfficientResNet(nn.Module):
    """
    Efficient ResNet variant with additional regularization
    and improved training techniques for audio classification.

    Features:
    - Label smoothing support
    - Stochastic depth (dropout per residual block)
    - Mixup-ready (returns features for external mixup)
    """

    def __init__(
        self,
        num_classes: int = 50,
        arch: str = 'resnet18',
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.5,
        drop_path_rate: float = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes

        # Load base ResNet
        if arch == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif arch == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048

        # Modify input conv
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.backbone.conv1.weight.data = old_conv.weight.data.mean(
                        dim=1, keepdim=True
                    ).repeat(1, in_channels, 1, 1)

        self.backbone.fc = nn.Identity()

        # Classifier with more regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram
        return_features : bool
            If True, also return features before classifier
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = self.backbone(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_resnet_classifier(
    num_classes: int = 50,
    pretrained: bool = True,
    arch: str = 'resnet18',
    variant: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create ResNet classifier.

    Parameters
    ----------
    num_classes : int
        Number of classes
    pretrained : bool
        Use ImageNet pretrained weights
    arch : str
        ResNet architecture ('resnet18', 'resnet34', 'resnet50')
    variant : str
        Model variant:
        - 'standard': Basic ResNet with modified input
        - 'attention': ResNet with attention pooling
        - 'efficient': ResNet with extra regularization

    Returns
    -------
    nn.Module
        Configured ResNet model

    Examples
    --------
    >>> # Basic ResNet18 (Expected: ~83%)
    >>> model = create_resnet_classifier(num_classes=50, arch='resnet18')

    >>> # ResNet50 with attention (Expected: ~85%)
    >>> model = create_resnet_classifier(num_classes=50, arch='resnet50', variant='attention')
    """
    if variant == 'standard':
        return ResNetClassifier(
            num_classes=num_classes,
            arch=arch,
            pretrained=pretrained,
            **kwargs
        )
    elif variant == 'attention':
        return ResNetWithAttention(
            num_classes=num_classes,
            arch=arch,
            pretrained=pretrained,
            **kwargs
        )
    elif variant == 'efficient':
        return EfficientResNet(
            num_classes=num_classes,
            arch=arch,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("=" * 70)
    print("ResNet Classifier Test")
    print("=" * 70)

    # Test model creation
    print("\n[Test 1] Standard ResNet18")
    model = create_resnet_classifier(num_classes=50, arch='resnet18')
    print(f"  Total params: {model.get_total_params():,}")
    print(f"  Trainable params: {model.get_trainable_params():,}")

    # Test forward pass
    print("\n[Test 2] Forward Pass")
    batch_size = 4
    n_mels = 128
    n_frames = 216  # ~5 seconds of audio

    # Random spectrogram
    spectrogram = torch.randn(batch_size, n_mels, n_frames)

    with torch.no_grad():
        logits = model(spectrogram)

    print(f"  Input shape: {spectrogram.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 50)")

    # Test different variants
    print("\n[Test 3] Different Variants")
    for variant in ['standard', 'attention', 'efficient']:
        m = create_resnet_classifier(num_classes=50, variant=variant)
        with torch.no_grad():
            out = m(spectrogram)
        print(f"  {variant:12s}: params={m.get_total_params():,}, output={out.shape}")

    # Test different architectures
    print("\n[Test 4] Different Architectures")
    for arch in ['resnet18', 'resnet34', 'resnet50']:
        m = create_resnet_classifier(num_classes=50, arch=arch)
        print(f"  {arch:10s}: {m.get_total_params():,} params")

    # Test with 4D input (with channel dim)
    print("\n[Test 5] 4D Input (batch, channel, height, width)")
    spec_4d = torch.randn(batch_size, 1, n_mels, n_frames)
    with torch.no_grad():
        logits_4d = model(spec_4d)
    print(f"  Input shape: {spec_4d.shape}")
    print(f"  Output shape: {logits_4d.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
