"""
BEATs model wrapper for ESC-50 classification.

BEATs: Audio Pre-Training with Acoustic Tokenizers
Reference: https://arxiv.org/abs/2212.09058
Official repo: https://github.com/microsoft/unilm/tree/master/beats

Expected accuracy: ~89.25% (fine-tune) to ~92.15% (with Adapter) on ESC-50
Reference: [Ref-1] 张鑫恺等 89.25%, [Ref-2] 苏慧学等 92.15% (with Adapter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os


class Adapter(nn.Module):
    """
    Adapter module for efficient fine-tuning.

    Adds a small bottleneck layer that can be trained while keeping
    the main model frozen. This improves performance while reducing
    training cost.

    Reference: [Ref-2] BEATs + Adapter achieved 92.15% on ESC-50
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class BEATsClassifier(nn.Module):
    """
    BEATs-based classifier for ESC-50.

    Architecture:
    - BEATs encoder (frozen or fine-tuned)
    - Optional Adapter layers
    - Classification head (MLP)

    Usage:
    >>> model = BEATsClassifier(num_classes=50, use_adapter=True)
    >>> logits = model(waveform)  # waveform: (batch, samples)
    """

    def __init__(
        self,
        num_classes: int = 50,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        use_adapter: bool = True,
        adapter_bottleneck: int = 64,
        classifier_hidden: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize BEATs classifier.

        Parameters
        ----------
        num_classes : int
            Number of output classes (50 for ESC-50)
        checkpoint_path : str, optional
            Path to BEATs checkpoint (.pt file)
        freeze_encoder : bool
            Whether to freeze BEATs encoder weights
        use_adapter : bool
            Whether to use Adapter layers (recommended)
        adapter_bottleneck : int
            Bottleneck dimension for Adapter
        classifier_hidden : int
            Hidden dimension for classification head
        dropout : float
            Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.use_adapter = use_adapter

        # Load BEATs model
        self.beats = self._load_beats(checkpoint_path)

        # Get embedding dimension from BEATs config
        self.embed_dim = self.beats.cfg.encoder_embed_dim if hasattr(self.beats, 'cfg') else 768

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.beats.parameters():
                param.requires_grad = False

        # Add Adapter if specified
        if use_adapter:
            self.adapter = Adapter(
                input_dim=self.embed_dim,
                bottleneck_dim=adapter_bottleneck,
                dropout=dropout
            )
        else:
            self.adapter = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def _load_beats(self, checkpoint_path: Optional[str] = None):
        """Load BEATs model from checkpoint."""
        try:
            # Try to import BEATs
            from BEATs import BEATs, BEATsConfig

            if checkpoint_path is None:
                # Default checkpoint path
                checkpoint_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../../checkpoints/BEATs_iter3_plus_AS2M.pt'
                )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"BEATs checkpoint not found at {checkpoint_path}. "
                    "Please download from: "
                    "https://github.com/microsoft/unilm/tree/master/beats"
                )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            model = BEATs(cfg)
            model.load_state_dict(checkpoint['model'])

            return model

        except ImportError:
            print("BEATs not installed. Using placeholder model.")
            print("To install: git clone https://github.com/microsoft/unilm.git")
            print("            cd unilm/beats && pip install -e .")
            return self._create_placeholder_model()

    def _create_placeholder_model(self):
        """Create a placeholder model when BEATs is not available."""
        class PlaceholderBEATs(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type('Config', (), {'encoder_embed_dim': 768})()
                # Simple conv encoder as placeholder
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=10, stride=5),
                    nn.GELU(),
                    nn.Conv1d(64, 128, kernel_size=8, stride=4),
                    nn.GELU(),
                    nn.Conv1d(128, 256, kernel_size=4, stride=2),
                    nn.GELU(),
                    nn.Conv1d(256, 512, kernel_size=4, stride=2),
                    nn.GELU(),
                    nn.Conv1d(512, 768, kernel_size=4, stride=2),
                    nn.GELU(),
                )

            def extract_features(self, x, padding_mask=None):
                # x: (batch, samples)
                x = x.unsqueeze(1)  # (batch, 1, samples)
                x = self.encoder(x)  # (batch, 768, time)
                x = x.transpose(1, 2)  # (batch, time, 768)
                return x

        return PlaceholderBEATs()

    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        waveform : torch.Tensor
            Input waveform of shape (batch, samples)
        padding_mask : torch.Tensor, optional
            Padding mask of shape (batch, samples)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        # Extract features from BEATs
        # Output shape: (batch, time, embed_dim)
        features = self.beats.extract_features(waveform, padding_mask=padding_mask)

        # Handle tuple output (some BEATs versions)
        if isinstance(features, tuple):
            features = features[0]

        # Apply Adapter if enabled
        if self.adapter is not None:
            features = self.adapter(features)

        # Global average pooling over time
        # (batch, time, embed_dim) -> (batch, embed_dim)
        features = features.mean(dim=1)

        # Classification
        logits = self.classifier(features)

        return logits

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_beats_classifier(
    num_classes: int = 50,
    checkpoint_path: Optional[str] = None,
    mode: str = 'adapter'
) -> BEATsClassifier:
    """
    Factory function to create BEATs classifier.

    Parameters
    ----------
    num_classes : int
        Number of classes
    checkpoint_path : str, optional
        Path to BEATs checkpoint
    mode : str
        Training mode:
        - 'adapter': Freeze encoder, train Adapter + classifier (recommended)
        - 'finetune': Train entire model
        - 'linear': Freeze encoder, only train classifier

    Returns
    -------
    BEATsClassifier
        Configured model
    """
    if mode == 'adapter':
        return BEATsClassifier(
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            freeze_encoder=True,
            use_adapter=True
        )
    elif mode == 'finetune':
        return BEATsClassifier(
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            freeze_encoder=False,
            use_adapter=False
        )
    elif mode == 'linear':
        return BEATsClassifier(
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            freeze_encoder=True,
            use_adapter=False
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    print("=" * 70)
    print("BEATs Classifier Test")
    print("=" * 70)

    # Test model creation
    print("\n[Test 1] Model Creation (Adapter mode)")
    model = create_beats_classifier(num_classes=50, mode='adapter')
    print(f"  Total params: {model.get_total_params():,}")
    print(f"  Trainable params: {model.get_trainable_params():,}")

    # Test forward pass
    print("\n[Test 2] Forward Pass")
    batch_size = 4
    num_samples = 16000 * 5  # 5 seconds at 16kHz

    # Random waveform
    waveform = torch.randn(batch_size, num_samples)

    # Forward pass
    with torch.no_grad():
        logits = model(waveform)

    print(f"  Input shape: {waveform.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 50)")

    # Test different modes
    print("\n[Test 3] Different Training Modes")
    for mode in ['adapter', 'finetune', 'linear']:
        m = create_beats_classifier(num_classes=50, mode=mode)
        print(f"  {mode:10s}: trainable={m.get_trainable_params():,} / total={m.get_total_params():,}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
