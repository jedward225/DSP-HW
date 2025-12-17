"""
Training script for ESC-50 sound classification.

Supports multiple models:
- BEATs (Path A): Pre-trained on AudioSet, fine-tune with Adapter
- CLAP (Path A): Zero-shot or fine-tune Audio Encoder
- ResNet18 (Path B): ImageNet pre-trained, fine-tune on spectrograms

Usage:
    # BEATs + Adapter (Expected: ~92%)
    python -m src.classification.train --model beats --mode adapter

    # CLAP Zero-shot (Expected: ~92%)
    python -m src.classification.train --model clap --mode zeroshot

    # CLAP Fine-tune (Expected: ~98%)
    python -m src.classification.train --model clap --mode finetune

    # ResNet18 (Expected: ~83%)
    python -m src.classification.train --model resnet18 --feature stft --n_fft 2048 --hop 512
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset import ESC50Dataset, create_dataloaders


class WaveformDataset(Dataset):
    """Dataset wrapper for waveform input (BEATs/CLAP)."""

    def __init__(
        self,
        root: str = 'ESC-50',
        folds: list = [1, 2, 3, 4],
        sr: int = 16000,  # BEATs uses 16kHz, CLAP uses 48kHz
        duration: float = 5.0,
        augment: bool = False
    ):
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)

        # Use ESC50Dataset for metadata and loading
        self.base_dataset = ESC50Dataset(
            root=root,
            folds=folds,
            feature_type='waveform',
            sr=sr,
            duration=duration,
            augment=augment
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        waveform, label, filename = self.base_dataset[idx]

        # Pad or truncate to target length
        if waveform.size(0) < self.target_length:
            padding = self.target_length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.size(0) > self.target_length:
            waveform = waveform[:self.target_length]

        return waveform, label


class SpectrogramDataset(Dataset):
    """Dataset wrapper for spectrogram input (ResNet18)."""

    def __init__(
        self,
        root: str = 'ESC-50',
        folds: list = [1, 2, 3, 4],
        feature_type: str = 'mel',
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        target_frames: int = 216,  # ~5 seconds at default settings
        augment: bool = False,
        augment_config: Optional[Dict] = None
    ):
        self.feature_type = feature_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_frames = target_frames
        self.augment = augment
        self.augment_config = augment_config or {}

        # Load base dataset for waveforms
        self.base_dataset = ESC50Dataset(
            root=root,
            folds=folds,
            feature_type='waveform',
            sr=sr,
            augment=False  # We'll do augmentation on spectrograms
        )

        # Import feature extraction
        from src.classification.features import extract_features, pad_or_truncate, normalize_features
        self.extract_features = extract_features
        self.pad_or_truncate = pad_or_truncate
        self.normalize_features = normalize_features

        # Import augmentation if needed
        if augment:
            from src.classification.augment import AudioAugmenter
            self.augmenter = AudioAugmenter(**self.augment_config)
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        waveform, label, filename = self.base_dataset[idx]

        # Extract features
        features = self.extract_features(
            waveform.numpy(),
            sr=self.sr,
            feature_type=self.feature_type,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc
        )

        # Apply augmentation
        if self.augmenter is not None:
            features = self.augmenter(features)

        # Pad/truncate to fixed length
        features = self.pad_or_truncate(features, self.target_frames)

        # Normalize
        features = self.normalize_features(features, method='standard')

        return torch.from_numpy(features).float(), label


def create_model(
    model_type: str,
    mode: str = 'adapter',
    num_classes: int = 50,
    **kwargs
) -> nn.Module:
    """Create model based on type and mode."""

    if model_type == 'beats':
        from src.classification.models.beats import create_beats_classifier
        model = create_beats_classifier(
            num_classes=num_classes,
            mode=mode,
            **kwargs
        )

    elif model_type == 'clap':
        from src.classification.models.clap import create_clap_classifier
        model = create_clap_classifier(
            num_classes=num_classes,
            mode=mode,
            **kwargs
        )

    elif model_type == 'resnet18':
        from src.classification.models.resnet import create_resnet_classifier
        model = create_resnet_classifier(
            num_classes=num_classes,
            pretrained=True,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        pbar.set_postfix({
            'loss': total_loss / (len(all_preds) // targets.size(0)),
            'acc': 100. * correct / total
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def train(
    model_type: str = 'beats',
    mode: str = 'adapter',
    feature_type: str = 'mel',
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    data_root: str = 'ESC-50',
    output_dir: str = 'checkpoints',
    use_augment: bool = False,
    num_workers: int = 0,
    **kwargs
):
    """Main training function."""

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_type}_{mode}_{timestamp}"

    # Create dataloaders based on model type
    if model_type in ['beats', 'clap']:
        # Waveform input
        sr = 16000 if model_type == 'beats' else 48000
        train_dataset = WaveformDataset(
            root=data_root,
            folds=[1, 2, 3, 4],
            sr=sr,
            augment=use_augment
        )
        test_dataset = WaveformDataset(
            root=data_root,
            folds=[5],
            sr=sr,
            augment=False
        )

    else:
        # Spectrogram input (ResNet18)
        augment_config = {
            'use_time_mask': True,
            'use_freq_mask': True,
            'time_mask_param': 40,
            'freq_mask_param': 27,
            'num_time_masks': 2,
            'num_freq_masks': 2
        } if use_augment else {}

        train_dataset = SpectrogramDataset(
            root=data_root,
            folds=[1, 2, 3, 4],
            feature_type=feature_type,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            augment=use_augment,
            augment_config=augment_config
        )
        test_dataset = SpectrogramDataset(
            root=data_root,
            folds=[5],
            feature_type=feature_type,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            augment=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    model = create_model(model_type, mode=mode, num_classes=50, **kwargs)
    model = model.to(device)

    print(f"Model: {model_type} ({mode})")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer (with label smoothing to reduce overfitting)
    label_smoothing = kwargs.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Different optimizers for different models
    if model_type == 'clap' and mode == 'zeroshot':
        # Zero-shot doesn't need training
        print("\nZero-shot mode: evaluating without training...")
        test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)
        print(f"Zero-shot Accuracy: {test_acc:.2f}%")
        return

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate
        test_loss, test_acc, preds, targets = evaluate(
            model, test_loader, criterion, device, epoch
        )

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': {
                    'model_type': model_type,
                    'mode': mode,
                    'feature_type': feature_type,
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'n_mels': n_mels
                }
            }, output_dir / f"{exp_name}_best.pt")
            print(f"  [Saved] New best accuracy: {best_acc:.2f}%")

    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_acc:.2f}%")

    # Save training history
    with open(output_dir / f"{exp_name}_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train ESC-50 classifier')

    # Model settings
    parser.add_argument('--model', type=str, default='beats',
                        choices=['beats', 'clap', 'resnet18'],
                        help='Model type')
    parser.add_argument('--mode', type=str, default='adapter',
                        help='Training mode (adapter/finetune/zeroshot)')

    # Feature settings (for ResNet18)
    parser.add_argument('--feature', type=str, default='mel',
                        choices=['stft', 'mel', 'mfcc', 'mfcc_delta'],
                        help='Feature type for ResNet18')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size')
    parser.add_argument('--hop', type=int, default=512,
                        help='Hop length')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of Mel bands')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')

    # Other settings
    parser.add_argument('--data_root', type=str, default='ESC-50',
                        help='ESC-50 dataset root')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Run training
    train(
        model_type=args.model,
        mode=args.mode,
        feature_type=args.feature,
        n_fft=args.n_fft,
        hop_length=args.hop,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_augment=args.augment,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
