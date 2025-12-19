"""
Train mel spectrogram autoencoder on ESC-50 dataset.

Usage:
    python train_autoencoder.py --data_dir /path/to/ESC-50 --output_dir models/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.autoencoder import MelAutoencoder
from src.dsp_core import log_melspectrogram


class MelSpectrogramDataset(Dataset):
    """Dataset that loads audio and returns mel spectrograms."""

    def __init__(
        self,
        data_dir: str,
        fold: int = None,
        exclude_fold: int = None,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_frames: int = 256,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to ESC-50 dataset
            fold: If specified, only use this fold
            exclude_fold: If specified, exclude this fold (for train/val split)
            sr: Sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
            n_frames: Number of frames to pad/truncate to
        """
        import pandas as pd
        import librosa

        self.data_dir = Path(data_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frames = n_frames

        # Load metadata
        meta_path = self.data_dir / 'meta' / 'esc50.csv'
        self.meta = pd.read_csv(meta_path)

        # Filter by fold
        if fold is not None:
            self.meta = self.meta[self.meta['fold'] == fold]
        elif exclude_fold is not None:
            self.meta = self.meta[self.meta['fold'] != exclude_fold]

        self.files = self.meta['filename'].tolist()
        self.labels = self.meta['target'].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import librosa

        # Load audio
        audio_path = self.data_dir / 'audio' / self.files[idx]
        waveform, _ = librosa.load(audio_path, sr=self.sr)

        # Compute mel spectrogram
        mel = log_melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Pad or truncate
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
        else:
            mel = np.zeros_like(mel)

        return torch.from_numpy(mel).float(), self.labels[idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_mel, _ in dataloader:
        batch_mel = batch_mel.to(device, non_blocking=True)

        # Forward pass
        reconstructed, _ = model(batch_mel)

        # Compute loss
        loss = criterion(reconstructed.squeeze(1), batch_mel)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_mel, _ in dataloader:
            batch_mel = batch_mel.to(device, non_blocking=True)
            reconstructed, _ = model(batch_mel)
            loss = criterion(reconstructed.squeeze(1), batch_mel)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_fold(args, fold, device):
    """Train model for a single fold."""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}")
    print(f"{'='*50}")

    output_dir = Path(args.output_dir)

    # Create datasets
    print("Loading datasets...")
    train_dataset = MelSpectrogramDataset(
        args.data_dir,
        exclude_fold=fold,
    )
    val_dataset = MelSpectrogramDataset(
        args.data_dir,
        fold=fold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = MelAutoencoder(
        n_mels=128,
        n_frames=256,
        latent_dim=args.latent_dim,
    ).to(device)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'fold': fold,
                'config': {
                    'n_mels': 128,
                    'n_frames': 256,
                    'latent_dim': args.latent_dim,
                },
            }
            torch.save(checkpoint, output_dir / f'autoencoder_esc50_fold{fold}.pt')

    print(f"Fold {fold} - Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss, history


def main():
    parser = argparse.ArgumentParser(description='Train mel autoencoder with 5-fold CV')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ESC-50 dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--folds', type=str, default='1,2,3,4,5',
                        help='Folds to train (comma-separated)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse folds
    folds = [int(f) for f in args.folds.split(',')]

    # Train each fold
    all_results = {}
    all_histories = {}

    for fold in folds:
        best_loss, history = train_fold(args, fold, device)
        all_results[f'fold{fold}'] = best_loss
        all_histories[f'fold{fold}'] = history

    # Summary
    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation Results")
    print("=" * 50)
    losses = list(all_results.values())
    for fold, loss in all_results.items():
        print(f"{fold}: {loss:.6f}")
    print(f"Mean: {np.mean(losses):.6f} ± {np.std(losses):.6f}")

    # Save results
    with open(output_dir / 'autoencoder_5fold_results.json', 'w') as f:
        json.dump({
            'results': all_results,
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
