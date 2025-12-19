"""
Train supervised contrastive encoder on ESC-50 dataset.

Usage:
    python train_contrastive.py --data_dir /path/to/ESC-50 --output_dir models/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.contrastive import ContrastiveEncoder, SupConLoss
from src.dsp_core import log_melspectrogram


class MelSpectrogramDataset(Dataset):
    """Dataset that loads audio and returns mel spectrograms with labels."""

    def __init__(
        self,
        data_dir: str,
        fold: int = None,
        exclude_fold: int = None,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_frames: int = 216,
    ):
        import pandas as pd

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
    """Train for one epoch with SupCon loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_mel, batch_labels in dataloader:
        batch_mel = batch_mel.to(device, non_blocking=True)
        batch_labels = torch.tensor(batch_labels).to(device, non_blocking=True)

        # Forward pass - get L2-normalized projections
        projections = model(batch_mel)

        # Compute SupCon loss
        loss = criterion(projections, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate_retrieval(model, train_loader, val_loader, device):
    """
    Validate using retrieval accuracy.

    For each val sample, find nearest neighbor in train set
    and check if labels match.
    """
    model.eval()

    # Extract train embeddings
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for batch_mel, batch_labels in train_loader:
            batch_mel = batch_mel.to(device, non_blocking=True)
            emb = model.get_embedding(batch_mel)
            train_embeddings.append(emb)
            train_labels.extend(batch_labels)

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.tensor(train_labels).to(device)

    # Evaluate on val set
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_mel, batch_labels in val_loader:
            batch_mel = batch_mel.to(device, non_blocking=True)
            batch_labels = torch.tensor(batch_labels).to(device)

            # Get embeddings
            val_emb = model.get_embedding(batch_mel)

            # Compute cosine similarity
            similarity = torch.matmul(val_emb, train_embeddings.T)

            # Find nearest neighbor
            nn_idx = similarity.argmax(dim=1)
            nn_labels = train_labels[nn_idx]

            correct += (nn_labels == batch_labels).sum().item()
            total += len(batch_labels)

    return correct / total


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
        drop_last=True,  # Important for contrastive learning
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
    model = ContrastiveEncoder(
        n_mels=128,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
    ).to(device)

    # Setup training
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_retrieval_acc = 0
    retrieval_acc = 0.0  # Initialize to avoid undefined error if epochs < 10
    history = {'train_loss': [], 'retrieval_acc': []}

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate retrieval accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            retrieval_acc = validate_retrieval(model, train_loader, val_loader, device)
            history['retrieval_acc'].append(retrieval_acc)

            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f}, Retrieval Acc: {retrieval_acc:.4f}")

            # Save best model for this fold
            if retrieval_acc > best_retrieval_acc:
                best_retrieval_acc = retrieval_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'retrieval_acc': retrieval_acc,
                    'fold': fold,
                    'config': {
                        'n_mels': 128,
                        'n_frames': train_dataset.n_frames,
                        'n_fft': train_dataset.n_fft,
                        'hop_length': train_dataset.hop_length,
                        'embed_dim': args.embed_dim,
                        'proj_dim': args.proj_dim,
                    },
                }
                torch.save(checkpoint, output_dir / f'contrastive_esc50_fold{fold}.pt')
                print(f"  -> Saved best model for fold {fold} (retrieval_acc: {retrieval_acc:.4f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")

        history['train_loss'].append(train_loss)
        scheduler.step()

    # Compute final retrieval accuracy if not computed on last epoch
    if args.epochs % 10 != 0:
        retrieval_acc = validate_retrieval(model, train_loader, val_loader, device)
        print(f"Final retrieval accuracy: {retrieval_acc:.4f}")

    print(f"Fold {fold} - Best Retrieval Acc: {best_retrieval_acc:.4f}")
    return best_retrieval_acc, history


def main():
    parser = argparse.ArgumentParser(description='Train supervised contrastive encoder with 5-fold CV')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ESC-50 dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (larger is better for contrastive)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for SupCon loss')
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
        best_acc, history = train_fold(args, fold, device)
        all_results[f'fold{fold}'] = best_acc
        all_histories[f'fold{fold}'] = history

    # Summary
    print("\n" + "=" * 50)
    print("5-Fold Cross-Validation Results")
    print("=" * 50)
    accs = list(all_results.values())
    for fold, acc in all_results.items():
        print(f"{fold}: {acc:.4f}")
    print(f"Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Save results
    with open(output_dir / 'contrastive_5fold_results.json', 'w') as f:
        json.dump({
            'results': all_results,
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
