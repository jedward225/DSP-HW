"""
ESC-50 Dataset Loader for PyTorch

This module provides a PyTorch Dataset class for the ESC-50 environmental sound classification dataset.

ESC-50 Structure:
    - 2000 audio clips (5 seconds each)
    - 50 classes (40 samples per class)
    - 5 folds for cross-validation
    - Sample rate: 44100 Hz

Usage:
    >>> from src.utils.dataset import ESC50Dataset
    >>> dataset = ESC50Dataset(root='ESC-50', folds=[1, 2, 3, 4], feature_type='waveform')
    >>> waveform, label, filename = dataset[0]
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union, Tuple, Literal
import librosa


class ESC50Dataset(Dataset):
    """
    PyTorch Dataset for ESC-50 environmental sound classification.

    Parameters
    ----------
    root : str
        Root directory of ESC-50 dataset (contains 'audio' and 'meta' folders)
    folds : List[int]
        List of folds to include (1-5). For training: [1,2,3,4], for testing: [5]
    feature_type : {'waveform', 'mfcc', 'mel_spectrogram'}
        Type of features to extract:
        - 'waveform': Raw audio signal (for neural networks)
        - 'mfcc': MFCC features using our dsp_core implementation
        - 'mel_spectrogram': Log-Mel spectrogram (for CNN/Transformer)
    sr : int, optional
        Target sample rate. If None, uses original 44100 Hz
    duration : float, optional
        Audio duration in seconds. If None, uses full clip (5 seconds)
    n_mfcc : int
        Number of MFCC coefficients (default: 13)
    n_fft : int
        FFT window size (default: 2048)
    hop_length : int, optional
        Number of samples between frames. If None, n_fft // 4
    n_mels : int
        Number of Mel bands (default: 128)
    augment : bool
        Whether to apply data augmentation (time shift, pitch shift, noise)

    Returns
    -------
    Tuple[torch.Tensor, int, str]
        - features: Audio features (shape depends on feature_type)
        - label: Integer class label (0-49)
        - filename: Audio filename

    Examples
    --------
    >>> # For MFCC-based retrieval (Task 1)
    >>> train_set = ESC50Dataset(root='ESC-50', folds=[1,2,3,4], feature_type='mfcc')
    >>> test_set = ESC50Dataset(root='ESC-50', folds=[5], feature_type='mfcc')

    >>> # For neural network training (Task 2)
    >>> train_set = ESC50Dataset(root='ESC-50', folds=[1,2,3,4],
    ...                           feature_type='mel_spectrogram', augment=True)
    """

    def __init__(
        self,
        root: str = 'ESC-50',
        folds: List[int] = [1, 2, 3, 4],
        feature_type: Literal['waveform', 'mfcc', 'mel_spectrogram'] = 'waveform',
        sr: Optional[int] = None,
        duration: Optional[float] = None,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        augment: bool = False
    ):
        self.root = root
        self.folds = folds
        self.feature_type = feature_type
        self.sr = sr if sr is not None else 44100  # Default ESC-50 sample rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.n_mels = n_mels
        self.augment = augment

        # Load metadata
        meta_path = os.path.join(root, 'meta', 'esc50.csv')
        self.metadata = pd.read_csv(meta_path)

        # Filter by folds
        self.metadata = self.metadata[self.metadata['fold'].isin(folds)].reset_index(drop=True)

        # Get class names
        self.class_names = sorted(self.metadata['category'].unique())
        self.num_classes = len(self.class_names)

        print(f"[ESC50Dataset] Loaded {len(self.metadata)} samples from folds {folds}")
        print(f"[ESC50Dataset] Feature type: {feature_type}")
        print(f"[ESC50Dataset] Sample rate: {self.sr} Hz")
        print(f"[ESC50Dataset] Number of classes: {self.num_classes}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load and process audio sample.

        Returns
        -------
        features : torch.Tensor
            Shape depends on feature_type:
            - waveform: (n_samples,) e.g., (220500,) for 5s @ 44100Hz
            - mfcc: (n_mfcc, n_frames) e.g., (13, 44)
            - mel_spectrogram: (n_mels, n_frames) e.g., (128, 44)
        label : int
            Class label (0-49)
        filename : str
            Audio filename
        """
        # Get metadata
        row = self.metadata.iloc[idx]
        filename = row['filename']
        label = row['target']

        # Load audio
        audio_path = os.path.join(self.root, 'audio', filename)
        y, sr_orig = librosa.load(audio_path, sr=self.sr, duration=self.duration)

        # Data augmentation (only for training)
        if self.augment:
            y = self._augment(y)

        # Extract features
        if self.feature_type == 'waveform':
            features = torch.from_numpy(y).float()

        elif self.feature_type == 'mfcc':
            # Use our hand-written MFCC implementation
            from src.dsp_core import mfcc
            mfcc_features = mfcc(
                y,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            features = torch.from_numpy(mfcc_features).float()

        elif self.feature_type == 'mel_spectrogram':
            # Compute log-Mel spectrogram for CNN/Transformer
            from src.dsp_core import stft
            from src.dsp_core.mfcc import mel_filterbank
            from src.dsp_core.stft import power_to_db

            # STFT
            S = stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

            # Power spectrum
            power_spec = np.abs(S) ** 2

            # Mel filterbank
            mel_basis = mel_filterbank(
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels
            )

            # Apply Mel filters
            mel_spec = np.dot(mel_basis, power_spec)

            # Convert to dB
            log_mel_spec = power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

            features = torch.from_numpy(log_mel_spec).float()

        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        return features, label, filename

    def _augment(self, y: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to audio signal.

        Augmentation techniques:
        1. Time shift: Randomly shift audio in time
        2. Pitch shift: Randomly shift pitch by [-2, 2] semitones
        3. Add noise: Add Gaussian noise with SNR [20, 40] dB

        Parameters
        ----------
        y : np.ndarray
            Input audio signal

        Returns
        -------
        np.ndarray
            Augmented audio signal
        """
        # 1. Time shift (with 50% probability)
        if np.random.rand() < 0.5:
            shift_amount = np.random.randint(-self.sr // 2, self.sr // 2)
            y = np.roll(y, shift_amount)

        # 2. Pitch shift (with 30% probability)
        if np.random.rand() < 0.3:
            n_steps = np.random.uniform(-2, 2)  # Semitones
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        # 3. Add noise (with 30% probability)
        if np.random.rand() < 0.3:
            # Random SNR between 20-40 dB
            snr_db = np.random.uniform(20, 40)
            signal_power = np.mean(y ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(y))
            y = y + noise

        return y

    def get_class_name(self, label: int) -> str:
        """Get class name from integer label."""
        return self.metadata[self.metadata['target'] == label]['category'].iloc[0]

    def get_fold_distribution(self) -> pd.DataFrame:
        """Get distribution of samples across folds."""
        return self.metadata.groupby('fold')['target'].count().reset_index(
            name='count'
        )

    def get_class_distribution(self) -> pd.DataFrame:
        """Get distribution of samples across classes."""
        return self.metadata.groupby('category')['target'].count().reset_index(
            name='count'
        ).sort_values('count', ascending=False)


def create_dataloaders(
    root: str = 'ESC-50',
    train_folds: List[int] = [1, 2, 3, 4],
    test_folds: List[int] = [5],
    feature_type: str = 'waveform',
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = False,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    Parameters
    ----------
    root : str
        Root directory of ESC-50 dataset
    train_folds : List[int]
        Folds to use for training (default: [1,2,3,4])
    test_folds : List[int]
        Folds to use for testing (default: [5])
    feature_type : str
        Type of features to extract
    batch_size : int
        Batch size for DataLoader
    num_workers : int
        Number of parallel workers for data loading
    augment_train : bool
        Whether to apply augmentation to training set
    **dataset_kwargs
        Additional arguments for ESC50Dataset

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Testing data loader

    Examples
    --------
    >>> train_loader, test_loader = create_dataloaders(
    ...     feature_type='mel_spectrogram',
    ...     batch_size=32,
    ...     augment_train=True
    ... )
    """
    # Create datasets
    train_dataset = ESC50Dataset(
        root=root,
        folds=train_folds,
        feature_type=feature_type,
        augment=augment_train,
        **dataset_kwargs
    )

    test_dataset = ESC50Dataset(
        root=root,
        folds=test_folds,
        feature_type=feature_type,
        augment=False,  # Never augment test set
        **dataset_kwargs
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("=" * 70)
    print("ESC-50 Dataset Test")
    print("=" * 70)

    # Test 1: Waveform dataset
    print("\n[Test 1] Waveform dataset")
    dataset = ESC50Dataset(root='ESC-50', folds=[1], feature_type='waveform', sr=22050)
    waveform, label, filename = dataset[0]
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Label: {label} ({dataset.get_class_name(label)})")
    print(f"  Filename: {filename}")

    # Test 2: MFCC dataset
    print("\n[Test 2] MFCC dataset")
    dataset = ESC50Dataset(root='ESC-50', folds=[1], feature_type='mfcc', sr=22050)
    mfcc_feat, label, filename = dataset[0]
    print(f"  MFCC shape: {mfcc_feat.shape}")
    print(f"  Label: {label} ({dataset.get_class_name(label)})")

    # Test 3: Mel spectrogram dataset
    print("\n[Test 3] Mel spectrogram dataset")
    dataset = ESC50Dataset(root='ESC-50', folds=[1], feature_type='mel_spectrogram', sr=22050)
    mel_spec, label, filename = dataset[0]
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    print(f"  Label: {label}")

    # Test 4: DataLoader
    print("\n[Test 4] DataLoader")
    train_loader, test_loader = create_dataloaders(
        root='ESC-50',
        feature_type='mfcc',
        batch_size=8,
        num_workers=0,
        sr=22050
    )

    batch_features, batch_labels, batch_filenames = next(iter(train_loader))
    print(f"  Batch features shape: {batch_features.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Number of train batches: {len(train_loader)}")
    print(f"  Number of test batches: {len(test_loader)}")

    # Test 5: Class distribution
    print("\n[Test 5] Class distribution")
    dataset = ESC50Dataset(root='ESC-50', folds=[1, 2, 3, 4, 5], feature_type='waveform')
    class_dist = dataset.get_class_distribution()
    print(f"  Total classes: {len(class_dist)}")
    print(f"  Samples per class: {class_dist['count'].values[0]} (should be 40)")

    print("\n" + "=" * 70)
    print("All tests passed! ")
    print("=" * 70)
