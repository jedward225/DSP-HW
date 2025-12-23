"""
ESC-50 Dataset loader for audio retrieval experiments.

ESC-50 is a dataset of 2000 environmental audio recordings (5 seconds each)
organized into 50 semantic classes, with 40 clips per class.
The dataset is pre-arranged into 5 folds for cross-validation.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dsp_core import load_audio


@dataclass
class AudioSample:
    """Container for an audio sample."""
    filename: str
    fold: int
    target: int
    category: str
    waveform: Optional[np.ndarray] = None
    sr: int = 44100


class ESC50Dataset(Dataset):
    """
    ESC-50 Dataset class for audio retrieval.

    The dataset supports 5-fold cross-validation where each fold
    can be used as query set while the remaining folds form the gallery.

    Args:
        root_dir: Root directory containing ESC-50 dataset
        sr: Target sampling rate (default: 44100)
        preload: If True, preload all audio into memory
        transform: Optional transform to apply to waveforms
    """

    NUM_CLASSES = 50
    NUM_FOLDS = 5
    SAMPLES_PER_CLASS = 40
    TOTAL_SAMPLES = 2000
    DEFAULT_SR = 44100
    DURATION = 5.0  # seconds

    def __init__(
        self,
        root_dir: str,
        sr: int = 44100,
        preload: bool = False,
        transform=None
    ):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / 'audio'
        self.meta_path = self.root_dir / 'meta' / 'esc50.csv'
        self.sr = sr
        self.preload = preload
        self.transform = transform

        # Load metadata
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.metadata = pd.read_csv(self.meta_path)
        self._validate_dataset()

        # Build indices
        self._build_indices()

        # Preload audio if requested
        self._audio_cache: Dict[str, np.ndarray] = {}
        if preload:
            self._preload_audio()

    def _validate_dataset(self):
        """Validate that the dataset structure is correct."""
        if len(self.metadata) != self.TOTAL_SAMPLES:
            raise ValueError(
                f"Expected {self.TOTAL_SAMPLES} samples, found {len(self.metadata)}"
            )

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

    def _build_indices(self):
        """Build helper indices for efficient access."""
        # Index by fold
        self.fold_indices: Dict[int, List[int]] = {
            fold: self.metadata[self.metadata['fold'] == fold].index.tolist()
            for fold in range(1, self.NUM_FOLDS + 1)
        }

        # Index by category
        self.category_indices: Dict[int, List[int]] = {
            cat: self.metadata[self.metadata['target'] == cat].index.tolist()
            for cat in range(self.NUM_CLASSES)
        }

        # Category names
        self.categories = self.metadata.groupby('target')['category'].first().to_dict()

    def _preload_audio(self):
        """Preload all audio files into memory."""
        for idx in range(len(self.metadata)):
            filename = self.metadata.iloc[idx]['filename']
            if filename not in self._audio_cache:
                audio_path = self.audio_dir / filename
                waveform, _ = load_audio(str(audio_path), sr=self.sr)
                self._audio_cache[filename] = waveform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                - waveform: torch.Tensor of shape (num_samples,)
                - sr: sample rate
                - target: class index
                - category: class name
                - fold: fold number
                - filename: audio filename
        """
        row = self.metadata.iloc[idx]
        filename = row['filename']

        # Load or retrieve cached audio
        if filename in self._audio_cache:
            waveform = self._audio_cache[filename]
        else:
            audio_path = self.audio_dir / filename
            waveform, _ = load_audio(str(audio_path), sr=self.sr)
            if self.preload:
                self._audio_cache[filename] = waveform

        # Apply transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Convert to torch tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        return {
            'waveform': waveform,
            'sr': self.sr,
            'target': row['target'],
            'category': row['category'],
            'fold': row['fold'],
            'filename': filename,
            'idx': idx
        }

    def get_fold_data(self, fold: int) -> List[Dict]:
        """
        Get all samples from a specific fold.

        Args:
            fold: Fold number (1-5)

        Returns:
            List of sample dictionaries
        """
        if fold not in range(1, self.NUM_FOLDS + 1):
            raise ValueError(f"Fold must be 1-5, got {fold}")

        indices = self.fold_indices[fold]
        return [self[idx] for idx in indices]

    def get_query_gallery_split(
        self,
        query_fold: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into query and gallery sets for retrieval.

        Args:
            query_fold: Fold to use as query set (1-5)

        Returns:
            Tuple of (query_samples, gallery_samples)
        """
        if query_fold not in range(1, self.NUM_FOLDS + 1):
            raise ValueError(f"query_fold must be 1-5, got {query_fold}")

        gallery_folds = [f for f in range(1, self.NUM_FOLDS + 1) if f != query_fold]

        query_samples = self.get_fold_data(query_fold)
        gallery_samples = []
        for fold in gallery_folds:
            gallery_samples.extend(self.get_fold_data(fold))

        return query_samples, gallery_samples

    def get_query_gallery_indices(
        self,
        query_fold: int
    ) -> Tuple[List[int], List[int]]:
        """
        Get indices for query and gallery sets.

        Args:
            query_fold: Fold to use as query set (1-5)

        Returns:
            Tuple of (query_indices, gallery_indices)
        """
        if query_fold not in range(1, self.NUM_FOLDS + 1):
            raise ValueError(f"query_fold must be 1-5, got {query_fold}")

        query_indices = self.fold_indices[query_fold]
        gallery_indices = []
        for fold in range(1, self.NUM_FOLDS + 1):
            if fold != query_fold:
                gallery_indices.extend(self.fold_indices[fold])

        return query_indices, gallery_indices

    def get_category_name(self, target: int) -> str:
        """Get the category name for a target index."""
        return self.categories.get(target, f"unknown_{target}")

    def get_category_samples(self, target: int) -> List[Dict]:
        """Get all samples from a specific category."""
        indices = self.category_indices.get(target, [])
        return [self[idx] for idx in indices]

    def get_sample_by_filename(self, filename: str) -> Optional[Dict]:
        """Get a sample by its filename."""
        matches = self.metadata[self.metadata['filename'] == filename]
        if len(matches) == 0:
            return None
        idx = matches.index[0]
        return self[idx]

    @property
    def class_names(self) -> List[str]:
        """Get list of all class names in order."""
        return [self.categories[i] for i in range(self.NUM_CLASSES)]

    def __repr__(self) -> str:
        return (
            f"ESC50Dataset(\n"
            f"  root_dir={self.root_dir},\n"
            f"  num_samples={len(self)},\n"
            f"  num_classes={self.NUM_CLASSES},\n"
            f"  num_folds={self.NUM_FOLDS},\n"
            f"  sr={self.sr},\n"
            f"  preloaded={len(self._audio_cache) > 0}\n"
            f")"
        )


def create_fold_splits(
    dataset: ESC50Dataset,
    num_folds: int = 5
) -> List[Tuple[List[int], List[int]]]:
    """
    Create all fold splits for cross-validation.

    Args:
        dataset: ESC50Dataset instance
        num_folds: Number of folds (default: 5)

    Returns:
        List of (query_indices, gallery_indices) tuples
    """
    splits = []
    for fold in range(1, num_folds + 1):
        query_indices, gallery_indices = dataset.get_query_gallery_indices(fold)
        splits.append((query_indices, gallery_indices))
    return splits
