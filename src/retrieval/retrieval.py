"""
Sound Retrieval System

This module implements the main retrieval logic for the ESC-50 sound retrieval task.
It integrates feature extraction, distance computation, and evaluation metrics.

Usage:
    # Command line
    python -m src.retrieval.retrieval --feature mfcc --distance cosine

    # Python API
    >>> from src.retrieval.retrieval import SoundRetrieval
    >>> retrieval = SoundRetrieval(feature_type='mfcc', distance_method='cosine')
    >>> results = retrieval.run(data_root='ESC-50')
"""

import os
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import time
import json

from .features import extract_features, FeatureExtractor, aggregate_features
from .distances import compute_similarity_matrix
from .metrics import evaluate_retrieval, compute_retrieval_labels


class SoundRetrieval:
    """
    Sound Retrieval System for ESC-50 dataset.

    This class provides an end-to-end pipeline for audio retrieval:
    1. Load dataset (Fold 5 as query, Fold 1-4 as database)
    2. Extract features
    3. Compute similarity/distance matrix
    4. Evaluate retrieval performance

    Parameters
    ----------
    feature_type : str
        Feature type: 'mfcc', 'mfcc_delta', 'mel_spectrogram', 'stft'
    distance_method : str
        Distance/similarity method: 'cosine', 'euclidean', 'dtw'
    sr : int
        Target sample rate
    n_fft : int
        FFT window size
    hop_length : int, optional
        Hop length between frames
    n_mfcc : int
        Number of MFCC coefficients
    n_mels : int
        Number of Mel bands
    aggregation : str, optional
        Feature aggregation method: 'mean', 'max', 'std', 'mean_std', or None (keep frames)
    dtw_window : int, optional
        Sakoe-Chiba window for DTW (speeds up computation)

    Examples
    --------
    >>> retrieval = SoundRetrieval(
    ...     feature_type='mfcc',
    ...     distance_method='cosine',
    ...     n_mfcc=13
    ... )
    >>> results = retrieval.run(data_root='ESC-50')
    >>> print(f"MRR@10: {results['MRR@10']:.4f}")
    """

    def __init__(
        self,
        feature_type: str = 'mfcc',
        distance_method: str = 'cosine',
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        n_mfcc: int = 13,
        n_mels: int = 128,
        aggregation: Optional[str] = None,
        dtw_window: Optional[int] = None,
        verbose: bool = True
    ):
        self.feature_type = feature_type
        self.distance_method = distance_method
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.aggregation = aggregation
        self.dtw_window = dtw_window
        self.verbose = verbose

        # Feature extractor
        self.extractor = FeatureExtractor(
            feature_type=feature_type,
            sr=sr,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mfcc=n_mfcc,
            n_mels=n_mels
        )

    def load_and_extract_features(
        self,
        data_root: str,
        folds: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load audio from specified folds and extract features.

        Parameters
        ----------
        data_root : str
            Path to ESC-50 dataset root
        folds : List[int]
            Folds to load (1-5)

        Returns
        -------
        features : np.ndarray
            Extracted features, shape (n_samples, n_features, n_frames) or (n_samples, n_features) if aggregated
        labels : np.ndarray
            Labels for each sample
        filenames : List[str]
            Filenames for each sample
        """
        import pandas as pd
        import librosa

        # Load metadata
        meta_path = os.path.join(data_root, 'meta', 'esc50.csv')
        metadata = pd.read_csv(meta_path)
        metadata = metadata[metadata['fold'].isin(folds)].reset_index(drop=True)

        features_list = []
        labels_list = []
        filenames_list = []

        if self.verbose:
            print(f"Loading {len(metadata)} samples from folds {folds}...")

        iterator = tqdm(range(len(metadata)), desc="Extracting features") if self.verbose else range(len(metadata))

        for idx in iterator:
            row = metadata.iloc[idx]
            filename = row['filename']
            label = row['target']

            # Load audio
            audio_path = os.path.join(data_root, 'audio', filename)
            y, _ = librosa.load(audio_path, sr=self.sr)

            # Extract features
            features = self.extractor.extract(y)

            # Aggregate if specified
            if self.aggregation is not None:
                features = aggregate_features(features, method=self.aggregation)

            features_list.append(features)
            labels_list.append(label)
            filenames_list.append(filename)

        # Stack features
        features = np.stack(features_list, axis=0)
        labels = np.array(labels_list)

        return features, labels, filenames_list

    def compute_similarity(
        self,
        query_features: np.ndarray,
        database_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity matrix between queries and database.

        Parameters
        ----------
        query_features : np.ndarray
            Query features
        database_features : np.ndarray
            Database features

        Returns
        -------
        np.ndarray
            Similarity matrix, shape (n_queries, n_database)
        """
        kwargs = {}
        if self.distance_method == 'dtw' and self.dtw_window is not None:
            kwargs['window'] = self.dtw_window

        # For non-aggregated features with cosine/euclidean, we need to flatten
        if self.aggregation is None and self.distance_method in ['cosine', 'euclidean']:
            kwargs['flatten'] = True

        return compute_similarity_matrix(
            query_features,
            database_features,
            method=self.distance_method,
            **kwargs
        )

    def run(
        self,
        data_root: str = 'ESC-50',
        query_folds: List[int] = [5],
        database_folds: List[int] = [1, 2, 3, 4],
        k_values: List[int] = [10, 20]
    ) -> Dict[str, Any]:
        """
        Run the full retrieval pipeline.

        Parameters
        ----------
        data_root : str
            Path to ESC-50 dataset root
        query_folds : List[int]
            Folds to use as queries (default: [5])
        database_folds : List[int]
            Folds to use as database (default: [1, 2, 3, 4])
        k_values : List[int]
            K values for evaluation metrics

        Returns
        -------
        Dict[str, Any]
            Results dictionary containing:
            - Evaluation metrics (MRR, NDCG, Recall, Precision at each K)
            - Timing information
            - Configuration
        """
        results = {
            'config': self.get_config(),
            'timing': {}
        }

        # Step 1: Extract features
        if self.verbose:
            print("\n" + "=" * 60)
            print("Sound Retrieval Experiment")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Feature type: {self.feature_type}")
            print(f"  Distance method: {self.distance_method}")
            print(f"  Sample rate: {self.sr}")
            print(f"  n_fft: {self.n_fft}, hop_length: {self.hop_length}")
            if self.feature_type in ['mfcc', 'mfcc_delta']:
                print(f"  n_mfcc: {self.n_mfcc}")
            print(f"  Aggregation: {self.aggregation}")

        # Load query set
        start_time = time.time()
        if self.verbose:
            print(f"\n[Step 1] Loading query set (Fold {query_folds})...")
        query_features, query_labels, query_filenames = self.load_and_extract_features(
            data_root, query_folds
        )
        results['timing']['query_extraction'] = time.time() - start_time

        # Load database
        start_time = time.time()
        if self.verbose:
            print(f"\n[Step 2] Loading database (Fold {database_folds})...")
        db_features, db_labels, db_filenames = self.load_and_extract_features(
            data_root, database_folds
        )
        results['timing']['database_extraction'] = time.time() - start_time

        if self.verbose:
            print(f"\nQuery shape: {query_features.shape}")
            print(f"Database shape: {db_features.shape}")

        # Step 2: Compute similarity matrix
        start_time = time.time()
        if self.verbose:
            print(f"\n[Step 3] Computing {self.distance_method} similarity matrix...")

        similarity_matrix = self.compute_similarity(query_features, db_features)
        results['timing']['similarity_computation'] = time.time() - start_time

        if self.verbose:
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Similarity computation time: {results['timing']['similarity_computation']:.2f}s")

        # Step 3: Get retrieved labels
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarity_matrix, axis=1)

        # Get labels of sorted database items
        max_k = max(k_values)
        retrieved_labels = np.zeros((len(query_labels), max_k), dtype=int)
        for i in range(len(query_labels)):
            for j in range(max_k):
                db_idx = sorted_indices[i, j]
                retrieved_labels[i, j] = db_labels[db_idx]

        # Step 4: Evaluate
        start_time = time.time()
        if self.verbose:
            print(f"\n[Step 4] Evaluating retrieval performance...")

        metrics = evaluate_retrieval(
            query_labels,
            retrieved_labels,
            k_values=k_values,
            verbose=self.verbose
        )
        results['timing']['evaluation'] = time.time() - start_time

        # Merge metrics into results
        results.update(metrics)

        # Total time
        total_time = sum(results['timing'].values())
        results['timing']['total'] = total_time

        if self.verbose:
            print(f"\nTotal time: {total_time:.2f}s")

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return {
            'feature_type': self.feature_type,
            'distance_method': self.distance_method,
            'sr': self.sr,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mfcc': self.n_mfcc,
            'n_mels': self.n_mels,
            'aggregation': self.aggregation,
            'dtw_window': self.dtw_window
        }


def run_grid_search(
    data_root: str = 'ESC-50',
    feature_types: List[str] = ['mfcc'],
    distance_methods: List[str] = ['cosine'],
    n_fft_values: List[int] = [2048],
    hop_length_values: List[int] = [512],
    n_mfcc_values: List[int] = [13],
    aggregations: List[Optional[str]] = [None],
    k_values: List[int] = [10, 20],
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run grid search over hyperparameters.

    Parameters
    ----------
    data_root : str
        Path to ESC-50 dataset
    feature_types : List[str]
        Feature types to try
    distance_methods : List[str]
        Distance methods to try
    n_fft_values : List[int]
        FFT sizes to try
    hop_length_values : List[int]
        Hop lengths to try
    n_mfcc_values : List[int]
        MFCC coefficients to try
    aggregations : List[Optional[str]]
        Aggregation methods to try
    k_values : List[int]
        K values for evaluation
    output_file : str, optional
        Save results to JSON file

    Returns
    -------
    List[Dict]
        List of results for each configuration
    """
    all_results = []

    # Generate all configurations
    configs = []
    for feature_type in feature_types:
        for distance_method in distance_methods:
            for n_fft in n_fft_values:
                for hop_length in hop_length_values:
                    for n_mfcc in n_mfcc_values:
                        for aggregation in aggregations:
                            configs.append({
                                'feature_type': feature_type,
                                'distance_method': distance_method,
                                'n_fft': n_fft,
                                'hop_length': hop_length,
                                'n_mfcc': n_mfcc,
                                'aggregation': aggregation
                            })

    print(f"Running grid search with {len(configs)} configurations...")

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*60}")

        retrieval = SoundRetrieval(**config, verbose=True)
        results = retrieval.run(data_root=data_root, k_values=k_values)

        all_results.append(results)

        # Save intermediate results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Grid Search Summary")
    print("=" * 80)
    print(f"{'Feature':>12} | {'Distance':>10} | {'n_fft':>6} | {'hop':>5} | {'MRR@10':>8} | {'Recall@10':>10}")
    print("-" * 80)

    for result in all_results:
        config = result['config']
        print(f"{config['feature_type']:>12} | "
              f"{config['distance_method']:>10} | "
              f"{config['n_fft']:>6} | "
              f"{config['hop_length']:>5} | "
              f"{result.get('MRR@10', 0):>8.4f} | "
              f"{result.get('Recall@10', 0):>10.2%}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Sound Retrieval on ESC-50 Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_root', type=str, default='ESC-50',
                        help='Path to ESC-50 dataset')

    # Feature arguments
    parser.add_argument('--feature', type=str, default='mfcc',
                        choices=['mfcc', 'mfcc_delta', 'mel_spectrogram', 'stft'],
                        help='Feature type')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC coefficients')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of Mel bands')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate')

    # Distance arguments
    parser.add_argument('--distance', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'dtw'],
                        help='Distance/similarity method')
    parser.add_argument('--dtw_window', type=int, default=None,
                        help='Sakoe-Chiba window for DTW')

    # Aggregation
    parser.add_argument('--aggregation', type=str, default=None,
                        choices=['mean', 'max', 'std', 'mean_std', None],
                        help='Feature aggregation method')

    # Evaluation
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 20],
                        help='K values for evaluation')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    # Grid search
    parser.add_argument('--grid_search', action='store_true',
                        help='Run grid search')

    args = parser.parse_args()

    if args.grid_search:
        # Run grid search with default parameters
        results = run_grid_search(
            data_root=args.data_root,
            feature_types=['mfcc', 'mfcc_delta'],
            distance_methods=['cosine', 'euclidean'],
            n_fft_values=[1024, 2048],
            hop_length_values=[256, 512],
            n_mfcc_values=[13, 20],
            aggregations=[None, 'mean'],
            k_values=args.k_values,
            output_file=args.output
        )
    else:
        # Run single experiment
        retrieval = SoundRetrieval(
            feature_type=args.feature,
            distance_method=args.distance,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
            aggregation=args.aggregation,
            dtw_window=args.dtw_window
        )

        results = retrieval.run(
            data_root=args.data_root,
            k_values=args.k_values
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
