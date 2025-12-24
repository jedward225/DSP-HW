#!/usr/bin/env python
"""
Grid Search Script for ESC-50 Classification.

Systematically tests different feature extraction parameters and model configurations
to find optimal settings for the classification task.

Usage:
    # Run all experiments
    python scripts/grid_search.py --mode all

    # Run only feature parameter search
    python scripts/grid_search.py --mode features

    # Run quick test (fewer epochs)
    python scripts/grid_search.py --mode quick
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from itertools import product
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Grid search configurations
FEATURE_GRID = {
    'feature_type': ['stft', 'mel', 'mfcc'],
    'n_fft': [1024, 2048, 4096],
    'hop_length': [256, 512, 1024],
    'n_mels': [64, 128, 256],  # Only used for mel/mfcc
}

# Quick test config (fewer combinations)
QUICK_GRID = {
    'feature_type': ['mel'],
    'n_fft': [2048],
    'hop_length': [512],
    'n_mels': [128],
}

# Model variants to test
MODEL_VARIANTS = {
    'resnet18_standard': {
        'model': 'resnet18',
        'variant': 'standard',
    },
    'resnet18_attention': {
        'model': 'resnet18',
        'variant': 'attention',
    },
    'resnet34_standard': {
        'model': 'resnet34',
        'variant': 'standard',
    },
}


def run_experiment(
    feature_type: str,
    n_fft: int,
    hop_length: int,
    n_mels: int = 128,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: str = 'checkpoints/grid_search',
    use_augment: bool = False,
    device: str = 'cuda'
) -> dict:
    """
    Run a single training experiment.

    Returns
    -------
    dict
        Results including best accuracy, config, and training time
    """
    exp_name = f"{feature_type}_nfft{n_fft}_hop{hop_length}"
    if feature_type in ['mel', 'mfcc', 'mfcc_delta']:
        exp_name += f"_mels{n_mels}"

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")

    # Build command
    cmd = [
        sys.executable, '-m', 'src.classification.train',
        '--model', 'resnet18',
        '--feature', feature_type,
        '--n_fft', str(n_fft),
        '--hop', str(hop_length),
        '--n_mels', str(n_mels),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--output_dir', output_dir,
        '--device', device,
    ]

    if use_augment:
        cmd.append('--augment')

    # Run training
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        training_time = time.time() - start_time

        # Parse output for best accuracy
        output = result.stdout + result.stderr
        best_acc = 0.0

        for line in output.split('\n'):
            if 'Best accuracy:' in line or 'best accuracy:' in line:
                try:
                    best_acc = float(line.split(':')[-1].strip().replace('%', ''))
                except:
                    pass
            elif 'Test Acc:' in line:
                try:
                    acc = float(line.split('Test Acc:')[-1].strip().split('%')[0])
                    best_acc = max(best_acc, acc)
                except:
                    pass

        return {
            'exp_name': exp_name,
            'feature_type': feature_type,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'best_acc': best_acc,
            'training_time': training_time,
            'success': result.returncode == 0,
            'output': output[-2000:] if len(output) > 2000 else output  # Last 2000 chars
        }

    except subprocess.TimeoutExpired:
        return {
            'exp_name': exp_name,
            'feature_type': feature_type,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'best_acc': 0.0,
            'training_time': 3600,
            'success': False,
            'output': 'Timeout'
        }
    except Exception as e:
        return {
            'exp_name': exp_name,
            'feature_type': feature_type,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'best_acc': 0.0,
            'training_time': 0,
            'success': False,
            'output': str(e)
        }


def run_grid_search(
    grid: dict,
    epochs: int = 30,
    output_dir: str = 'checkpoints/grid_search',
    device: str = 'cuda',
    use_augment: bool = False
) -> list:
    """Run grid search over feature parameters."""

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"grid_search_results_{timestamp}.json"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    experiments = []
    for feat_type in grid['feature_type']:
        for n_fft in grid['n_fft']:
            for hop_length in grid['hop_length']:
                if feat_type in ['mel', 'mfcc', 'mfcc_delta']:
                    for n_mels in grid['n_mels']:
                        experiments.append({
                            'feature_type': feat_type,
                            'n_fft': n_fft,
                            'hop_length': hop_length,
                            'n_mels': n_mels
                        })
                else:
                    experiments.append({
                        'feature_type': feat_type,
                        'n_fft': n_fft,
                        'hop_length': hop_length,
                        'n_mels': 128  # Default, not used
                    })

    print(f"Total experiments to run: {len(experiments)}")
    print(f"Results will be saved to: {results_file}")

    for i, exp_config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running experiment...")

        result = run_experiment(
            **exp_config,
            epochs=epochs,
            output_dir=output_dir,
            device=device,
            use_augment=use_augment
        )
        results.append(result)

        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  Best Accuracy: {result['best_acc']:.2f}%")
        print(f"  Training Time: {result['training_time']:.1f}s")

    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['best_acc'], reverse=True)

    print("\nTop 5 configurations:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"  {i+1}. {r['exp_name']}: {r['best_acc']:.2f}%")

    print(f"\nFull results saved to: {results_file}")

    return results


def run_model_comparison(
    feature_type: str = 'mel',
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    epochs: int = 30,
    output_dir: str = 'checkpoints/model_comparison',
    device: str = 'cuda'
) -> list:
    """Compare different model variants with same feature config."""

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"model_comparison_{timestamp}.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, config in MODEL_VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        # Build command
        cmd = [
            sys.executable, '-m', 'src.classification.train',
            '--model', config['model'],
            '--feature', feature_type,
            '--n_fft', str(n_fft),
            '--hop', str(hop_length),
            '--n_mels', str(n_mels),
            '--epochs', str(epochs),
            '--output_dir', output_dir,
            '--device', device,
        ]

        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            training_time = time.time() - start_time

            # Parse best accuracy
            output = result.stdout + result.stderr
            best_acc = 0.0
            for line in output.split('\n'):
                if 'Best accuracy:' in line:
                    try:
                        best_acc = float(line.split(':')[-1].strip().replace('%', ''))
                    except:
                        pass

            results.append({
                'model_name': name,
                'config': config,
                'best_acc': best_acc,
                'training_time': training_time,
                'success': result.returncode == 0
            })

        except Exception as e:
            results.append({
                'model_name': name,
                'config': config,
                'best_acc': 0.0,
                'training_time': 0,
                'success': False,
                'error': str(e)
            })

        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    for r in sorted(results, key=lambda x: x['best_acc'], reverse=True):
        print(f"  {r['model_name']}: {r['best_acc']:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Grid Search for ESC-50 Classification')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'features', 'models', 'quick'],
                        help='Search mode')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per experiment')
    parser.add_argument('--output_dir', type=str, default='checkpoints/grid_search',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')

    args = parser.parse_args()

    if args.mode == 'quick':
        print("Running quick test with reduced grid...")
        run_grid_search(
            grid=QUICK_GRID,
            epochs=10,
            output_dir=args.output_dir,
            device=args.device
        )

    elif args.mode == 'features':
        print("Running feature parameter grid search...")
        run_grid_search(
            grid=FEATURE_GRID,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            use_augment=args.augment
        )

    elif args.mode == 'models':
        print("Running model comparison...")
        run_model_comparison(
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device
        )

    elif args.mode == 'all':
        print("Running full grid search (features + models)...")

        # First, find best feature config
        print("\n" + "=" * 70)
        print("PHASE 1: Feature Parameter Search")
        print("=" * 70)
        feature_results = run_grid_search(
            grid=FEATURE_GRID,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            use_augment=args.augment
        )

        # Get best feature config
        best_feature = max(feature_results, key=lambda x: x['best_acc'])
        print(f"\nBest feature config: {best_feature['exp_name']} ({best_feature['best_acc']:.2f}%)")

        # Then, compare models with best feature config
        print("\n" + "=" * 70)
        print("PHASE 2: Model Comparison with Best Features")
        print("=" * 70)
        model_results = run_model_comparison(
            feature_type=best_feature['feature_type'],
            n_fft=best_feature['n_fft'],
            hop_length=best_feature['hop_length'],
            n_mels=best_feature['n_mels'],
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device
        )


if __name__ == '__main__':
    main()
