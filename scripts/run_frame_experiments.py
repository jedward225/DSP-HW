#!/usr/bin/env python3
"""
Frame Length / Hop Length Hyperparameter Experiments for Classification.

This script runs ResNet18 classification with different FFT configurations
to compare the effect of frame length (n_fft) and hop length on accuracy.

Usage:
    python scripts/run_frame_experiments.py --gpu 0 --epochs 30
    python scripts/run_frame_experiments.py --quick  # Fast test with fewer epochs
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


# Experiment configurations: (n_fft, hop_length)
FRAME_CONFIGS = [
    (1024, 256),   # Short frame, fine temporal resolution
    (2048, 512),   # Standard (default)
    (4096, 1024),  # Medium frame
    (8192, 2048),  # Long frame, coarse temporal resolution
]

# Config index mapping for parallel runs
FRAME_CONFIG_NAMES = ['1024_256', '2048_512', '4096_1024', '8192_2048']

# Feature types to test
FEATURE_TYPES = ['mel', 'mfcc']


def run_single_experiment(
    n_fft: int,
    hop_length: int,
    feature_type: str,
    epochs: int,
    batch_size: int,
    device: str,
    data_root: str,
    output_dir: Path
) -> dict:
    """Run a single experiment with given configuration."""

    from src.classification.train import train

    exp_name = f"resnet18_{feature_type}_nfft{n_fft}_hop{hop_length}"
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  n_fft={n_fft}, hop_length={hop_length}, feature={feature_type}")
    print(f"{'='*60}")

    try:
        history = train(
            model_type='resnet18',
            mode='standard',
            feature_type=feature_type,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128,
            batch_size=batch_size,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-4,
            device=device,
            data_root=data_root,
            output_dir=str(output_dir),
            use_augment=True,
            num_workers=0,
            mixup_lam=0.0,
            unfreeze_epoch=epochs + 1,  # Don't unfreeze for ResNet
            label_smoothing=0.1
        )

        best_acc = max(history['test_acc'])
        final_acc = history['test_acc'][-1]

        return {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'feature_type': feature_type,
            'best_acc': best_acc,
            'final_acc': final_acc,
            'epochs': epochs,
            'status': 'success'
        }

    except Exception as e:
        print(f"Error in experiment: {e}")
        return {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'feature_type': feature_type,
            'best_acc': 0.0,
            'final_acc': 0.0,
            'epochs': epochs,
            'status': f'error: {str(e)}'
        }


def run_all_experiments(
    epochs: int = 30,
    batch_size: int = 32,
    device: str = 'cuda',
    data_root: str = 'ESC-50',
    feature_types: list = None,
    config_idx: int = None
):
    """Run all frame length/hop length experiments."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use config-specific output dir if running single config
    if config_idx is not None:
        output_dir = PROJECT_ROOT / 'checkpoints' / 'frame_experiments' / f'config{config_idx}_{timestamp}'
    else:
        output_dir = PROJECT_ROOT / 'checkpoints' / 'frame_experiments' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_types = feature_types or FEATURE_TYPES

    # Filter frame configs if specific index provided
    if config_idx is not None:
        frame_configs = [FRAME_CONFIGS[config_idx]]
    else:
        frame_configs = FRAME_CONFIGS

    results = []

    total_experiments = len(frame_configs) * len(feature_types)
    current = 0

    for feature_type in feature_types:
        for n_fft, hop_length in frame_configs:
            current += 1
            print(f"\n[{current}/{total_experiments}] ", end="")

            result = run_single_experiment(
                n_fft=n_fft,
                hop_length=hop_length,
                feature_type=feature_type,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                data_root=data_root,
                output_dir=output_dir
            )
            results.append(result)

            # Save intermediate results
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

    return results, output_dir


def print_results_table(results: list):
    """Print results as a formatted table."""

    print("\n" + "="*70)
    print("FRAME LENGTH / HOP LENGTH EXPERIMENT RESULTS")
    print("="*70)

    # Group by feature type
    feature_types = sorted(set(r['feature_type'] for r in results))

    for feature_type in feature_types:
        print(f"\n## Feature Type: {feature_type.upper()}")
        print("-"*50)
        print(f"{'n_fft':>8} | {'hop_length':>10} | {'Best Acc':>10} | {'Status':>10}")
        print("-"*50)

        feature_results = [r for r in results if r['feature_type'] == feature_type]
        feature_results.sort(key=lambda x: x['best_acc'], reverse=True)

        for r in feature_results:
            status = 'OK' if r['status'] == 'success' else 'FAIL'
            print(f"{r['n_fft']:>8} | {r['hop_length']:>10} | {r['best_acc']:>9.2f}% | {status:>10}")

    print("\n" + "="*70)

    # Find best overall
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x['best_acc'])
        print(f"Best Configuration:")
        print(f"  Feature: {best['feature_type']}")
        print(f"  n_fft: {best['n_fft']}, hop_length: {best['hop_length']}")
        print(f"  Accuracy: {best['best_acc']:.2f}%")
    print("="*70)


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table for the report."""

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{不同帧长/帧移配置下 ResNet18 的分类精度}",
        r"\label{tab:frame_length_exp}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"\textbf{特征类型} & \textbf{(帧长, 帧移)} & \textbf{准确率 (\%)} \\",
        r"\midrule",
    ]

    for feature_type in ['mel', 'mfcc']:
        feature_results = [r for r in results if r['feature_type'] == feature_type and r['status'] == 'success']
        feature_results.sort(key=lambda x: x['n_fft'])

        for i, r in enumerate(feature_results):
            feat_col = feature_type.upper() if i == 0 else ""
            config = f"({r['n_fft']}, {r['hop_length']})"
            acc = f"{r['best_acc']:.2f}"
            lines.append(f"{feat_col} & {config} & {acc} \\\\")

        if feature_type != 'mfcc':
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run frame length/hop experiments')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--data_root', type=str, default='ESC-50', help='Dataset root')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    parser.add_argument('--feature', type=str, default=None, choices=['mel', 'mfcc'],
                        help='Only test specific feature type')
    parser.add_argument('--config_idx', type=int, default=None, choices=[0, 1, 2, 3],
                        help='Only run specific frame config (0-3)')

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        torch.cuda.set_device(args.gpu)
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Quick mode for testing
    epochs = 5 if args.quick else args.epochs

    # Feature types
    feature_types = [args.feature] if args.feature else FEATURE_TYPES

    # Run experiments
    results, output_dir = run_all_experiments(
        epochs=epochs,
        batch_size=args.batch_size,
        device=device,
        data_root=args.data_root,
        feature_types=feature_types,
        config_idx=args.config_idx
    )

    # Print results
    print_results_table(results)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    print("\nLaTeX Table for Report:")
    print(latex_table)

    # Save LaTeX table
    with open(output_dir / 'latex_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
