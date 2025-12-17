#!/usr/bin/env python3
"""
Sound Retrieval Experiment Script

This script runs all retrieval methods (M1-M7) on the ESC-50 dataset
with 5-fold cross-validation.

Usage:
    python run_retrieval.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset, create_fold_splits
from src.retrieval import (
    create_method_m1,
    create_method_m2,
    create_method_m3,
    create_method_m4,
    create_method_m5,
    create_method_m6,
    create_method_m7,
)
from src.metrics.retrieval_metrics import aggregate_metrics
from src.metrics.bootstrap import aggregate_metrics_with_ci
from src.utils.logging import ExperimentLogger
from src.utils.seed import get_seed_from_config, set_seed

# Initialize rich console
console = Console()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file."""
    log_file = output_dir / 'experiment.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
        ]
    )
    return logging.getLogger('retrieval_experiment')


def create_methods(config: Dict, device: str, sr: int) -> Dict:
    """Create all retrieval methods based on configuration."""
    methods = {}
    feat_cfg = config.get('features', {})
    methods_cfg = config.get('methods', {})

    # Common parameters
    common_params = {
        'device': device,
        'sr': sr,
        'n_mfcc': feat_cfg.get('n_mfcc', 20),
        'n_mels': feat_cfg.get('n_mels', 128),
        'n_fft': feat_cfg.get('n_fft', 2048),
        'hop_length': feat_cfg.get('hop_length', 512),
        'fmin': feat_cfg.get('fmin', 0.0),
        'fmax': feat_cfg.get('fmax', None),
        'window': feat_cfg.get('window', 'hann'),
    }

    # M1: MFCC + Pool + Cosine
    if methods_cfg.get('M1_MFCC_Pool_Cos', True):
        methods['M1_MFCC_Pool_Cos'] = create_method_m1(**common_params)

    # M2: MFCC + Delta + Pool
    if methods_cfg.get('M2_MFCC_Delta_Pool', True):
        methods['M2_MFCC_Delta_Pool'] = create_method_m2(
            **common_params,
            delta_width=feat_cfg.get('delta_width', 9)
        )

    # M3: LogMel + Pool
    if methods_cfg.get('M3_LogMel_Pool', True):
        methods['M3_LogMel_Pool'] = create_method_m3(**common_params)

    # M4: Spectral Statistics
    if methods_cfg.get('M4_Spectral_Stat', True):
        methods['M4_Spectral_Stat'] = create_method_m4(**common_params)

    # M5: DTW
    if methods_cfg.get('M5_MFCC_DTW', True):
        dtw_cfg = config.get('dtw', {})
        methods['M5_MFCC_DTW'] = create_method_m5(
            device='cpu',  # DTW always uses CPU with Numba
            sr=sr,
            n_mfcc=dtw_cfg.get('n_mfcc', 13),
            n_mels=dtw_cfg.get('n_mels', 64),
            n_fft=feat_cfg.get('n_fft', 2048),
            hop_length=feat_cfg.get('hop_length', 512),
            sakoe_chiba_radius=dtw_cfg.get('sakoe_chiba_radius', -1),
            use_delta=dtw_cfg.get('use_delta', False),
        )

    # M6: BoAW
    if methods_cfg.get('M6_BoAW_ChiSq', True):
        boaw_cfg = config.get('boaw', {})
        methods['M6_BoAW_ChiSq'] = create_method_m6(
            device=device,
            sr=sr,
            n_mfcc=boaw_cfg.get('n_mfcc', 13),
            n_mels=boaw_cfg.get('n_mels', 64),
            n_fft=feat_cfg.get('n_fft', 2048),
            hop_length=feat_cfg.get('hop_length', 512),
            n_clusters=boaw_cfg.get('n_clusters', 128),
        )

    # M7: Multi-resolution
    if methods_cfg.get('M7_MultiRes_Fusion', True):
        multires_cfg = config.get('multires', {})
        methods['M7_MultiRes_Fusion'] = create_method_m7(
            device=device,
            sr=sr,
            short_n_fft=multires_cfg.get('short_n_fft', 1024),
            short_hop_length=multires_cfg.get('short_hop_length', 256),
            long_n_fft=multires_cfg.get('long_n_fft', 4096),
            long_hop_length=multires_cfg.get('long_hop_length', 1024),
            n_mfcc=feat_cfg.get('n_mfcc', 20),
            n_mels=feat_cfg.get('n_mels', 128),
            fusion_weights=tuple(multires_cfg.get('fusion_weights', [0.5, 0.5])),
        )

    return methods


def evaluate_method_fold(
    method,
    query_samples: List[Dict],
    gallery_samples: List[Dict],
    progress: Progress,
    task_id: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Evaluate a single method on one fold."""
    # Build gallery
    method.build_gallery(gallery_samples, show_progress=False)

    # Evaluate queries
    all_metrics = []
    for i, query in enumerate(query_samples):
        metrics = method.evaluate_query(query)
        all_metrics.append(metrics)
        progress.update(task_id, advance=1)

    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)

    # Extract mean values
    result = {}
    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and 'mean' in stats:
            result[metric_name] = stats['mean']

    return result


def run_experiment(config: Dict, output_dir: Path, logger: logging.Logger):
    """Run the full retrieval experiment."""

    # Get settings
    dataset_cfg = config.get('dataset', {})
    eval_cfg = config.get('evaluation', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    sr = dataset_cfg.get('sr', 22050)
    folds = eval_cfg.get('folds', None)
    if folds is None:
        num_folds = eval_cfg.get('num_folds', 5)
        folds = list(range(1, num_folds + 1))
    else:
        folds = [int(f) for f in folds]
        num_folds = len(folds)

    # Print experiment header
    console.print(Panel.fit(
        "[bold blue]Sound Retrieval Experiment[/bold blue]\n"
        f"Dataset: ESC-50 | Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("EXPERIMENT STARTED")
    logger.info(f"Device: {device}")
    logger.info(f"Sample rate: {sr}")
    logger.info(f"Folds: {folds}")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(
        root_dir=str(dataset_root),
        sr=sr,
        preload=dataset_cfg.get('preload', False)
    )
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples, {dataset.NUM_CLASSES} classes")
    logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.NUM_CLASSES} classes")

    # Create methods
    console.print("\n[bold]Initializing retrieval methods...[/bold]")
    methods = create_methods(config, device, sr)
    for name in methods:
        console.print(f"  [green]✓[/green] {name}")
    logger.info(f"Methods initialized: {list(methods.keys())}")

    # Results storage
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }

    queries_per_fold = len(dataset) // dataset.NUM_FOLDS

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:

        # Overall progress
        overall_task = progress.add_task(
            "[cyan]Overall Progress",
            total=len(methods) * len(folds)
        )

        # Run each method
        for method_name, method in methods.items():
            method_results = {
                'folds': {},
                'mean': {},
                'std': {}
            }

            # Add method task
            method_task = progress.add_task(
                f"[yellow]{method_name}",
                total=len(folds) * queries_per_fold
            )

            fold_metrics = []

            logger.info(f"Running method: {method_name}")

            for fold in folds:
                # Get train/test split
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                logger.info(f"  Fold {fold}: {len(query_samples)} queries, {len(gallery_samples)} gallery")

                # Reset method
                method.clear_gallery()

                # Evaluate
                fold_result = evaluate_method_fold(
                    method, query_samples, gallery_samples,
                    progress, method_task, logger
                )

                method_results['folds'][str(fold)] = fold_result
                fold_metrics.append(fold_result)

                progress.update(overall_task, advance=1)

            # Compute mean, std, and bootstrap CI across folds
            seed = get_seed_from_config(config)
            aggregated_with_ci = aggregate_metrics_with_ci(fold_metrics, random_state=seed)
            method_results['ci'] = {}
            for metric, stats in aggregated_with_ci.items():
                method_results['mean'][metric] = float(stats['mean'])
                method_results['std'][metric] = float(stats['std'])
                method_results['ci'][metric] = {
                    'lower': float(stats['ci_lower']),
                    'upper': float(stats['ci_upper']),
                    'width': float(stats['ci_width']),
                }

            results['methods'][method_name] = method_results

            # Log results with CI
            logger.info(f"  Results for {method_name}:")
            for metric, value in method_results['mean'].items():
                ci = method_results['ci'].get(metric, {})
                ci_str = f" [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]" if ci else ""
                logger.info(f"    {metric}: {value:.4f} ± {method_results['std'][metric]:.4f}{ci_str}")

            # Remove completed task
            progress.remove_task(method_task)

    # Display results table
    console.print("\n")
    display_results_table(results, console)

    # Save results
    save_results(results, output_dir, logger)

    return results


def display_results_table(results: Dict, console: Console):
    """Display results in a formatted table."""
    table = Table(
        title="[bold]Retrieval Results (Mean ± Std across folds)[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    table.add_column("Method", style="bold")
    table.add_column("Hit@10", justify="center")
    table.add_column("P@10", justify="center")
    table.add_column("MRR@10", justify="center")
    table.add_column("mAP@20", justify="center")
    table.add_column("NDCG@10", justify="center")

    # Add rows
    for method_name, method_results in results['methods'].items():
        mean = method_results['mean']
        std = method_results['std']

        def fmt(metric):
            m = mean.get(metric, 0)
            s = std.get(metric, 0)
            return f"{m:.3f}±{s:.3f}"

        table.add_row(
            method_name,
            fmt('hit@10'),
            fmt('precision@10'),
            fmt('mrr@10'),
            fmt('map@20'),
            fmt('ndcg@10'),
        )

    console.print(table)


def save_results(results: Dict, output_dir: Path, logger: logging.Logger):
    """Save results to files."""
    # Save JSON
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]✓[/green] Results saved to {json_path}")
    logger.info(f"Results saved to {json_path}")

    # Save CSV summary
    csv_path = output_dir / 'results_summary.csv'
    with open(csv_path, 'w') as f:
        # Header
        metrics = list(list(results['methods'].values())[0]['mean'].keys())
        f.write('Method,' + ','.join([f'{m}_mean,{m}_std' for m in metrics]) + '\n')

        # Rows
        for method_name, method_results in results['methods'].items():
            row = [method_name]
            for m in metrics:
                row.append(f"{method_results['mean'].get(m, 0):.4f}")
                row.append(f"{method_results['std'].get(m, 0):.4f}")
            f.write(','.join(row) + '\n')

    console.print(f"[green]✓[/green] Summary saved to {csv_path}")
    logger.info(f"Summary saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Sound Retrieval Experiment")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'retrieval' / 'configs' / 'default.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / config.get('output', {}).get('results_dir', 'experiments/retrieval/results') / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)

    seed = get_seed_from_config(config)
    if seed is not None:
        set_seed(seed, deterministic=bool(config.get('deterministic', False)))
        logger.info(f"Random seed set to {seed}")

    try:
        # Run experiment
        results = run_experiment(config, output_dir, logger)

        console.print(Panel.fit(
            "[bold green]Experiment completed successfully![/bold green]\n"
            f"Results saved to: {output_dir}",
            border_style="green"
        ))

    except Exception as e:
        logger.exception("Experiment failed")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
