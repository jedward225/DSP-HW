#!/usr/bin/env python3
"""
Deep Learning Retriever Evaluation Script

This script evaluates trained deep learning retrievers (Autoencoder, CNN, Contrastive)
on the ESC-50 dataset with 5-fold cross-validation.

Prerequisites:
    Run the training scripts first to generate model checkpoints:
    - python experiments/training/train_autoencoder.py --data_dir ESC-50
    - python experiments/training/train_cnn.py --data_dir ESC-50
    - python experiments/training/train_contrastive.py --data_dir ESC-50

Usage:
    python run_deep_retrievers.py [--models-dir MODELS_DIR] [--output OUTPUT_DIR]
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
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.metrics.retrieval_metrics import aggregate_metrics
from src.metrics.bootstrap import aggregate_metrics_with_ci

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
    return logging.getLogger('deep_retrieval_experiment')


def find_model_checkpoints(models_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Find available model checkpoints.

    Returns dict mapping model name to checkpoint path (or None if not found).
    """
    checkpoints = {
        'Autoencoder': None,
        'CNN': None,
        'Contrastive': None,
    }

    # Autoencoder: look for best model
    ae_path = models_dir / 'autoencoder_esc50_best.pt'
    if ae_path.exists():
        checkpoints['Autoencoder'] = ae_path
    else:
        ae_final = models_dir / 'autoencoder_esc50_final.pt'
        if ae_final.exists():
            checkpoints['Autoencoder'] = ae_final

    # CNN: 5-fold CV produces per-fold checkpoints, use fold 5 by default
    # or any available fold
    for fold in [5, 1, 2, 3, 4]:
        cnn_path = models_dir / f'cnn_esc50_fold{fold}.pt'
        if cnn_path.exists():
            checkpoints['CNN'] = cnn_path
            break

    # Contrastive: look for best model
    cont_path = models_dir / 'contrastive_esc50_best.pt'
    if cont_path.exists():
        checkpoints['Contrastive'] = cont_path
    else:
        cont_final = models_dir / 'contrastive_esc50_final.pt'
        if cont_final.exists():
            checkpoints['Contrastive'] = cont_final

    return checkpoints


def create_deep_retrievers(
    checkpoints: Dict[str, Optional[Path]],
    device: str,
    sr: int,
) -> Dict:
    """
    Create deep learning retrievers from trained checkpoints.

    Only creates retrievers for available checkpoints.
    """
    from src.retrieval.autoencoder_retriever import AutoencoderRetriever
    from src.retrieval.cnn_retriever import CNNRetriever
    from src.retrieval.contrastive_retriever import ContrastiveRetriever

    methods = {}

    if checkpoints['Autoencoder'] is not None:
        try:
            methods['Deep_Autoencoder'] = AutoencoderRetriever(
                model_path=str(checkpoints['Autoencoder']),
                name='Deep_Autoencoder',
                device=device,
                sr=sr,
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load Autoencoder: {e}[/yellow]")

    if checkpoints['CNN'] is not None:
        try:
            methods['Deep_CNN'] = CNNRetriever(
                model_path=str(checkpoints['CNN']),
                name='Deep_CNN',
                device=device,
                sr=sr,
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load CNN: {e}[/yellow]")

    if checkpoints['Contrastive'] is not None:
        try:
            methods['Deep_Contrastive'] = ContrastiveRetriever(
                model_path=str(checkpoints['Contrastive']),
                name='Deep_Contrastive',
                device=device,
                sr=sr,
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load Contrastive: {e}[/yellow]")

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


def run_experiment(
    models_dir: Path,
    config: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    """Run the deep retriever evaluation experiment."""

    # Get settings
    dataset_cfg = config.get('dataset', {})
    eval_cfg = config.get('evaluation', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = 'cpu'

    sr = dataset_cfg.get('sr', 22050)
    folds = eval_cfg.get('folds', None)
    if folds is None:
        num_folds = eval_cfg.get('num_folds', 5)
        folds = list(range(1, num_folds + 1))
    else:
        folds = [int(f) for f in folds]

    # Print experiment header
    console.print(Panel.fit(
        "[bold blue]Deep Learning Retriever Evaluation[/bold blue]\n"
        f"Dataset: ESC-50 | Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("DEEP RETRIEVER EXPERIMENT STARTED")
    logger.info(f"Device: {device}")
    logger.info(f"Models dir: {models_dir}")
    logger.info(f"Folds: {folds}")
    logger.info("=" * 60)

    # Find model checkpoints
    console.print("\n[bold]Searching for model checkpoints...[/bold]")
    checkpoints = find_model_checkpoints(models_dir)

    # For CNN, prefer fold-specific checkpoints if present (avoids fold leakage).
    cnn_fold_checkpoints: Dict[int, Path] = {}
    for fold in folds:
        fold_ckpt = models_dir / f'cnn_esc50_fold{fold}.pt'
        if fold_ckpt.exists():
            cnn_fold_checkpoints[int(fold)] = fold_ckpt

    found_any = False
    for name, path in checkpoints.items():
        if path is not None:
            console.print(f"  [green]✓[/green] {name}: {path.name}")
            logger.info(f"Found {name}: {path}")
            found_any = True
        else:
            console.print(f"  [red]✗[/red] {name}: not found")
            logger.info(f"Missing {name}")

    if cnn_fold_checkpoints:
        logger.info(
            f"CNN fold checkpoints available: {sorted(cnn_fold_checkpoints.keys())} (will use per-fold checkpoints when evaluating Deep_CNN)"
        )

    if not found_any:
        console.print("\n[red]No model checkpoints found![/red]")
        console.print("Please run training scripts first:")
        console.print("  python experiments/training/train_autoencoder.py --data_dir ESC-50")
        console.print("  python experiments/training/train_cnn.py --data_dir ESC-50")
        console.print("  python experiments/training/train_contrastive.py --data_dir ESC-50")
        return None

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(
        root_dir=str(dataset_root),
        sr=sr,
        preload=dataset_cfg.get('preload', False)
    )
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples, {dataset.NUM_CLASSES} classes")
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Create methods
    console.print("\n[bold]Initializing deep learning retrievers...[/bold]")
    methods = create_deep_retrievers(checkpoints, device, sr)

    if not methods:
        console.print("[red]No retrievers could be initialized![/red]")
        return None

    for name in methods:
        console.print(f"  [green]✓[/green] {name}")
    logger.info(f"Methods initialized: {list(methods.keys())}")

    # Results storage
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'models_dir': str(models_dir),
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
    ) as progress:

        total_tasks = len(methods) * len(folds) * queries_per_fold
        overall_task = progress.add_task(
            "[cyan]Overall Progress",
            total=total_tasks
        )

        for method_name, method in methods.items():
            method_results = {
                'mean': {},
                'std': {},
                'ci': {},
                'folds': {}
            }
            fold_metrics = []
            if method_name == 'Deep_CNN' and not cnn_fold_checkpoints:
                logger.warning(
                    "Deep_CNN is being evaluated without fold-specific checkpoints; results may be optimistic due to fold leakage."
                )

            method_task = progress.add_task(
                f"[yellow]{method_name}",
                total=len(folds) * queries_per_fold
            )

            for fold in folds:
                # Get fold split
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                fold_method = method
                if method_name == 'Deep_CNN':
                    fold_ckpt = cnn_fold_checkpoints.get(int(fold))
                    if fold_ckpt is not None:
                        from src.retrieval.cnn_retriever import CNNRetriever

                        fold_method = CNNRetriever(
                            model_path=str(fold_ckpt),
                            name='Deep_CNN',
                            device=device,
                            sr=sr,
                        )
                        logger.info(f"Deep_CNN fold {fold}: using checkpoint {fold_ckpt.name}")
                    else:
                        fallback = getattr(method, 'model_path', None)
                        if fallback:
                            logger.warning(
                                f"Deep_CNN fold {fold}: missing cnn_esc50_fold{fold}.pt; falling back to {Path(fallback).name}"
                            )
                        else:
                            logger.warning(
                                f"Deep_CNN fold {fold}: missing cnn_esc50_fold{fold}.pt and no fallback checkpoint path found"
                            )

                # Evaluate
                fold_result = evaluate_method_fold(
                    fold_method,
                    query_samples,
                    gallery_samples,
                    progress,
                    method_task,
                    logger,
                )

                # Release per-fold CNN model to avoid holding multiple checkpoints on GPU.
                if method_name == 'Deep_CNN' and fold_ckpt is not None:
                    fold_method.clear_gallery()
                    del fold_method
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                method_results['folds'][str(fold)] = fold_result
                fold_metrics.append(fold_result)

                progress.update(overall_task, advance=queries_per_fold)

            # Compute mean, std, and bootstrap CI across folds
            aggregated_with_ci = aggregate_metrics_with_ci(fold_metrics)
            for metric, stats in aggregated_with_ci.items():
                method_results['mean'][metric] = float(stats['mean'])
                method_results['std'][metric] = float(stats['std'])
                method_results['ci'][metric] = {
                    'lower': float(stats['ci_lower']),
                    'upper': float(stats['ci_upper']),
                    'width': float(stats['ci_width']),
                }

            results['methods'][method_name] = method_results

            # Log results
            logger.info(f"Results for {method_name}:")
            for metric, value in method_results['mean'].items():
                ci = method_results['ci'].get(metric, {})
                ci_str = f" [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]" if ci else ""
                logger.info(f"  {metric}: {value:.4f} ± {method_results['std'][metric]:.4f}{ci_str}")

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
        title="[bold]Deep Retriever Results (Mean ± Std across folds)[/bold]",
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
        if results['methods']:
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained deep learning retrievers on ESC-50'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'retrieval' / 'configs' / 'default.yaml'),
        help='Path to config file'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default=str(PROJECT_ROOT / 'models'),
        help='Directory containing trained model checkpoints'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/deep_retrievers/<timestamp>)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'deep_retrievers' / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    # Run experiment
    models_dir = Path(args.models_dir)
    results = run_experiment(models_dir, config, output_dir, logger)

    if results:
        console.print(f"\n[bold green]Experiment completed![/bold green]")
        console.print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
