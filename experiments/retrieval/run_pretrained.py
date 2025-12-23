#!/usr/bin/env python3
"""
Pretrained Model Retriever Evaluation Script

This script evaluates pretrained audio models (CLAP, BEATs, Hybrid) on the ESC-50
dataset with 5-fold cross-validation.

Prerequisites:
    1. Download CLAP checkpoint:
       - URL: https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt
       - Place at: laion_clap/music_audioset_epoch_15_esc_90.14.pt

    2. Clone CLAP source:
       git clone https://github.com/LAION-AI/CLAP.git

    3. Download BEATs checkpoint:
       - URL: https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
       - Place at: beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt

    4. Clone BEATs source:
       git clone https://github.com/microsoft/unilm.git

Usage:
    python run_pretrained.py [--methods clap,beats,hybrid] [--output OUTPUT_DIR]

    # Run only CLAP
    python run_pretrained.py --methods clap

    # Run only BEATs
    python run_pretrained.py --methods beats

    # Run all (default)
    python run_pretrained.py --methods clap,beats,hybrid
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


# Default checkpoint paths
DEFAULT_CLAP_CHECKPOINT = PROJECT_ROOT / 'laion_clap' / 'music_audioset_epoch_15_esc_90.14.pt'
DEFAULT_BEATS_CHECKPOINT = PROJECT_ROOT / 'beats' / 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'


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
    return logging.getLogger('pretrained_retrieval_experiment')


def check_prerequisites() -> Dict[str, Dict]:
    """
    Check if pretrained model prerequisites are available.

    Returns dict with availability status and paths for each model.
    """
    status = {
        'clap': {
            'available': False,
            'checkpoint': None,
            'source': None,
            'missing': [],
        },
        'beats': {
            'available': False,
            'checkpoint': None,
            'source': None,
            'missing': [],
        },
    }

    # Check CLAP
    clap_checkpoint = DEFAULT_CLAP_CHECKPOINT
    clap_source = PROJECT_ROOT / 'CLAP' / 'src'

    if clap_checkpoint.exists():
        status['clap']['checkpoint'] = clap_checkpoint
    else:
        status['clap']['missing'].append(f"Checkpoint: {clap_checkpoint}")

    if clap_source.exists():
        status['clap']['source'] = clap_source
    else:
        status['clap']['missing'].append(f"Source: {clap_source}")

    status['clap']['available'] = len(status['clap']['missing']) == 0

    # Check BEATs
    beats_checkpoint = DEFAULT_BEATS_CHECKPOINT
    beats_source = PROJECT_ROOT / 'unilm' / 'beats'

    if beats_checkpoint.exists():
        status['beats']['checkpoint'] = beats_checkpoint
    else:
        status['beats']['missing'].append(f"Checkpoint: {beats_checkpoint}")

    if beats_source.exists():
        status['beats']['source'] = beats_source
    else:
        status['beats']['missing'].append(f"Source: {beats_source}")

    status['beats']['available'] = len(status['beats']['missing']) == 0

    return status


def create_pretrained_retrievers(
    requested_methods: List[str],
    prereq_status: Dict,
    device: str,
    sr: int,
) -> Dict:
    """
    Create pretrained model retrievers.

    Args:
        requested_methods: List of methods to create ('clap', 'beats', 'hybrid')
        prereq_status: Prerequisites status from check_prerequisites()
        device: Computation device
        sr: Sample rate

    Returns:
        Dict of method_name -> retriever
    """
    methods = {}

    # CLAP (M8)
    if 'clap' in requested_methods:
        if prereq_status['clap']['available']:
            try:
                from src.retrieval.clap_retriever import CLAPRetriever
                methods['M8_CLAP'] = CLAPRetriever(
                    name='M8_CLAP',
                    device=device,
                    sr=sr,
                    checkpoint_path=str(prereq_status['clap']['checkpoint']),
                )
                console.print(f"  [green]✓[/green] M8_CLAP initialized")
            except Exception as e:
                console.print(f"  [red]✗[/red] M8_CLAP failed: {e}")
        else:
            console.print(f"  [yellow]⊘[/yellow] M8_CLAP skipped (missing prerequisites)")

    # BEATs
    if 'beats' in requested_methods:
        if prereq_status['beats']['available']:
            try:
                from src.retrieval.beats_retriever import BEATsRetriever
                methods['BEATs'] = BEATsRetriever(
                    name='BEATs',
                    device=device,
                    sr=sr,
                    checkpoint_path=str(prereq_status['beats']['checkpoint']),
                )
                console.print(f"  [green]✓[/green] BEATs initialized")
            except Exception as e:
                console.print(f"  [red]✗[/red] BEATs failed: {e}")
        else:
            console.print(f"  [yellow]⊘[/yellow] BEATs skipped (missing prerequisites)")

    # Hybrid (M9) - requires CLAP
    if 'hybrid' in requested_methods:
        if prereq_status['clap']['available']:
            try:
                from src.retrieval.hybrid_retriever import HybridRetriever
                methods['M9_Hybrid'] = HybridRetriever(
                    name='M9_Hybrid',
                    device=device,
                    sr=sr,
                    clap_weight=0.7,
                    mfcc_weight=0.3,
                    clap_checkpoint=str(prereq_status['clap']['checkpoint']),
                )
                console.print(f"  [green]✓[/green] M9_Hybrid initialized (CLAP=0.7 + MFCC=0.3)")
            except Exception as e:
                console.print(f"  [red]✗[/red] M9_Hybrid failed: {e}")
        else:
            console.print(f"  [yellow]⊘[/yellow] M9_Hybrid skipped (requires CLAP)")

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
    requested_methods: List[str],
    config: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    """Run the pretrained model evaluation experiment."""

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
        "[bold blue]Pretrained Model Retriever Evaluation[/bold blue]\n"
        f"Dataset: ESC-50 | Device: {device} | Folds: {folds}\n"
        f"Methods: {', '.join(requested_methods)}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("PRETRAINED MODEL EXPERIMENT STARTED")
    logger.info(f"Device: {device}")
    logger.info(f"Requested methods: {requested_methods}")
    logger.info(f"Folds: {folds}")
    logger.info("=" * 60)

    # Check prerequisites
    console.print("\n[bold]Checking prerequisites...[/bold]")
    prereq_status = check_prerequisites()

    for model, status in prereq_status.items():
        if status['available']:
            console.print(f"  [green]✓[/green] {model.upper()}: ready")
        else:
            console.print(f"  [red]✗[/red] {model.upper()}: missing")
            for missing in status['missing']:
                console.print(f"      - {missing}")

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
    console.print("\n[bold]Initializing pretrained retrievers...[/bold]")
    methods = create_pretrained_retrievers(requested_methods, prereq_status, device, sr)

    if not methods:
        console.print("\n[red]No retrievers could be initialized![/red]")
        console.print("\nTo use these methods, you need to download the prerequisites:")
        console.print("\n[bold]CLAP:[/bold]")
        console.print("  1. git clone https://github.com/LAION-AI/CLAP.git")
        console.print("  2. Download checkpoint from HuggingFace:")
        console.print("     https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt")
        console.print("  3. Place at: laion_clap/music_audioset_epoch_15_esc_90.14.pt")
        console.print("\n[bold]BEATs:[/bold]")
        console.print("  1. git clone https://github.com/microsoft/unilm.git")
        console.print("  2. Download checkpoint:")
        console.print("     https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        console.print("  3. Place at: beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        return None

    logger.info(f"Methods initialized: {list(methods.keys())}")

    # Results storage
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'requested_methods': requested_methods,
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

            method_task = progress.add_task(
                f"[yellow]{method_name}",
                total=len(folds) * queries_per_fold
            )

            for fold in folds:
                # Get fold split
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                # Evaluate
                fold_result = evaluate_method_fold(
                    method,
                    query_samples,
                    gallery_samples,
                    progress,
                    method_task,
                    logger,
                )

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
        title="[bold]Pretrained Model Results (Mean ± Std across folds)[/bold]",
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
        if results['methods']:
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pretrained audio model retrievers (CLAP, BEATs, Hybrid) on ESC-50'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'retrieval' / 'configs' / 'default.yaml'),
        help='Path to config file'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default='clap,beats,hybrid',
        help='Comma-separated list of methods to evaluate: clap, beats, hybrid (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/pretrained/<timestamp>)'
    )

    args = parser.parse_args()

    # Parse methods
    requested_methods = [m.strip().lower() for m in args.methods.split(',')]
    valid_methods = {'clap', 'beats', 'hybrid'}
    for m in requested_methods:
        if m not in valid_methods:
            console.print(f"[red]Invalid method: {m}. Valid options: {valid_methods}[/red]")
            sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'pretrained' / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    # Run experiment
    results = run_experiment(requested_methods, config, output_dir, logger)

    if results:
        console.print(f"\n[bold green]Experiment completed![/bold green]")
        console.print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
