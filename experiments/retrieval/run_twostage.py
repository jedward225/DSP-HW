#!/usr/bin/env python3
"""
Two-Stage Retrieval Experiments

This script tests two-stage retrieval with varying N (candidates from stage 1):
  Stage 1: Fast M1/M3 retrieval → Top-N candidates
  Stage 2: DTW re-ranking → Final Top-K

Usage:
    python run_twostage.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Rich imports
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.retrieval import (
    create_method_m1,
    create_method_m3,
    create_method_m5,
    create_twostage_retriever,
)
from src.metrics.retrieval_metrics import aggregate_metrics

console = Console()


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'twostage.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('twostage')


def evaluate_retriever(
    retriever,
    query_samples: List[dict],
    gallery_samples: List[dict],
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate retriever and return metrics."""
    # Build gallery
    retriever.build_gallery(gallery_samples, show_progress=False)

    # Evaluate queries
    all_metrics = []
    total_time = 0

    for query in query_samples:
        start = time.perf_counter()
        metrics = retriever.evaluate_query(query)
        total_time += time.perf_counter() - start
        all_metrics.append(metrics)

        if progress and task_id:
            progress.update(task_id, advance=1)

    # Aggregate
    agg = aggregate_metrics(all_metrics)
    result = {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}
    result['avg_query_time_ms'] = total_time / len(query_samples) * 1000

    return result


def run_twostage_experiments(config_path: str, output_dir: Path):
    """Run two-stage retrieval experiments."""
    yaml_config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    dataset_cfg = yaml_config.get('dataset', {})
    feat_cfg = yaml_config.get('features', {})
    dtw_cfg = yaml_config.get('dtw', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)
    folds = yaml_config.get('evaluation', {}).get('folds', [1, 2, 3, 4, 5])

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    console.print(Panel.fit(
        "[bold blue]Two-Stage Retrieval Experiments[/bold blue]\n"
        f"Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("TWO-STAGE RETRIEVAL EXPERIMENTS STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # N values to test
    gallery_size = len(dataset) * 4 // 5  # 80% for gallery
    n_values = [20, 50, 100, 200, 500, 1000, gallery_size]
    n_values = [n for n in n_values if n <= gallery_size]

    console.print(f"\n[bold]Testing N values: {n_values}[/bold]")

    # Create base retrievers
    coarse_retriever = create_method_m1(
        device=device,
        sr=sr,
        n_mfcc=feat_cfg.get('n_mfcc', 20),
        n_mels=feat_cfg.get('n_mels', 128),
        n_fft=feat_cfg.get('n_fft', 2048),
        hop_length=feat_cfg.get('hop_length', 512),
    )

    fine_retriever = create_method_m5(
        device='cpu',  # DTW uses CPU/Numba
        sr=sr,
        n_mfcc=dtw_cfg.get('n_mfcc', 13),
        n_mels=dtw_cfg.get('n_mels', 64),
        n_fft=feat_cfg.get('n_fft', 2048),
        hop_length=feat_cfg.get('hop_length', 512),
    )

    results = {}
    queries_per_fold = len(dataset) // 5

    # First, run baseline methods
    console.print("\n[bold cyan]Running Baseline Methods[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # M1 baseline
        task = progress.add_task("[cyan]M1 (Coarse)", total=len(folds) * queries_per_fold)
        m1_fold_metrics = []
        for fold in folds:
            query_samples, gallery_samples = dataset.get_query_gallery_split(fold)
            metrics = evaluate_retriever(coarse_retriever, query_samples, gallery_samples, progress, task)
            m1_fold_metrics.append(metrics)

        m1_results = {
            'mean': {k: float(np.mean([m[k] for m in m1_fold_metrics])) for k in m1_fold_metrics[0].keys()},
            'std': {k: float(np.std([m[k] for m in m1_fold_metrics])) for k in m1_fold_metrics[0].keys()},
        }
        results['M1_baseline'] = m1_results

        # M5 baseline (on small subset due to speed)
        task = progress.add_task("[cyan]M5 (Fine)", total=len(folds) * queries_per_fold)
        m5_fold_metrics = []
        for fold in folds:
            query_samples, gallery_samples = dataset.get_query_gallery_split(fold)
            metrics = evaluate_retriever(fine_retriever, query_samples, gallery_samples, progress, task)
            m5_fold_metrics.append(metrics)

        m5_results = {
            'mean': {k: float(np.mean([m[k] for m in m5_fold_metrics])) for k in m5_fold_metrics[0].keys()},
            'std': {k: float(np.std([m[k] for m in m5_fold_metrics])) for k in m5_fold_metrics[0].keys()},
        }
        results['M5_baseline'] = m5_results

    # Run two-stage with different N
    console.print("\n[bold cyan]Running Two-Stage Retrieval[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total = len(n_values) * len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Two-Stage", total=total)

        for n in n_values:
            progress.update(task, description=f"[cyan]Two-Stage N={n}")

            fold_metrics = []
            for fold in folds:
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                # Create fresh retrievers for each fold
                coarse = create_method_m1(
                    device=device,
                    sr=sr,
                    n_mfcc=feat_cfg.get('n_mfcc', 20),
                    n_mels=feat_cfg.get('n_mels', 128),
                    n_fft=feat_cfg.get('n_fft', 2048),
                    hop_length=feat_cfg.get('hop_length', 512),
                )
                fine = create_method_m5(
                    device='cpu',
                    sr=sr,
                    n_mfcc=dtw_cfg.get('n_mfcc', 13),
                    n_mels=dtw_cfg.get('n_mels', 64),
                    n_fft=feat_cfg.get('n_fft', 2048),
                    hop_length=feat_cfg.get('hop_length', 512),
                )

                twostage = create_twostage_retriever(
                    coarse_retriever=coarse,
                    fine_retriever=fine,
                    top_n=n,
                )

                metrics = evaluate_retriever(twostage, query_samples, gallery_samples, progress, task)
                fold_metrics.append(metrics)

            results[f'TwoStage_N{n}'] = {
                'n': n,
                'mean': {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()},
                'std': {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()},
            }

            logger.info(f"N={n}: hit@10={results[f'TwoStage_N{n}']['mean'].get('hit@10', 0):.4f}")

    # Display results
    console.print("\n")
    table = Table(title="Two-Stage Retrieval Results", box=box.ROUNDED)
    table.add_column("Method", style="bold")
    table.add_column("N", justify="center")
    table.add_column("Hit@10", justify="center")
    table.add_column("mAP@20", justify="center")
    table.add_column("Query Time (ms)", justify="center")

    # Baselines
    table.add_row(
        "M1 (Coarse only)",
        "-",
        f"{m1_results['mean'].get('hit@10', 0):.4f}",
        f"{m1_results['mean'].get('map@20', 0):.4f}",
        f"{m1_results['mean'].get('avg_query_time_ms', 0):.2f}",
        style="dim"
    )
    table.add_row(
        "M5 (Fine only)",
        "-",
        f"{m5_results['mean'].get('hit@10', 0):.4f}",
        f"{m5_results['mean'].get('map@20', 0):.4f}",
        f"{m5_results['mean'].get('avg_query_time_ms', 0):.2f}",
        style="dim"
    )

    table.add_row("─" * 10, "─" * 5, "─" * 10, "─" * 10, "─" * 12)

    # Two-stage results
    for n in n_values:
        key = f'TwoStage_N{n}'
        r = results[key]
        table.add_row(
            f"Two-Stage",
            str(n),
            f"{r['mean'].get('hit@10', 0):.4f}±{r['std'].get('hit@10', 0):.4f}",
            f"{r['mean'].get('map@20', 0):.4f}±{r['std'].get('map@20', 0):.4f}",
            f"{r['mean'].get('avg_query_time_ms', 0):.2f}",
        )

    console.print(table)

    # Save results
    with open(output_dir / 'n_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Retrieval Experiments")
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
        help='Output directory'
    )
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'twostage' / timestamp

    try:
        run_twostage_experiments(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Two-stage experiments completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
