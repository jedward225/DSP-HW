#!/usr/bin/env python3
"""
Multi-Feature Fusion Experiments

This script tests:
  1. Late Fusion: Weighted combination of distances
  2. Rank Fusion: Reciprocal Rank Fusion (RRF)

Usage:
    python run_fusion.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from itertools import product
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
    create_method_m2,
    create_method_m3,
    create_method_m4,
    create_late_fusion,
    create_rank_fusion,
)
from src.metrics.retrieval_metrics import aggregate_metrics
from src.utils.seed import get_seed_from_config, set_seed

console = Console()


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'fusion.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('fusion')


def create_base_retrievers(config: Dict, device: str, sr: int) -> Dict:
    """Create base retrievers for fusion."""
    feat_cfg = config.get('features', {})

    common_params = {
        'device': device,
        'sr': sr,
        'n_mfcc': feat_cfg.get('n_mfcc', 20),
        'n_mels': feat_cfg.get('n_mels', 128),
        'n_fft': feat_cfg.get('n_fft', 2048),
        'hop_length': feat_cfg.get('hop_length', 512),
    }

    return {
        'M1_MFCC': create_method_m1(**common_params),
        'M2_MFCC_Delta': create_method_m2(**common_params, delta_width=9),
        'M3_LogMel': create_method_m3(**common_params),
        'M4_Spectral': create_method_m4(**common_params),
    }


def evaluate_retriever(
    retriever,
    query_samples: List[dict],
    gallery_samples: List[dict],
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a retriever on query/gallery split."""
    # Build gallery
    retriever.build_gallery(gallery_samples, show_progress=False)

    # Evaluate queries
    all_metrics = []
    for query in query_samples:
        metrics = retriever.evaluate_query(query)
        all_metrics.append(metrics)
        if progress and task_id:
            progress.update(task_id, advance=1)

    # Aggregate
    agg = aggregate_metrics(all_metrics)
    return {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}


def run_late_fusion_grid(
    base_retrievers: Dict,
    dataset: ESC50Dataset,
    folds: List[int],
    logger: logging.Logger,
) -> List[Dict]:
    """
    Run late fusion with weight grid search.

    Uses M1 (MFCC), M3 (LogMel), M4 (Spectral) with weights α, β, γ.
    """
    console.print("\n[bold cyan]Late Fusion Grid Search[/bold cyan]")

    # Select retrievers for fusion
    retriever_names = ['M1_MFCC', 'M3_LogMel', 'M4_Spectral']
    retrievers = [base_retrievers[name] for name in retriever_names]

    # Weight grid: step of 0.25, sum to 1
    weight_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    weight_combos = []
    for w1 in weight_values:
        for w2 in weight_values:
            w3 = 1.0 - w1 - w2
            if 0 <= w3 <= 1.0 and abs(w1 + w2 + w3 - 1.0) < 0.01:
                weight_combos.append((w1, w2, w3))

    # Remove duplicates and sort
    weight_combos = list(set(weight_combos))
    weight_combos.sort()

    console.print(f"  Retrievers: {retriever_names}")
    console.print(f"  Weight combinations: {len(weight_combos)}")

    results = []
    queries_per_fold = len(dataset) // 5

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total = len(weight_combos) * len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Late Fusion", total=total)

        for weights in weight_combos:
            fold_metrics = []

            for fold in folds:
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                # Create fusion retriever
                fusion_retriever = create_late_fusion(
                    retrievers=retrievers,
                    weights=list(weights),
                    device=retrievers[0].device,
                    sr=retrievers[0].sr,
                )

                # Evaluate
                metrics = evaluate_retriever(
                    fusion_retriever, query_samples, gallery_samples, progress, task
                )
                fold_metrics.append(metrics)

            # Average across folds
            mean_metrics = {}
            std_metrics = {}
            for metric in fold_metrics[0].keys():
                values = [m[metric] for m in fold_metrics]
                mean_metrics[metric] = float(np.mean(values))
                std_metrics[metric] = float(np.std(values))

            results.append({
                'weights': {name: w for name, w in zip(retriever_names, weights)},
                'mean': mean_metrics,
                'std': std_metrics,
            })

            logger.info(f"  Weights {weights}: hit@10={mean_metrics.get('hit@10', 0):.4f}")

    return results


def run_rank_fusion(
    base_retrievers: Dict,
    dataset: ESC50Dataset,
    folds: List[int],
    logger: logging.Logger,
) -> Dict:
    """Run rank fusion (RRF) experiment."""
    console.print("\n[bold cyan]Rank Fusion (RRF)[/bold cyan]")

    # Use M1, M2, M3 for rank fusion
    retriever_names = ['M1_MFCC', 'M2_MFCC_Delta', 'M3_LogMel']
    retrievers = [base_retrievers[name] for name in retriever_names]

    console.print(f"  Retrievers: {retriever_names}")

    fold_metrics = []
    queries_per_fold = len(dataset) // 5

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total = len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Rank Fusion", total=total)

        for fold in folds:
            query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

            # Create RRF retriever
            rrf_retriever = create_rank_fusion(
                retrievers=retrievers,
                rrf_k=60,
                device=retrievers[0].device,
                sr=retrievers[0].sr,
            )

            # Evaluate
            metrics = evaluate_retriever(
                rrf_retriever, query_samples, gallery_samples, progress, task
            )
            fold_metrics.append(metrics)

    # Average across folds
    mean_metrics = {}
    std_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [m[metric] for m in fold_metrics]
        mean_metrics[metric] = float(np.mean(values))
        std_metrics[metric] = float(np.std(values))

    logger.info(f"  RRF: hit@10={mean_metrics.get('hit@10', 0):.4f}")

    return {
        'retrievers': retriever_names,
        'rrf_k': 60,
        'mean': mean_metrics,
        'std': std_metrics,
    }


def display_late_fusion_results(results: List[Dict]):
    """Display late fusion results."""
    # Sort by hit@10
    results_sorted = sorted(results, key=lambda r: r['mean'].get('hit@10', 0), reverse=True)

    table = Table(title="Late Fusion Results (Top 10)", box=box.ROUNDED)
    table.add_column("Rank", style="bold")
    table.add_column("MFCC", justify="center")
    table.add_column("LogMel", justify="center")
    table.add_column("Spectral", justify="center")
    table.add_column("Hit@10", justify="center", style="green")
    table.add_column("mAP@20", justify="center")

    for i, r in enumerate(results_sorted[:10]):
        weights = r['weights']
        style = "bold green" if i == 0 else ""
        table.add_row(
            str(i + 1),
            f"{weights.get('M1_MFCC', 0):.2f}",
            f"{weights.get('M3_LogMel', 0):.2f}",
            f"{weights.get('M4_Spectral', 0):.2f}",
            f"{r['mean'].get('hit@10', 0):.4f}±{r['std'].get('hit@10', 0):.4f}",
            f"{r['mean'].get('map@20', 0):.4f}±{r['std'].get('map@20', 0):.4f}",
            style=style
        )

    console.print(table)


def display_comparison_table(
    base_results: Dict[str, Dict],
    late_fusion_best: Dict,
    rank_fusion: Dict,
):
    """Display comparison of individual methods vs fusion."""
    table = Table(title="Fusion vs Individual Methods", box=box.ROUNDED)
    table.add_column("Method", style="bold")
    table.add_column("Hit@10", justify="center")
    table.add_column("P@10", justify="center")
    table.add_column("mAP@20", justify="center")

    # Individual methods
    for name, result in base_results.items():
        table.add_row(
            name,
            f"{result['mean'].get('hit@10', 0):.4f}",
            f"{result['mean'].get('precision@10', 0):.4f}",
            f"{result['mean'].get('map@20', 0):.4f}",
        )

    # Divider
    table.add_row("─" * 10, "─" * 10, "─" * 10, "─" * 10)

    # Late fusion
    table.add_row(
        "Late Fusion (Best)",
        f"{late_fusion_best['mean'].get('hit@10', 0):.4f}",
        f"{late_fusion_best['mean'].get('precision@10', 0):.4f}",
        f"{late_fusion_best['mean'].get('map@20', 0):.4f}",
        style="bold cyan"
    )

    # Rank fusion
    table.add_row(
        "Rank Fusion (RRF)",
        f"{rank_fusion['mean'].get('hit@10', 0):.4f}",
        f"{rank_fusion['mean'].get('precision@10', 0):.4f}",
        f"{rank_fusion['mean'].get('map@20', 0):.4f}",
        style="bold magenta"
    )

    console.print(table)


def run_fusion_experiments(config_path: str, output_dir: Path):
    """Run all fusion experiments."""
    yaml_config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed = get_seed_from_config(yaml_config)
    if seed is not None:
        set_seed(seed, deterministic=bool(yaml_config.get('deterministic', False)))
        logger.info(f"Random seed set to {seed}")

    dataset_cfg = yaml_config.get('dataset', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)
    folds = yaml_config.get('evaluation', {}).get('folds', [1, 2, 3, 4, 5])

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    console.print(Panel.fit(
        "[bold blue]Multi-Feature Fusion Experiments[/bold blue]\n"
        f"Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("FUSION EXPERIMENTS STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Create base retrievers
    console.print("\n[bold]Creating base retrievers...[/bold]")
    base_retrievers = create_base_retrievers(yaml_config, device, sr)
    for name in base_retrievers:
        console.print(f"  [green]✓[/green] {name}")

    # Evaluate individual methods first (for comparison)
    console.print("\n[bold cyan]Evaluating Individual Methods[/bold cyan]")
    base_results = {}
    queries_per_fold = len(dataset) // 5

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total = len(base_retrievers) * len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Individual methods", total=total)

        for name, retriever in base_retrievers.items():
            fold_metrics = []
            for fold in folds:
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)
                metrics = evaluate_retriever(retriever, query_samples, gallery_samples, progress, task)
                fold_metrics.append(metrics)

            mean_metrics = {}
            for metric in fold_metrics[0].keys():
                values = [m[metric] for m in fold_metrics]
                mean_metrics[metric] = float(np.mean(values))

            base_results[name] = {'mean': mean_metrics}

    # Late fusion grid search
    late_fusion_results = run_late_fusion_grid(base_retrievers, dataset, folds, logger)
    display_late_fusion_results(late_fusion_results)

    # Find best late fusion
    late_fusion_best = max(late_fusion_results, key=lambda r: r['mean'].get('hit@10', 0))

    # Rank fusion
    rank_fusion_result = run_rank_fusion(base_retrievers, dataset, folds, logger)

    # Display comparison
    console.print("\n")
    display_comparison_table(base_results, late_fusion_best, rank_fusion_result)

    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'individual_methods': base_results,
        'late_fusion': {
            'all_weights': late_fusion_results,
            'best': late_fusion_best,
        },
        'rank_fusion': rank_fusion_result,
    }

    with open(output_dir / 'late_fusion.json', 'w') as f:
        json.dump({'results': late_fusion_results, 'best': late_fusion_best}, f, indent=2)

    with open(output_dir / 'rank_fusion.json', 'w') as f:
        json.dump(rank_fusion_result, f, indent=2)

    with open(output_dir / 'all_fusion.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Feature Fusion Experiments")
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
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'fusion' / timestamp

    try:
        run_fusion_experiments(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Fusion experiments completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
