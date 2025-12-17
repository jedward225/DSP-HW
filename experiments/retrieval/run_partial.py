#!/usr/bin/env python3
"""
Partial Query Retrieval Experiments (加分项)

This script tests retrieval with partial (short) query clips:
  - Query durations: 0.5s, 1s, 2s, 3s (vs full 5s)
  - Protocol: Query uses cropped audio, gallery uses full audio
  - Matching: Sliding window over gallery with min aggregation

Usage:
    python run_partial.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
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
from src.retrieval import create_method_m1, create_partial_retriever
from src.metrics.retrieval_metrics import aggregate_metrics

console = Console()


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'partial.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('partial')


def evaluate_partial_retriever(
    retriever,
    query_samples: List[dict],
    gallery_samples: List[dict],
    crop_mode: str = 'center',
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate partial query retriever.

    Args:
        retriever: PartialQueryRetriever instance
        query_samples: Query samples (will be cropped)
        gallery_samples: Gallery samples (full length)
        crop_mode: How to crop queries ('center', 'start', 'random')
        progress: Progress bar
        task_id: Task ID for progress update

    Returns:
        Aggregated metrics
    """
    # Build gallery with sliding window features
    retriever.build_gallery(gallery_samples, show_progress=False)

    # Evaluate queries
    all_metrics = []

    for query in query_samples:
        waveform = query['waveform']
        query_label = query['target']

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        waveform_tensor = torch.from_numpy(waveform).float()

        # Retrieve using partial matching
        indices, labels = retriever.retrieve_with_labels(waveform_tensor)

        # Compute metrics
        from src.metrics.retrieval_metrics import compute_all_metrics
        num_relevant = (retriever._gallery_labels == query_label).sum().item()
        metrics = compute_all_metrics(labels, query_label, num_relevant)
        all_metrics.append(metrics)

        if progress and task_id:
            progress.update(task_id, advance=1)

    # Aggregate
    agg = aggregate_metrics(all_metrics)
    return {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}


def run_partial_experiments(config_path: str, output_dir: Path):
    """Run partial query retrieval experiments."""
    yaml_config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    dataset_cfg = yaml_config.get('dataset', {})
    feat_cfg = yaml_config.get('features', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)
    folds = yaml_config.get('evaluation', {}).get('folds', [1, 2, 3, 4, 5])

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    console.print(Panel.fit(
        "[bold blue]Partial Query Retrieval Experiments (加分项)[/bold blue]\n"
        f"Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("PARTIAL QUERY RETRIEVAL EXPERIMENTS STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Query durations to test (in seconds)
    # ESC-50 audio is 5 seconds, so we test various shorter durations
    query_durations = [0.5, 1.0, 2.0, 3.0, 5.0]  # 5.0 is baseline (full query)
    stride_ratio = 0.5  # stride = duration * 0.5

    console.print(f"\n[bold]Testing query durations: {query_durations} seconds[/bold]")

    results = {}
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
        total = len(query_durations) * len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Partial Query Experiments", total=total)

        for duration in query_durations:
            progress.update(task, description=f"[cyan]Query duration: {duration}s")

            fold_metrics = []

            for fold in folds:
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                # Create base retriever
                base_retriever = create_method_m1(
                    device=device,
                    sr=sr,
                    n_mfcc=feat_cfg.get('n_mfcc', 20),
                    n_mels=feat_cfg.get('n_mels', 128),
                    n_fft=feat_cfg.get('n_fft', 2048),
                    hop_length=feat_cfg.get('hop_length', 512),
                )

                if duration >= 5.0:
                    # Full query - use base retriever directly
                    base_retriever.build_gallery(gallery_samples, show_progress=False)
                    all_metrics = []
                    for query in query_samples:
                        metrics = base_retriever.evaluate_query(query)
                        all_metrics.append(metrics)
                        progress.update(task, advance=1)
                    agg = aggregate_metrics(all_metrics)
                    metrics = {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}
                else:
                    # Partial query - use partial retriever
                    stride = duration * stride_ratio
                    partial_retriever = create_partial_retriever(
                        base_retriever=base_retriever,
                        query_duration_s=duration,
                        stride_s=stride,
                        aggregation='min',
                    )
                    metrics = evaluate_partial_retriever(
                        partial_retriever, query_samples, gallery_samples,
                        crop_mode='center', progress=progress, task_id=task
                    )

                fold_metrics.append(metrics)

            # Aggregate across folds
            results[f'{duration}s'] = {
                'duration_s': duration,
                'mean': {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()},
                'std': {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()},
            }

            logger.info(f"Duration {duration}s: hit@10={results[f'{duration}s']['mean'].get('hit@10', 0):.4f}")

    # Display results
    console.print("\n")
    table = Table(title="Partial Query Retrieval Results", box=box.ROUNDED)
    table.add_column("Query Duration", style="bold")
    table.add_column("Hit@10", justify="center")
    table.add_column("P@10", justify="center")
    table.add_column("mAP@20", justify="center")
    table.add_column("Relative to Full", justify="center")

    # Get full query baseline
    full_hit10 = results['5.0s']['mean'].get('hit@10', 1.0)

    for duration in query_durations:
        key = f'{duration}s'
        r = results[key]
        hit10 = r['mean'].get('hit@10', 0)

        relative = hit10 / full_hit10 * 100 if full_hit10 > 0 else 0

        # Color based on relative performance
        if duration >= 5.0:
            style = "bold green"
            rel_str = "100% (baseline)"
        elif relative >= 95:
            style = "green"
            rel_str = f"{relative:.1f}%"
        elif relative >= 85:
            style = "yellow"
            rel_str = f"{relative:.1f}%"
        else:
            style = "red"
            rel_str = f"{relative:.1f}%"

        table.add_row(
            f"{duration}s",
            f"{hit10:.4f}±{r['std'].get('hit@10', 0):.4f}",
            f"{r['mean'].get('precision@10', 0):.4f}",
            f"{r['mean'].get('map@20', 0):.4f}",
            rel_str,
            style=style
        )

    console.print(table)

    # Save results
    with open(output_dir / 'partial_query.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary = {
        'query_durations': query_durations,
        'baseline_hit10': full_hit10,
        'duration_vs_accuracy': {
            f'{d}s': results[f'{d}s']['mean'].get('hit@10', 0) for d in query_durations
        }
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Partial Query Retrieval Experiments")
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
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'partial' / timestamp

    try:
        run_partial_experiments(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Partial query experiments completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
