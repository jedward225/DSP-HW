#!/usr/bin/env python3
"""
Deep Learning Retriever Evaluation Script (Multi-GPU Parallel Version)

This script evaluates trained deep learning retrievers (Autoencoder, CNN, Contrastive)
on the ESC-50 dataset with 5-fold cross-validation, using multiple GPUs in parallel.

Prerequisites:
    Run the training scripts first to generate per-fold model checkpoints:
    - python experiments/training/train_autoencoder.py --data_dir ESC-50 --folds 1,2,3,4,5
    - python experiments/training/train_cnn.py --data_dir ESC-50 --folds 1,2,3,4,5
    - python experiments/training/train_contrastive.py --data_dir ESC-50 --folds 1,2,3,4,5

Usage:
    python run_deep_retrievers.py [--models-dir MODELS_DIR] [--num-gpus 8]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.metrics.retrieval_metrics import aggregate_metrics
from src.metrics.bootstrap import aggregate_metrics_with_ci

# Initialize rich console
console = Console()


@dataclass
class EvalTask:
    """Represents a single evaluation task."""
    model_type: str  # 'Autoencoder', 'CNN', 'Contrastive'
    fold: int
    checkpoint_path: str
    gpu_id: int


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


def discover_fold_checkpoints(models_dir: Path, folds: List[int]) -> Dict[str, Dict[int, Path]]:
    """
    Discover all per-fold checkpoints for each model type.

    Returns:
        Dict mapping model_type -> {fold: checkpoint_path}
    """
    checkpoints = {
        'Autoencoder': {},
        'CNN': {},
        'Contrastive': {},
    }

    for fold in folds:
        # Autoencoder
        ae_path = models_dir / f'autoencoder_esc50_fold{fold}.pt'
        if ae_path.exists():
            checkpoints['Autoencoder'][fold] = ae_path

        # CNN
        cnn_path = models_dir / f'cnn_esc50_fold{fold}.pt'
        if cnn_path.exists():
            checkpoints['CNN'][fold] = cnn_path

        # Contrastive
        cont_path = models_dir / f'contrastive_esc50_fold{fold}.pt'
        if cont_path.exists():
            checkpoints['Contrastive'][fold] = cont_path

    return checkpoints


def evaluate_single_task(args: Tuple) -> Dict:
    """
    Worker function to evaluate a single (model_type, fold) on a specific GPU.

    This function runs in a separate process with its own CUDA context.
    """
    task_dict, dataset_root, sr = args

    # Reconstruct task from dict (dataclass not always picklable)
    model_type = task_dict['model_type']
    fold = task_dict['fold']
    checkpoint_path = task_dict['checkpoint_path']
    gpu_id = task_dict['gpu_id']

    device = f'cuda:{gpu_id}'

    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)

        # Import retrievers here to avoid issues with multiprocessing
        from src.retrieval.autoencoder_retriever import AutoencoderRetriever
        from src.retrieval.cnn_retriever import CNNRetriever
        from src.retrieval.contrastive_retriever import ContrastiveRetriever

        # Load dataset
        dataset = ESC50Dataset(
            root_dir=dataset_root,
            sr=sr,
            preload=False
        )
        query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

        # Create retriever based on model type
        if model_type == 'Autoencoder':
            retriever = AutoencoderRetriever(
                model_path=checkpoint_path,
                name='Deep_Autoencoder',
                device=device,
                sr=sr,
            )
        elif model_type == 'CNN':
            retriever = CNNRetriever(
                model_path=checkpoint_path,
                name='Deep_CNN',
                device=device,
                sr=sr,
            )
        elif model_type == 'Contrastive':
            retriever = ContrastiveRetriever(
                model_path=checkpoint_path,
                name='Deep_Contrastive',
                device=device,
                sr=sr,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Build gallery
        retriever.build_gallery(gallery_samples, show_progress=False)

        # Evaluate all queries
        all_metrics = []
        for query in query_samples:
            metrics = retriever.evaluate_query(query)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics)

        # Extract mean values
        result = {}
        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict) and 'mean' in stats:
                result[metric_name] = stats['mean']

        # Cleanup
        retriever.clear_gallery()
        del retriever
        torch.cuda.empty_cache()

        return {
            'model_type': model_type,
            'fold': fold,
            'gpu_id': gpu_id,
            'metrics': result,
            'status': 'success',
            'error': None,
        }

    except Exception as e:
        return {
            'model_type': model_type,
            'fold': fold,
            'gpu_id': gpu_id,
            'metrics': {},
            'status': 'error',
            'error': str(e),
        }


def run_parallel_evaluation(
    tasks: List[EvalTask],
    dataset_root: str,
    sr: int,
    num_workers: int,
) -> List[Dict]:
    """
    Run evaluation tasks in parallel across multiple GPUs.

    Args:
        tasks: List of evaluation tasks
        dataset_root: Path to ESC-50 dataset
        sr: Sample rate
        num_workers: Number of parallel workers (typically = num_gpus)

    Returns:
        List of result dictionaries
    """
    # Convert tasks to dicts for pickling
    task_args = [
        (
            {
                'model_type': t.model_type,
                'fold': t.fold,
                'checkpoint_path': t.checkpoint_path,
                'gpu_id': t.gpu_id,
            },
            dataset_root,
            sr,
        )
        for t in tasks
    ]

    # Use spawn to ensure clean CUDA contexts
    ctx = mp.get_context('spawn')

    results = []
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap for ordered results with progress tracking
        for result in pool.imap(evaluate_single_task, task_args):
            results.append(result)
            # Print progress
            status = "✓" if result['status'] == 'success' else "✗"
            if result['status'] == 'success':
                hit10 = result['metrics'].get('hit@10', 0)
                detail = f"Hit@10={hit10:.4f}"
            else:
                detail = result['error']
            console.print(
                f"  [{status}] {result['model_type']} fold {result['fold']} "
                f"(GPU {result['gpu_id']}): {detail}"
            )

    return results


def aggregate_results_by_method(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate per-fold results into per-method statistics.

    Args:
        results: List of per-task results

    Returns:
        Dict mapping method_name -> {mean, std, ci, folds}
    """
    # Group results by model type
    by_method = {}
    for r in results:
        if r['status'] != 'success':
            continue

        model_type = r['model_type']
        method_name = f"Deep_{model_type}"

        if method_name not in by_method:
            by_method[method_name] = {'folds': {}, 'fold_metrics': []}

        by_method[method_name]['folds'][str(r['fold'])] = r['metrics']
        by_method[method_name]['fold_metrics'].append(r['metrics'])

    # Compute statistics for each method
    method_results = {}
    for method_name, data in by_method.items():
        fold_metrics = data['fold_metrics']

        if not fold_metrics:
            continue

        # Use bootstrap CI aggregation
        aggregated_with_ci = aggregate_metrics_with_ci(fold_metrics)

        method_results[method_name] = {
            'mean': {m: float(s['mean']) for m, s in aggregated_with_ci.items()},
            'std': {m: float(s['std']) for m, s in aggregated_with_ci.items()},
            'ci': {
                m: {
                    'lower': float(s['ci_lower']),
                    'upper': float(s['ci_upper']),
                    'width': float(s['ci_width']),
                }
                for m, s in aggregated_with_ci.items()
            },
            'folds': data['folds'],
        }

    return method_results


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


def display_fold_details(results: Dict, console: Console):
    """Display per-fold results for each method."""
    for method_name, method_results in results['methods'].items():
        console.print(f"\n[bold]{method_name} per-fold Hit@10:[/bold]")
        folds = method_results.get('folds', {})
        for fold_num in sorted(folds.keys(), key=int):
            hit10 = folds[fold_num].get('hit@10', 0)
            console.print(f"  Fold {fold_num}: {hit10:.4f}")


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
        description='Evaluate trained deep learning retrievers on ESC-50 (Multi-GPU)'
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
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=8,
        help='Number of GPUs to use for parallel evaluation'
    )
    parser.add_argument(
        '--folds',
        type=str,
        default='1,2,3,4,5',
        help='Folds to evaluate (comma-separated)'
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

    # Parse settings
    models_dir = Path(args.models_dir)
    dataset_cfg = config.get('dataset', {})
    sr = dataset_cfg.get('sr', 22050)
    dataset_root = str(PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50'))
    folds = [int(f) for f in args.folds.split(',')]

    # Detect available GPUs
    num_gpus_available = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, num_gpus_available)

    if num_gpus == 0:
        console.print("[red]No GPUs available! This script requires CUDA.[/red]")
        return

    # Print header
    console.print(Panel.fit(
        "[bold blue]Deep Learning Retriever Evaluation (Multi-GPU)[/bold blue]\n"
        f"Dataset: ESC-50 | GPUs: {num_gpus} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("DEEP RETRIEVER EXPERIMENT STARTED (MULTI-GPU)")
    logger.info(f"GPUs available: {num_gpus_available}, using: {num_gpus}")
    logger.info(f"Models dir: {models_dir}")
    logger.info(f"Folds: {folds}")
    logger.info("=" * 60)

    # Discover all per-fold checkpoints
    console.print("\n[bold]Discovering per-fold checkpoints...[/bold]")
    checkpoints = discover_fold_checkpoints(models_dir, folds)

    # Display found checkpoints
    total_found = 0
    for model_type, fold_ckpts in checkpoints.items():
        if fold_ckpts:
            console.print(f"  [green]✓[/green] {model_type}: {len(fold_ckpts)} fold checkpoints")
            for fold, path in sorted(fold_ckpts.items()):
                console.print(f"      Fold {fold}: {path.name}")
                logger.info(f"Found {model_type} fold {fold}: {path}")
            total_found += len(fold_ckpts)
        else:
            console.print(f"  [red]✗[/red] {model_type}: no fold checkpoints found")
            logger.warning(f"No checkpoints found for {model_type}")

    if total_found == 0:
        console.print("\n[red]No model checkpoints found![/red]")
        console.print("Please run training scripts first with --folds 1,2,3,4,5:")
        console.print("  python experiments/training/train_autoencoder.py --data_dir ESC-50 --folds 1,2,3,4,5")
        console.print("  python experiments/training/train_cnn.py --data_dir ESC-50 --folds 1,2,3,4,5")
        console.print("  python experiments/training/train_contrastive.py --data_dir ESC-50 --folds 1,2,3,4,5")
        return

    # Create evaluation tasks with round-robin GPU assignment
    console.print(f"\n[bold]Creating evaluation tasks ({total_found} total)...[/bold]")
    tasks = []
    gpu_idx = 0

    for model_type, fold_ckpts in checkpoints.items():
        for fold, ckpt_path in sorted(fold_ckpts.items()):
            task = EvalTask(
                model_type=model_type,
                fold=fold,
                checkpoint_path=str(ckpt_path),
                gpu_id=gpu_idx % num_gpus,
            )
            tasks.append(task)
            gpu_idx += 1

    console.print(f"  Created {len(tasks)} tasks across {num_gpus} GPUs")
    logger.info(f"Created {len(tasks)} evaluation tasks")

    # Show task distribution
    gpu_tasks = {}
    for t in tasks:
        if t.gpu_id not in gpu_tasks:
            gpu_tasks[t.gpu_id] = []
        gpu_tasks[t.gpu_id].append(f"{t.model_type[:4]}-F{t.fold}")

    for gpu_id in sorted(gpu_tasks.keys()):
        console.print(f"    GPU {gpu_id}: {', '.join(gpu_tasks[gpu_id])}")

    # Run parallel evaluation
    console.print(f"\n[bold]Running parallel evaluation on {num_gpus} GPUs...[/bold]")
    logger.info("Starting parallel evaluation")

    start_time = datetime.now()
    raw_results = run_parallel_evaluation(tasks, dataset_root, sr, num_gpus)
    end_time = datetime.now()

    elapsed = (end_time - start_time).total_seconds()
    console.print(f"\n[green]✓[/green] Evaluation completed in {elapsed:.1f} seconds")
    logger.info(f"Evaluation completed in {elapsed:.1f} seconds")

    # Check for errors
    errors = [r for r in raw_results if r['status'] == 'error']
    if errors:
        console.print(f"\n[yellow]Warning: {len(errors)} tasks failed:[/yellow]")
        for e in errors:
            console.print(f"  {e['model_type']} fold {e['fold']}: {e['error']}")
            logger.error(f"{e['model_type']} fold {e['fold']} failed: {e['error']}")

    # Aggregate results by method
    method_results = aggregate_results_by_method(raw_results)

    if not method_results:
        console.print("[red]No successful evaluations![/red]")
        return

    # Build final results structure
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'models_dir': str(models_dir),
        'num_gpus': num_gpus,
        'elapsed_seconds': elapsed,
        'methods': method_results,
    }

    # Display results
    console.print("\n")
    display_results_table(results, console)
    display_fold_details(results, console)

    # Log results
    for method_name, data in method_results.items():
        logger.info(f"Results for {method_name}:")
        for metric, value in data['mean'].items():
            std = data['std'].get(metric, 0)
            logger.info(f"  {metric}: {value:.4f} ± {std:.4f}")

    # Save results
    save_results(results, output_dir, logger)

    console.print(f"\n[bold green]Experiment completed![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    # Required for multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)
    main()
