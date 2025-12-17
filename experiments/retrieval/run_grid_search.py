#!/usr/bin/env python3
"""
Hyperparameter Grid Search for Sound Retrieval

This script implements the 3-step efficient search strategy from proposal Section 4:
  Step 1: Frame length × Hop length sweep (16 configs, single fold)
  Step 2: n_mels × n_mfcc sweep on top-3 configs (27 configs)
  Step 3: Window function comparison with full 5-fold CV

Usage:
    python run_grid_search.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
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
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.retrieval import create_method_m1, create_method_m2, create_method_m3
from src.metrics.retrieval_metrics import aggregate_metrics

console = Console()


@dataclass
class GridSearchConfig:
    """Configuration for a single grid search run."""
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    window: str
    # For display
    frame_length_ms: float = 0.0
    hop_length_ms: float = 0.0
    # Frequency range
    fmin: float = 0.0
    fmax: float = None  # None = sr/2

    def to_dict(self) -> Dict:
        return asdict(self)

    def __hash__(self):
        return hash((self.n_fft, self.hop_length, self.n_mels, self.n_mfcc, self.window, self.fmin, self.fmax))


@dataclass
class GridSearchResult:
    """Result from a single configuration evaluation."""
    config: GridSearchConfig
    metrics: Dict[str, float]
    fold_results: Dict[int, Dict[str, float]] = field(default_factory=dict)


def ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to samples."""
    return int(ms * sr / 1000)


def samples_to_ms(samples: int, sr: int) -> float:
    """Convert samples to milliseconds."""
    return samples * 1000 / sr


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file."""
    log_file = output_dir / 'grid_search.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
        ]
    )
    return logging.getLogger('grid_search')


def create_retriever(config: GridSearchConfig, device: str, sr: int):
    """Create a retriever with specified configuration."""
    return create_method_m1(
        device=device,
        sr=sr,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )


def evaluate_config(
    config: GridSearchConfig,
    dataset: ESC50Dataset,
    folds: List[int],
    device: str,
    sr: int,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> GridSearchResult:
    """Evaluate a single configuration on specified folds."""
    fold_results = {}
    all_metrics = []

    for fold in folds:
        # Create fresh retriever
        retriever = create_retriever(config, device, sr)

        # Get query/gallery split
        query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

        # Build gallery
        retriever.build_gallery(gallery_samples, show_progress=False)

        # Evaluate queries
        query_metrics = []
        for query in query_samples:
            metrics = retriever.evaluate_query(query)
            query_metrics.append(metrics)
            if progress and task_id:
                progress.update(task_id, advance=1)

        # Aggregate fold metrics
        fold_agg = aggregate_metrics(query_metrics)
        fold_result = {k: v['mean'] for k, v in fold_agg.items() if isinstance(v, dict) and 'mean' in v}
        fold_results[fold] = fold_result
        all_metrics.append(fold_result)

    # Compute mean across folds
    mean_metrics = {}
    if all_metrics:
        metric_names = all_metrics[0].keys()
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            mean_metrics[metric] = float(np.mean(values))

    return GridSearchResult(config=config, metrics=mean_metrics, fold_results=fold_results)


def run_step1(
    yaml_config: Dict,
    dataset: ESC50Dataset,
    device: str,
    sr: int,
    logger: logging.Logger,
    output_dir: Path,
) -> List[Tuple[GridSearchConfig, float]]:
    """
    Step 1: Frame length × Hop length sweep.

    Returns top-k configs sorted by primary metric.
    """
    step1_cfg = yaml_config['step1']
    eval_cfg = yaml_config['evaluation']
    primary_metric = eval_cfg['primary_metric']

    frame_lengths_ms = step1_cfg['frame_lengths_ms']
    hop_lengths_ms = step1_cfg['hop_lengths_ms']
    fixed = step1_cfg['fixed']
    folds = step1_cfg['folds']
    top_k = step1_cfg['top_k']

    # Convert to samples
    configs = []
    for frame_ms in frame_lengths_ms:
        for hop_ms in hop_lengths_ms:
            # Skip invalid combinations (hop > frame)
            if hop_ms > frame_ms:
                continue
            n_fft = ms_to_samples(frame_ms, sr)
            hop_length = ms_to_samples(hop_ms, sr)
            config = GridSearchConfig(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=fixed['n_mels'],
                n_mfcc=fixed['n_mfcc'],
                window=fixed['window'],
                frame_length_ms=frame_ms,
                hop_length_ms=hop_ms,
            )
            configs.append(config)

    console.print(f"\n[bold cyan]Step 1: Frame/Hop Sweep ({len(configs)} configs)[/bold cyan]")
    console.print(f"  Frame lengths: {frame_lengths_ms} ms")
    console.print(f"  Hop lengths: {hop_lengths_ms} ms")
    console.print(f"  Fixed: n_mels={fixed['n_mels']}, n_mfcc={fixed['n_mfcc']}, window={fixed['window']}")
    console.print(f"  Folds: {folds}")

    logger.info(f"Step 1: {len(configs)} configs, folds={folds}")

    results = []
    queries_per_config = len(dataset) // 5 * len(folds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Step 1 Progress", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_config(config, dataset, folds, device, sr, progress, overall)
            results.append(result)
            logger.info(f"  Config n_fft={config.n_fft}, hop={config.hop_length}: {primary_metric}={result.metrics.get(primary_metric, 0):.4f}")

    # Sort by primary metric (descending)
    results.sort(key=lambda r: r.metrics.get(primary_metric, 0), reverse=True)

    # Display results table
    table = Table(title="Step 1 Results: Frame/Hop Sweep", box=box.ROUNDED)
    table.add_column("Rank", style="bold")
    table.add_column("Frame (ms)", justify="center")
    table.add_column("Hop (ms)", justify="center")
    table.add_column("n_fft", justify="center")
    table.add_column("hop_len", justify="center")
    table.add_column(primary_metric, justify="center", style="green")

    for i, result in enumerate(results[:10]):  # Show top 10
        cfg = result.config
        style = "bold green" if i < top_k else ""
        table.add_row(
            str(i + 1),
            f"{cfg.frame_length_ms:.0f}",
            f"{cfg.hop_length_ms:.0f}",
            str(cfg.n_fft),
            str(cfg.hop_length),
            f"{result.metrics.get(primary_metric, 0):.4f}",
            style=style
        )

    console.print(table)

    # Save step 1 results
    step1_results = {
        'configs': [r.config.to_dict() for r in results],
        'metrics': [r.metrics for r in results],
        'primary_metric': primary_metric,
    }
    with open(output_dir / 'step1_frame_hop.json', 'w') as f:
        json.dump(step1_results, f, indent=2)

    # Return top-k configs
    top_configs = [(r.config, r.metrics.get(primary_metric, 0)) for r in results[:top_k]]
    return top_configs


def run_step2(
    yaml_config: Dict,
    top_configs: List[Tuple[GridSearchConfig, float]],
    dataset: ESC50Dataset,
    device: str,
    sr: int,
    logger: logging.Logger,
    output_dir: Path,
) -> Tuple[GridSearchConfig, float]:
    """
    Step 2: n_mels × n_mfcc sweep on top configs from Step 1.

    Returns best overall config.
    """
    step2_cfg = yaml_config['step2']
    eval_cfg = yaml_config['evaluation']
    primary_metric = eval_cfg['primary_metric']

    n_mels_list = step2_cfg['n_mels']
    n_mfcc_list = step2_cfg['n_mfcc']
    folds = step2_cfg['folds']
    top_k = step2_cfg['top_k']

    # Generate configs: each top config × n_mels × n_mfcc
    configs = []
    for base_config, _ in top_configs:
        for n_mels in n_mels_list:
            for n_mfcc in n_mfcc_list:
                config = GridSearchConfig(
                    n_fft=base_config.n_fft,
                    hop_length=base_config.hop_length,
                    n_mels=n_mels,
                    n_mfcc=n_mfcc,
                    window=base_config.window,
                    frame_length_ms=base_config.frame_length_ms,
                    hop_length_ms=base_config.hop_length_ms,
                )
                configs.append(config)

    console.print(f"\n[bold cyan]Step 2: n_mels × n_mfcc Sweep ({len(configs)} configs)[/bold cyan]")
    console.print(f"  n_mels: {n_mels_list}")
    console.print(f"  n_mfcc: {n_mfcc_list}")
    console.print(f"  Base configs: {len(top_configs)}")

    logger.info(f"Step 2: {len(configs)} configs")

    results = []
    queries_per_config = len(dataset) // 5 * len(folds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Step 2 Progress", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_config(config, dataset, folds, device, sr, progress, overall)
            results.append(result)
            logger.info(f"  Config n_mels={config.n_mels}, n_mfcc={config.n_mfcc}: {primary_metric}={result.metrics.get(primary_metric, 0):.4f}")

    # Sort by primary metric
    results.sort(key=lambda r: r.metrics.get(primary_metric, 0), reverse=True)

    # Display results table
    table = Table(title="Step 2 Results: MFCC Parameter Sweep", box=box.ROUNDED)
    table.add_column("Rank", style="bold")
    table.add_column("Frame (ms)", justify="center")
    table.add_column("Hop (ms)", justify="center")
    table.add_column("n_mels", justify="center")
    table.add_column("n_mfcc", justify="center")
    table.add_column(primary_metric, justify="center", style="green")

    for i, result in enumerate(results[:10]):
        cfg = result.config
        style = "bold green" if i < top_k else ""
        table.add_row(
            str(i + 1),
            f"{cfg.frame_length_ms:.0f}",
            f"{cfg.hop_length_ms:.0f}",
            str(cfg.n_mels),
            str(cfg.n_mfcc),
            f"{result.metrics.get(primary_metric, 0):.4f}",
            style=style
        )

    console.print(table)

    # Save step 2 results
    step2_results = {
        'configs': [r.config.to_dict() for r in results],
        'metrics': [r.metrics for r in results],
        'primary_metric': primary_metric,
    }
    with open(output_dir / 'step2_mfcc_params.json', 'w') as f:
        json.dump(step2_results, f, indent=2)

    # Return best config
    best_result = results[0]
    return best_result.config, best_result.metrics.get(primary_metric, 0)


def run_step3(
    yaml_config: Dict,
    best_config: GridSearchConfig,
    dataset: ESC50Dataset,
    device: str,
    sr: int,
    logger: logging.Logger,
    output_dir: Path,
) -> GridSearchResult:
    """
    Step 3: Window function comparison with full 5-fold CV.

    Returns final best config with full metrics.
    """
    step3_cfg = yaml_config['step3']
    eval_cfg = yaml_config['evaluation']
    primary_metric = eval_cfg['primary_metric']

    windows = step3_cfg['windows']
    folds = step3_cfg['folds']

    # Generate configs for each window
    configs = []
    for window in windows:
        config = GridSearchConfig(
            n_fft=best_config.n_fft,
            hop_length=best_config.hop_length,
            n_mels=best_config.n_mels,
            n_mfcc=best_config.n_mfcc,
            window=window,
            frame_length_ms=best_config.frame_length_ms,
            hop_length_ms=best_config.hop_length_ms,
        )
        configs.append(config)

    console.print(f"\n[bold cyan]Step 3: Window Function Comparison (Full 5-fold CV)[/bold cyan]")
    console.print(f"  Windows: {windows}")
    console.print(f"  Best config: n_fft={best_config.n_fft}, hop={best_config.hop_length}, n_mels={best_config.n_mels}, n_mfcc={best_config.n_mfcc}")

    logger.info(f"Step 3: {len(configs)} window functions, full 5-fold CV")

    results = []
    queries_per_config = len(dataset) // 5 * len(folds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Step 3 Progress", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_config(config, dataset, folds, device, sr, progress, overall)
            results.append(result)
            logger.info(f"  Window={config.window}: {primary_metric}={result.metrics.get(primary_metric, 0):.4f}")

    # Sort by primary metric
    results.sort(key=lambda r: r.metrics.get(primary_metric, 0), reverse=True)

    # Display results table with all metrics
    table = Table(title="Step 3 Results: Window Function (5-fold CV)", box=box.ROUNDED)
    table.add_column("Window", style="bold")
    for metric in eval_cfg['metrics']:
        table.add_column(metric, justify="center")

    for i, result in enumerate(results):
        cfg = result.config
        style = "bold green" if i == 0 else ""
        row = [cfg.window]
        for metric in eval_cfg['metrics']:
            row.append(f"{result.metrics.get(metric, 0):.4f}")
        table.add_row(*row, style=style)

    console.print(table)

    # Save step 3 results
    step3_results = {
        'configs': [r.config.to_dict() for r in results],
        'metrics': [r.metrics for r in results],
        'fold_results': {r.config.window: r.fold_results for r in results},
        'primary_metric': primary_metric,
    }
    with open(output_dir / 'step3_window.json', 'w') as f:
        json.dump(step3_results, f, indent=2)

    return results[0]


def run_step4(
    yaml_config: Dict,
    best_config: GridSearchConfig,
    dataset: ESC50Dataset,
    device: str,
    sr: int,
    logger: logging.Logger,
    output_dir: Path,
) -> GridSearchResult:
    """
    Step 4: Frequency range (fmin/fmax) sweep.

    Tests different frequency range combinations.
    Returns best config with optimal frequency range.
    """
    step4_cfg = yaml_config.get('step4', {})
    eval_cfg = yaml_config['evaluation']
    primary_metric = eval_cfg['primary_metric']

    if not step4_cfg.get('enabled', False):
        console.print("\n[yellow]Step 4 (fmin/fmax sweep) is disabled. Skipping.[/yellow]")
        return None

    fmin_fmax_list = step4_cfg.get('fmin_fmax', [[0, None]])
    folds = step4_cfg.get('folds', [5])

    # Generate configs for each frequency range
    configs = []
    for fmin, fmax in fmin_fmax_list:
        config = GridSearchConfig(
            n_fft=best_config.n_fft,
            hop_length=best_config.hop_length,
            n_mels=best_config.n_mels,
            n_mfcc=best_config.n_mfcc,
            window=best_config.window,
            frame_length_ms=best_config.frame_length_ms,
            hop_length_ms=best_config.hop_length_ms,
            fmin=fmin if fmin is not None else 0.0,
            fmax=fmax,  # None = sr/2
        )
        configs.append(config)

    console.print(f"\n[bold cyan]Step 4: Frequency Range Sweep ({len(configs)} configs)[/bold cyan]")
    console.print(f"  fmin/fmax combinations: {fmin_fmax_list}")
    console.print(f"  Best config: n_fft={best_config.n_fft}, hop={best_config.hop_length}, window={best_config.window}")

    logger.info(f"Step 4: {len(configs)} frequency range configs")

    results = []
    queries_per_config = len(dataset) // 5 * len(folds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Step 4 Progress", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_config(config, dataset, folds, device, sr, progress, overall)
            results.append(result)
            fmax_str = str(config.fmax) if config.fmax else "sr/2"
            logger.info(f"  fmin={config.fmin}, fmax={fmax_str}: {primary_metric}={result.metrics.get(primary_metric, 0):.4f}")

    # Sort by primary metric
    results.sort(key=lambda r: r.metrics.get(primary_metric, 0), reverse=True)

    # Display results table
    table = Table(title="Step 4 Results: Frequency Range Sweep", box=box.ROUNDED)
    table.add_column("Rank", style="bold")
    table.add_column("fmin (Hz)", justify="center")
    table.add_column("fmax (Hz)", justify="center")
    table.add_column(primary_metric, justify="center", style="green")

    for i, result in enumerate(results):
        cfg = result.config
        style = "bold green" if i == 0 else ""
        fmax_str = str(int(cfg.fmax)) if cfg.fmax else "sr/2"
        table.add_row(
            str(i + 1),
            str(int(cfg.fmin)),
            fmax_str,
            f"{result.metrics.get(primary_metric, 0):.4f}",
            style=style
        )

    console.print(table)

    # Save step 4 results
    step4_results = {
        'configs': [r.config.to_dict() for r in results],
        'metrics': [r.metrics for r in results],
        'primary_metric': primary_metric,
    }
    with open(output_dir / 'step4_freq_range.json', 'w') as f:
        json.dump(step4_results, f, indent=2)

    return results[0]


def save_best_config(result: GridSearchResult, output_dir: Path):
    """Save the best configuration to YAML."""
    config = result.config
    best_config = {
        'best_config': {
            'n_fft': config.n_fft,
            'hop_length': config.hop_length,
            'n_mels': config.n_mels,
            'n_mfcc': config.n_mfcc,
            'window': config.window,
            'frame_length_ms': config.frame_length_ms,
            'hop_length_ms': config.hop_length_ms,
            'fmin': config.fmin,
            'fmax': config.fmax,
        },
        'metrics': result.metrics,
        'fold_results': {str(k): v for k, v in result.fold_results.items()},
    }
    with open(output_dir / 'best_config.yaml', 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)


def run_grid_search(config_path: str, output_dir: Path):
    """Run the complete 3-step grid search."""
    # Load config
    yaml_config = load_config(config_path)

    # Setup output
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    # Get settings
    dataset_cfg = yaml_config.get('dataset', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)

    # Check CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    # Print header
    console.print(Panel.fit(
        "[bold blue]Hyperparameter Grid Search[/bold blue]\n"
        f"Strategy: 3-step efficient search | Device: {device}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("GRID SEARCH STARTED")
    logger.info(f"Device: {device}, Sample rate: {sr}")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(
        root_dir=str(dataset_root),
        sr=sr,
        preload=dataset_cfg.get('preload', False)
    )
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Step 1: Frame/Hop sweep
    top_configs = run_step1(yaml_config, dataset, device, sr, logger, output_dir)

    # Step 2: n_mels/n_mfcc sweep
    best_config, best_score = run_step2(yaml_config, top_configs, dataset, device, sr, logger, output_dir)

    # Step 3: Window function with full CV
    final_result = run_step3(yaml_config, best_config, dataset, device, sr, logger, output_dir)

    # Step 4: Frequency range sweep (optional)
    step4_result = run_step4(yaml_config, final_result.config, dataset, device, sr, logger, output_dir)
    if step4_result is not None:
        final_result = step4_result

    # Save best config
    save_best_config(final_result, output_dir)

    # Print final summary
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Best Configuration Found[/bold green]\n\n"
        f"Frame length: {final_result.config.frame_length_ms:.0f} ms (n_fft={final_result.config.n_fft})\n"
        f"Hop length: {final_result.config.hop_length_ms:.0f} ms (hop_length={final_result.config.hop_length})\n"
        f"n_mels: {final_result.config.n_mels}\n"
        f"n_mfcc: {final_result.config.n_mfcc}\n"
        f"Window: {final_result.config.window}\n\n"
        f"Hit@10: {final_result.metrics.get('hit@10', 0):.4f}\n"
        f"mAP@20: {final_result.metrics.get('map@20', 0):.4f}",
        border_style="green"
    ))

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'retrieval' / 'configs' / 'grid_search.yaml'),
        help='Path to grid search configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    args = parser.parse_args()

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'grid_search' / timestamp

    try:
        run_grid_search(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Grid search completed successfully![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
