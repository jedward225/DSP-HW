#!/usr/bin/env python3
"""
Efficiency Analysis for Sound Retrieval Methods

This script measures:
  1. Feature extraction time (ms per audio)
  2. Retrieval time (ms per query)
  3. Memory usage (MB)
  4. Throughput (queries per second)

Usage:
    python run_efficiency.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
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
    create_method_m5,
    create_method_m6,
    create_method_m7,
)

console = Console()


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for a single method."""
    method_name: str
    # Feature extraction
    feature_extract_time_ms: float  # Per audio
    feature_extract_std_ms: float
    # Gallery building
    gallery_build_time_s: float  # Total time to build gallery
    # Retrieval
    retrieval_time_ms: float  # Per query
    retrieval_std_ms: float
    # Memory
    feature_dim: int  # Dimension of feature vector
    gallery_memory_mb: float  # Memory for gallery features
    # Throughput
    throughput_qps: float  # Queries per second

    def to_dict(self) -> Dict:
        return asdict(self)


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Calculate memory usage of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)


def measure_feature_extraction(
    method,
    samples: List[dict],
    num_samples: int = 100,
) -> Tuple[float, float]:
    """Measure feature extraction time."""
    times = []

    # Warm up
    for sample in samples[:5]:
        waveform = sample['waveform']
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()
        _ = method.extract_features(waveform.to(method.device))

    # Measure
    for sample in samples[:num_samples]:
        waveform = sample['waveform']
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()

        if method.device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = method.extract_features(waveform.to(method.device))

        if method.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


def measure_gallery_build(
    method,
    gallery_samples: List[dict],
) -> float:
    """Measure gallery building time."""
    method.clear_gallery()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = time.perf_counter()
    method.build_gallery(gallery_samples, show_progress=False)
    if method.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed


def measure_retrieval(
    method,
    query_samples: List[dict],
    num_queries: int = 100,
    k: int = 10,
) -> Tuple[float, float]:
    """Measure retrieval time per query."""
    times = []

    # Warm up
    for sample in query_samples[:5]:
        waveform = sample['waveform']
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()
        _ = method.retrieve(waveform, k=k)

    # Measure
    for sample in query_samples[:num_queries]:
        waveform = sample['waveform']
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()

        if method.device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = method.retrieve(waveform, k=k)

        if method.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


def measure_method(
    method,
    query_samples: List[dict],
    gallery_samples: List[dict],
    num_samples: int = 100,
) -> EfficiencyMetrics:
    """Measure all efficiency metrics for a method."""
    # Feature extraction time
    feat_time_ms, feat_std = measure_feature_extraction(method, gallery_samples, num_samples)

    # Gallery build time
    build_time = measure_gallery_build(method, gallery_samples)

    # Get feature dimension and memory
    if method._gallery_features is not None:
        feature_dim = method._gallery_features.shape[1]
        gallery_memory = get_tensor_memory_mb(method._gallery_features)
    else:
        feature_dim = 0
        gallery_memory = 0.0

    # Retrieval time
    ret_time_ms, ret_std = measure_retrieval(method, query_samples, num_samples)

    # Throughput
    throughput = 1000.0 / ret_time_ms if ret_time_ms > 0 else 0

    return EfficiencyMetrics(
        method_name=method.name,
        feature_extract_time_ms=feat_time_ms,
        feature_extract_std_ms=feat_std,
        gallery_build_time_s=build_time,
        retrieval_time_ms=ret_time_ms,
        retrieval_std_ms=ret_std,
        feature_dim=feature_dim,
        gallery_memory_mb=gallery_memory,
        throughput_qps=throughput,
    )


def create_methods(config: Dict, device: str, sr: int) -> Dict:
    """Create all retrieval methods."""
    methods = {}
    feat_cfg = config.get('features', {})

    common_params = {
        'device': device,
        'sr': sr,
        'n_mfcc': feat_cfg.get('n_mfcc', 20),
        'n_mels': feat_cfg.get('n_mels', 128),
        'n_fft': feat_cfg.get('n_fft', 2048),
        'hop_length': feat_cfg.get('hop_length', 512),
    }

    methods['M1_MFCC_Pool_Cos'] = create_method_m1(**common_params)
    methods['M2_MFCC_Delta_Pool'] = create_method_m2(**common_params, delta_width=9)
    methods['M3_LogMel_Pool'] = create_method_m3(**common_params)
    methods['M4_Spectral_Stat'] = create_method_m4(**common_params)

    # M5 uses CPU due to Numba
    dtw_cfg = config.get('dtw', {})
    methods['M5_MFCC_DTW'] = create_method_m5(
        device='cpu',
        sr=sr,
        n_mfcc=dtw_cfg.get('n_mfcc', 13),
        n_mels=dtw_cfg.get('n_mels', 64),
        n_fft=feat_cfg.get('n_fft', 2048),
        hop_length=feat_cfg.get('hop_length', 512),
    )

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
    )

    return methods


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'efficiency.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('efficiency')


def display_results_table(results: List[EfficiencyMetrics]):
    """Display efficiency results."""
    table = Table(title="Efficiency Analysis Results", box=box.ROUNDED)
    table.add_column("Method", style="bold")
    table.add_column("Feat Ext (ms)", justify="right")
    table.add_column("Retrieval (ms)", justify="right")
    table.add_column("Throughput (q/s)", justify="right")
    table.add_column("Feat Dim", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for r in results:
        table.add_row(
            r.method_name,
            f"{r.feature_extract_time_ms:.2f}±{r.feature_extract_std_ms:.2f}",
            f"{r.retrieval_time_ms:.2f}±{r.retrieval_std_ms:.2f}",
            f"{r.throughput_qps:.1f}",
            str(r.feature_dim),
            f"{r.gallery_memory_mb:.2f}",
        )

    console.print(table)


def run_efficiency_analysis(config_path: str, output_dir: Path):
    """Run efficiency analysis."""
    yaml_config = load_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    dataset_cfg = yaml_config.get('dataset', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    console.print(Panel.fit(
        "[bold blue]Efficiency Analysis[/bold blue]\n"
        f"Device: {device}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("EFFICIENCY ANALYSIS STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Get fold 5 for testing
    query_samples, gallery_samples = dataset.get_query_gallery_split(5)

    # Create methods
    console.print("\n[bold]Initializing methods...[/bold]")
    methods = create_methods(yaml_config, device, sr)
    for name in methods:
        console.print(f"  [green]✓[/green] {name}")

    # Measure each method
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Measuring efficiency", total=len(methods))

        for name, method in methods.items():
            progress.update(task, description=f"[cyan]Measuring {name}")

            metrics = measure_method(method, query_samples, gallery_samples, num_samples=50)
            results.append(metrics)

            logger.info(f"{name}: feat={metrics.feature_extract_time_ms:.2f}ms, ret={metrics.retrieval_time_ms:.2f}ms")

            progress.update(task, advance=1)

    # Display results
    console.print("\n")
    display_results_table(results)

    # Save results
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'gallery_size': len(gallery_samples),
        'query_size': len(query_samples),
        'methods': [r.to_dict() for r in results],
    }

    with open(output_dir / 'timing.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Generate Pareto data (for plotting accuracy vs latency later)
    pareto_data = {
        r.method_name: {
            'retrieval_time_ms': r.retrieval_time_ms,
            'throughput_qps': r.throughput_qps,
            'memory_mb': r.gallery_memory_mb,
        }
        for r in results
    }

    with open(output_dir / 'pareto.json', 'w') as f:
        json.dump(pareto_data, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Efficiency Analysis")
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
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'efficiency' / timestamp

    try:
        run_efficiency_analysis(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Efficiency analysis completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
