#!/usr/bin/env python3
"""
Robustness Testing for Sound Retrieval Methods

This script tests robustness against:
  1. Additive noise (SNR = 20dB, 10dB, 0dB)
  2. Volume scaling (Gain = +6dB, -6dB)
  3. Speed perturbation (0.9x, 1.1x)
  4. Pitch shift (±1 semitone)

Protocol: Gallery uses clean audio, queries use perturbed audio.

Usage:
    python run_robustness.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
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
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.retrieval import create_method_m1, create_method_m2, create_method_m3
from src.utils.augmentation import AudioAugmenter
from src.metrics.retrieval_metrics import aggregate_metrics
from src.utils.seed import get_seed_from_config, set_seed

console = Console()


@dataclass
class PerturbationConfig:
    """Configuration for a perturbation experiment."""
    name: str
    perturbation_type: str  # 'noise', 'volume', 'speed', 'pitch'
    value: float  # SNR in dB, gain in dB, speed rate, semitones
    description: str

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(config_path: str) -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'robustness.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('robustness')


def apply_perturbation(
    waveform: np.ndarray,
    perturbation: PerturbationConfig,
    augmenter: AudioAugmenter,
    target_length: int,
) -> np.ndarray:
    """
    Apply perturbation to waveform.

    Args:
        waveform: Audio signal
        perturbation: Perturbation configuration
        augmenter: AudioAugmenter instance
        target_length: Target length for output (for speed changes)

    Returns:
        Perturbed waveform
    """
    if perturbation.perturbation_type == 'noise':
        return augmenter.add_noise(waveform, snr_db=perturbation.value)

    elif perturbation.perturbation_type == 'volume':
        return augmenter.scale_volume(waveform, gain_db=perturbation.value)

    elif perturbation.perturbation_type == 'speed':
        perturbed = augmenter.change_speed(waveform, rate=perturbation.value)
        # Pad or trim to original length
        if len(perturbed) < target_length:
            perturbed = np.pad(perturbed, (0, target_length - len(perturbed)))
        else:
            perturbed = perturbed[:target_length]
        return perturbed

    elif perturbation.perturbation_type == 'pitch':
        return augmenter.pitch_shift(waveform, semitones=perturbation.value)

    elif perturbation.perturbation_type == 'time_shift':
        # Time shift: value is the shift fraction (e.g., 0.1 = 10% shift)
        return augmenter.time_shift(waveform, shift_fraction=perturbation.value)

    elif perturbation.perturbation_type == 'clean':
        return waveform

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation.perturbation_type}")


def evaluate_robustness(
    retriever,
    query_samples: List[dict],
    gallery_samples: List[dict],
    perturbation: PerturbationConfig,
    augmenter: AudioAugmenter,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate retriever with perturbed queries on clean gallery.

    Args:
        retriever: Retrieval method
        query_samples: Query samples (will be perturbed)
        gallery_samples: Gallery samples (kept clean)
        perturbation: Perturbation to apply
        augmenter: AudioAugmenter instance
        progress: Progress bar
        task_id: Progress task ID

    Returns:
        Aggregated metrics
    """
    # Build gallery with clean audio
    retriever.build_gallery(gallery_samples, show_progress=False)

    # Evaluate with perturbed queries
    all_metrics = []
    for query in query_samples:
        # Get waveform
        waveform = query['waveform']
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        # Apply perturbation
        perturbed_waveform = apply_perturbation(
            waveform, perturbation, augmenter, len(waveform)
        )

        # Create perturbed query
        perturbed_query = query.copy()
        perturbed_query['waveform'] = perturbed_waveform

        # Evaluate
        metrics = retriever.evaluate_query(perturbed_query)
        all_metrics.append(metrics)

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

    # Aggregate
    agg = aggregate_metrics(all_metrics)
    return {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}


def run_robustness_test(
    perturbations: List[PerturbationConfig],
    dataset: ESC50Dataset,
    config: Dict,
    device: str,
    sr: int,
    folds: List[int],
    logger: logging.Logger,
) -> Dict[str, List[Dict]]:
    """
    Run robustness tests for all perturbations.

    Uses M1 (MFCC) as the reference method.
    """
    feat_cfg = config.get('features', {})

    # Create retriever
    retriever = create_method_m1(
        device=device,
        sr=sr,
        n_mfcc=feat_cfg.get('n_mfcc', 20),
        n_mels=feat_cfg.get('n_mels', 128),
        n_fft=feat_cfg.get('n_fft', 2048),
        hop_length=feat_cfg.get('hop_length', 512),
    )

    # Create augmenter
    augmenter = AudioAugmenter(sr=sr)

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
        total = len(perturbations) * len(folds) * queries_per_fold
        task = progress.add_task("[cyan]Robustness Testing", total=total)

        for perturbation in perturbations:
            progress.update(task, description=f"[cyan]{perturbation.name}")
            fold_metrics = []

            for fold in folds:
                query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

                metrics = evaluate_robustness(
                    retriever, query_samples, gallery_samples,
                    perturbation, augmenter, progress, task
                )
                fold_metrics.append(metrics)

            # Average across folds
            mean_metrics = {}
            std_metrics = {}
            for metric in fold_metrics[0].keys():
                values = [m[metric] for m in fold_metrics]
                mean_metrics[metric] = float(np.mean(values))
                std_metrics[metric] = float(np.std(values))

            results[perturbation.name] = {
                'config': perturbation.to_dict(),
                'mean': mean_metrics,
                'std': std_metrics,
            }

            logger.info(f"{perturbation.name}: hit@10={mean_metrics.get('hit@10', 0):.4f}")

    return results


def display_results_table(results: Dict[str, Dict], title: str):
    """Display robustness results."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Perturbation", style="bold")
    table.add_column("Value", justify="center")
    table.add_column("Hit@10", justify="center")
    table.add_column("P@10", justify="center")
    table.add_column("mAP@20", justify="center")
    table.add_column("Degradation", justify="center")

    # Get clean baseline
    clean_hit10 = results.get('clean', {}).get('mean', {}).get('hit@10', 1.0)

    for name, result in results.items():
        mean = result['mean']
        std = result['std']
        config = result['config']

        hit10 = mean.get('hit@10', 0)
        degradation = (clean_hit10 - hit10) / clean_hit10 * 100 if clean_hit10 > 0 else 0

        # Color based on degradation
        if name == 'clean':
            style = "bold green"
            degrad_str = "-"
        elif degradation < 5:
            style = "green"
            degrad_str = f"-{degradation:.1f}%"
        elif degradation < 15:
            style = "yellow"
            degrad_str = f"-{degradation:.1f}%"
        else:
            style = "red"
            degrad_str = f"-{degradation:.1f}%"

        table.add_row(
            config.get('perturbation_type', name),
            str(config.get('value', '-')),
            f"{hit10:.4f}±{std.get('hit@10', 0):.4f}",
            f"{mean.get('precision@10', 0):.4f}",
            f"{mean.get('map@20', 0):.4f}",
            degrad_str,
            style=style
        )

    console.print(table)


def run_robustness_experiments(config_path: str, output_dir: Path):
    """Run all robustness experiments."""
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
        "[bold blue]Robustness Testing[/bold blue]\n"
        f"Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("ROBUSTNESS TESTING STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Define perturbations
    perturbations = [
        # Baseline (clean)
        PerturbationConfig("clean", "clean", 0, "No perturbation"),

        # Noise
        PerturbationConfig("noise_20dB", "noise", 20, "Additive Gaussian noise at 20dB SNR"),
        PerturbationConfig("noise_10dB", "noise", 10, "Additive Gaussian noise at 10dB SNR"),
        PerturbationConfig("noise_0dB", "noise", 0, "Additive Gaussian noise at 0dB SNR"),

        # Volume
        PerturbationConfig("volume_+6dB", "volume", 6, "Volume increased by 6dB"),
        PerturbationConfig("volume_-6dB", "volume", -6, "Volume decreased by 6dB"),

        # Speed
        PerturbationConfig("speed_0.9x", "speed", 0.9, "Playback speed 0.9x (slower)"),
        PerturbationConfig("speed_1.1x", "speed", 1.1, "Playback speed 1.1x (faster)"),

        # Pitch (skip if librosa pitch_shift is slow)
        PerturbationConfig("pitch_-1", "pitch", -1, "Pitch shifted down by 1 semitone"),
        PerturbationConfig("pitch_+1", "pitch", 1, "Pitch shifted up by 1 semitone"),

        # Time shift
        PerturbationConfig("time_shift_0.1", "time_shift", 0.1, "Time shifted by 10%"),
        PerturbationConfig("time_shift_0.2", "time_shift", 0.2, "Time shifted by 20%"),
    ]

    console.print(f"\n[bold]Testing {len(perturbations)} perturbations...[/bold]")

    # Run tests
    all_results = run_robustness_test(
        perturbations, dataset, yaml_config, device, sr, folds, logger
    )

    # Display results by category
    console.print("\n")

    # Noise results
    noise_results = {k: v for k, v in all_results.items() if 'noise' in k or k == 'clean'}
    display_results_table(noise_results, "Noise Robustness")

    # Volume results
    volume_results = {k: v for k, v in all_results.items() if 'volume' in k or k == 'clean'}
    display_results_table(volume_results, "Volume Robustness")

    # Speed results
    speed_results = {k: v for k, v in all_results.items() if 'speed' in k or k == 'clean'}
    display_results_table(speed_results, "Speed Robustness")

    # Pitch results
    pitch_results = {k: v for k, v in all_results.items() if 'pitch' in k or k == 'clean'}
    display_results_table(pitch_results, "Pitch Robustness")

    # Save results
    with open(output_dir / 'noise.json', 'w') as f:
        json.dump(noise_results, f, indent=2)

    with open(output_dir / 'volume.json', 'w') as f:
        json.dump(volume_results, f, indent=2)

    with open(output_dir / 'speed.json', 'w') as f:
        json.dump(speed_results, f, indent=2)

    with open(output_dir / 'all_robustness.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Robustness Testing")
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
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'robustness' / timestamp

    try:
        run_robustness_experiments(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Robustness testing completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
