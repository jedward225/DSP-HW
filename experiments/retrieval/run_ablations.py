#!/usr/bin/env python3
"""
Feature Engineering Ablation Study

This script tests different feature engineering choices:
  1. Pre-emphasis: 0.97 vs none (0)
  2. CMVN: per-utterance vs global vs none
  3. Mel formula: HTK vs Slaney

Usage:
    python run_ablations.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import logging
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
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box

# Project imports
from src.data.esc50 import ESC50Dataset
from src.dsp_core import mfcc, pre_emphasis, cmvn
from src.retrieval.base import BaseRetriever
from src.features.pooling import mean_std_pool
from src.metrics.retrieval_metrics import aggregate_metrics
from src.utils.seed import get_seed_from_config, set_seed

console = Console()


class AblationRetriever(BaseRetriever):
    """
    Custom retriever for ablation studies.

    Supports:
    - Pre-emphasis coefficient
    - CMVN mode (none, utterance, global)
    - HTK vs Slaney mel formula
    """

    def __init__(
        self,
        name: str = "AblationRetriever",
        device: str = 'cpu',
        sr: int = 22050,
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        # Ablation parameters
        pre_emphasis_coef: float = 0.0,  # 0 = no pre-emphasis
        cmvn_mode: str = 'none',  # 'none', 'utterance', 'global'
        htk: bool = False,  # True = HTK, False = Slaney
    ):
        super().__init__(name=name, device=device, sr=sr)
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pre_emphasis_coef = pre_emphasis_coef
        self.cmvn_mode = cmvn_mode
        self.htk = htk

        # For global CMVN, store running statistics
        self._global_mean = None
        self._global_std = None

    def extract_features(
        self,
        waveform: torch.Tensor,
        sr: int = None
    ) -> torch.Tensor:
        """Extract MFCC features with ablation options."""
        sr = sr or self.sr

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        # Apply pre-emphasis if enabled
        if self.pre_emphasis_coef > 0:
            waveform_np = pre_emphasis(waveform_np, coef=self.pre_emphasis_coef)

        # Extract MFCC with HTK option
        mfcc_features = mfcc(
            y=waveform_np,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            htk=self.htk,
        )

        # Apply CMVN based on mode
        if self.cmvn_mode == 'utterance':
            mfcc_features = cmvn(mfcc_features, axis=-1, variance_normalization=True)
        elif self.cmvn_mode == 'global' and self._global_mean is not None:
            mfcc_features = (mfcc_features - self._global_mean) / (self._global_std + 1e-10)

        # Convert to torch and pool
        mfcc_tensor = torch.from_numpy(mfcc_features).float().to(self.device)
        pooled = mean_std_pool(mfcc_tensor, dim=-1)

        return pooled

    def compute_global_statistics(self, samples: List[dict]):
        """Compute global mean and std for CMVN from training data."""
        all_features = []
        for sample in samples:
            waveform = sample['waveform']
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()

            # Apply pre-emphasis
            if self.pre_emphasis_coef > 0:
                waveform = pre_emphasis(waveform, coef=self.pre_emphasis_coef)

            # Extract MFCC
            features = mfcc(
                y=waveform,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                htk=self.htk,
            )
            all_features.append(features)

        # Concatenate all frames
        all_features = np.concatenate(all_features, axis=1)
        self._global_mean = np.mean(all_features, axis=1, keepdims=True)
        self._global_std = np.std(all_features, axis=1, keepdims=True)

    def compute_distance(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine distance."""
        query_norm = query_features / (query_features.norm() + 1e-10)
        gallery_norm = gallery_features / (gallery_features.norm(dim=1, keepdim=True) + 1e-10)
        similarity = torch.matmul(gallery_norm, query_norm)
        return 1 - similarity


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    pre_emphasis_coef: float = 0.0
    cmvn_mode: str = 'none'
    htk: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_file = output_dir / 'ablations.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w')]
    )
    return logging.getLogger('ablations')


def evaluate_ablation(
    ablation_config: AblationConfig,
    dataset: ESC50Dataset,
    base_config: Dict,
    device: str,
    sr: int,
    folds: List[int],
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Dict:
    """Evaluate a single ablation configuration."""
    fold_results = {}
    all_metrics = []

    for fold in folds:
        # Create retriever
        retriever = AblationRetriever(
            name=ablation_config.name,
            device=device,
            sr=sr,
            n_mfcc=base_config.get('n_mfcc', 20),
            n_mels=base_config.get('n_mels', 128),
            n_fft=base_config.get('n_fft', 2048),
            hop_length=base_config.get('hop_length', 512),
            pre_emphasis_coef=ablation_config.pre_emphasis_coef,
            cmvn_mode=ablation_config.cmvn_mode,
            htk=ablation_config.htk,
        )

        # Get query/gallery split
        query_samples, gallery_samples = dataset.get_query_gallery_split(fold)

        # Compute global statistics if needed
        if ablation_config.cmvn_mode == 'global':
            retriever.compute_global_statistics(gallery_samples)

        # Build gallery
        retriever.build_gallery(gallery_samples, show_progress=False)

        # Evaluate queries
        query_metrics = []
        for query in query_samples:
            metrics = retriever.evaluate_query(query)
            query_metrics.append(metrics)
            if progress and task_id:
                progress.update(task_id, advance=1)

        # Aggregate
        fold_agg = aggregate_metrics(query_metrics)
        fold_result = {k: v['mean'] for k, v in fold_agg.items() if isinstance(v, dict) and 'mean' in v}
        fold_results[fold] = fold_result
        all_metrics.append(fold_result)

    # Compute mean and std across folds
    mean_metrics = {}
    std_metrics = {}
    if all_metrics:
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics]
            mean_metrics[metric] = float(np.mean(values))
            std_metrics[metric] = float(np.std(values))

    return {
        'config': ablation_config.to_dict(),
        'mean': mean_metrics,
        'std': std_metrics,
        'fold_results': {str(k): v for k, v in fold_results.items()},
    }


def run_preemphasis_ablation(
    dataset: ESC50Dataset,
    base_config: Dict,
    device: str,
    sr: int,
    folds: List[int],
    logger: logging.Logger,
) -> List[Dict]:
    """Run pre-emphasis ablation study."""
    console.print("\n[bold cyan]Pre-emphasis Ablation[/bold cyan]")

    configs = [
        AblationConfig(name="no_preemph", pre_emphasis_coef=0.0),
        AblationConfig(name="preemph_0.97", pre_emphasis_coef=0.97),
    ]

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
        task = progress.add_task("[cyan]Pre-emphasis", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_ablation(config, dataset, base_config, device, sr, folds, progress, task)
            results.append(result)
            logger.info(f"  {config.name}: hit@10={result['mean'].get('hit@10', 0):.4f}")

    return results


def run_cmvn_ablation(
    dataset: ESC50Dataset,
    base_config: Dict,
    device: str,
    sr: int,
    folds: List[int],
    logger: logging.Logger,
) -> List[Dict]:
    """Run CMVN ablation study."""
    console.print("\n[bold cyan]CMVN Ablation[/bold cyan]")

    configs = [
        AblationConfig(name="no_cmvn", cmvn_mode='none'),
        AblationConfig(name="cmvn_utterance", cmvn_mode='utterance'),
        AblationConfig(name="cmvn_global", cmvn_mode='global'),
    ]

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
        task = progress.add_task("[cyan]CMVN", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_ablation(config, dataset, base_config, device, sr, folds, progress, task)
            results.append(result)
            logger.info(f"  {config.name}: hit@10={result['mean'].get('hit@10', 0):.4f}")

    return results


def run_mel_formula_ablation(
    dataset: ESC50Dataset,
    base_config: Dict,
    device: str,
    sr: int,
    folds: List[int],
    logger: logging.Logger,
) -> List[Dict]:
    """Run HTK vs Slaney mel formula ablation."""
    console.print("\n[bold cyan]Mel Formula Ablation[/bold cyan]")

    configs = [
        AblationConfig(name="slaney", htk=False),
        AblationConfig(name="htk", htk=True),
    ]

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
        task = progress.add_task("[cyan]Mel Formula", total=len(configs) * queries_per_config)

        for config in configs:
            result = evaluate_ablation(config, dataset, base_config, device, sr, folds, progress, task)
            results.append(result)
            logger.info(f"  {config.name}: hit@10={result['mean'].get('hit@10', 0):.4f}")

    return results


def display_results_table(results: List[Dict], title: str, metrics: List[str]):
    """Display ablation results in a table."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Configuration", style="bold")
    for metric in metrics:
        table.add_column(metric, justify="center")

    # Find best for highlighting
    best_idx = 0
    best_score = 0
    for i, result in enumerate(results):
        if result['mean'].get('hit@10', 0) > best_score:
            best_score = result['mean'].get('hit@10', 0)
            best_idx = i

    for i, result in enumerate(results):
        style = "bold green" if i == best_idx else ""
        row = [result['config']['name']]
        for metric in metrics:
            mean = result['mean'].get(metric, 0)
            std = result['std'].get(metric, 0)
            row.append(f"{mean:.4f}±{std:.4f}")
        table.add_row(*row, style=style)

    console.print(table)


def run_ablations(config_path: str, output_dir: Path):
    """Run all ablation studies."""
    # Load base config
    yaml_config = load_config(config_path)

    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    seed = get_seed_from_config(yaml_config)
    if seed is not None:
        set_seed(seed, deterministic=bool(yaml_config.get('deterministic', False)))
        logger.info(f"Random seed set to {seed}")

    dataset_cfg = yaml_config.get('dataset', {})
    feature_cfg = yaml_config.get('features', {})
    device = yaml_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    sr = dataset_cfg.get('sr', 22050)
    folds = yaml_config.get('evaluation', {}).get('folds', [1, 2, 3, 4, 5])

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Set device='cpu' in config.")

    # Header
    console.print(Panel.fit(
        "[bold blue]Feature Engineering Ablation Study[/bold blue]\n"
        f"Device: {device} | Folds: {folds}",
        border_style="blue"
    ))

    logger.info("=" * 60)
    logger.info("ABLATION STUDY STARTED")
    logger.info("=" * 60)

    # Load dataset
    console.print("\n[bold]Loading ESC-50 dataset...[/bold]")
    dataset_root = PROJECT_ROOT / dataset_cfg.get('root_dir', 'ESC-50')
    dataset = ESC50Dataset(root_dir=str(dataset_root), sr=sr, preload=False)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")

    # Metrics to display
    metrics = ['hit@10', 'precision@10', 'mrr@10', 'map@20']

    all_results = {}

    # 1. Pre-emphasis ablation
    preemph_results = run_preemphasis_ablation(dataset, feature_cfg, device, sr, folds, logger)
    all_results['preemphasis'] = preemph_results
    display_results_table(preemph_results, "Pre-emphasis Results (5-fold CV)", metrics)

    with open(output_dir / 'preemphasis.json', 'w') as f:
        json.dump(preemph_results, f, indent=2)

    # 2. CMVN ablation
    cmvn_results = run_cmvn_ablation(dataset, feature_cfg, device, sr, folds, logger)
    all_results['cmvn'] = cmvn_results
    display_results_table(cmvn_results, "CMVN Results (5-fold CV)", metrics)

    with open(output_dir / 'cmvn.json', 'w') as f:
        json.dump(cmvn_results, f, indent=2)

    # 3. Mel formula ablation
    mel_results = run_mel_formula_ablation(dataset, feature_cfg, device, sr, folds, logger)
    all_results['mel_formula'] = mel_results
    display_results_table(mel_results, "Mel Formula Results (5-fold CV)", metrics)

    with open(output_dir / 'mel_formula.json', 'w') as f:
        json.dump(mel_results, f, indent=2)

    # Save combined results
    with open(output_dir / 'all_ablations.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Ablation Study")
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'retrieval' / 'configs' / 'default.yaml'),
        help='Path to base configuration file'
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
        output_dir = PROJECT_ROOT / 'experiments' / 'retrieval' / 'results' / 'ablations' / timestamp

    try:
        run_ablations(args.config, output_dir)
        console.print(Panel.fit(
            "[bold green]Ablation study completed![/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


if __name__ == '__main__':
    main()
