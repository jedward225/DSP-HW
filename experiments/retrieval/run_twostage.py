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
    create_ssim_retriever,
    create_twostage_retriever,
)
from src.metrics.retrieval_metrics import aggregate_metrics, compute_all_metrics
from src.utils.seed import get_seed_from_config, set_seed

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

        if progress is not None and task_id is not None:
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

    seed = get_seed_from_config(yaml_config)
    if seed is not None:
        set_seed(seed, deterministic=bool(yaml_config.get('deterministic', False)))
        logger.info(f"Random seed set to {seed}")

    dataset_cfg = yaml_config.get('dataset', {})
    feat_cfg = yaml_config.get('features', {})
    dtw_cfg = yaml_config.get('dtw', {})
    twostage_cfg = yaml_config.get('twostage', {})
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

    # Cache samples in-memory to avoid repeated disk I/O across folds/N.
    sample_cache: Dict[int, dict] = {}

    def get_sample(idx: int) -> dict:
        if idx not in sample_cache:
            sample_cache[idx] = dataset[idx]
        return sample_cache[idx]

    fold_splits: Dict[int, tuple] = {}
    max_queries_per_fold = twostage_cfg.get('max_queries_per_fold', None)
    for fold in folds:
        query_indices, gallery_indices = dataset.get_query_gallery_indices(fold)
        if max_queries_per_fold is not None:
            query_indices = query_indices[: int(max_queries_per_fold)]
        query_samples = [get_sample(i) for i in query_indices]
        gallery_samples = [get_sample(i) for i in gallery_indices]
        fold_splits[fold] = (query_indices, gallery_indices, query_samples, gallery_samples)

    # N values to test
    gallery_size = len(dataset) * 4 // 5  # 80% for gallery
    default_n_values = [20, 50, 100, 200, 500, 1000, gallery_size]
    n_values = twostage_cfg.get('n_values', default_n_values)
    n_values = sorted({int(n) for n in n_values if int(n) > 0 and int(n) <= gallery_size})

    console.print(f"\n[bold]Testing N values: {n_values}[/bold]")

    # MFCC extraction is CPU-bound (librosa). Using CUDA here adds overhead for this
    # script, so we run coarse retrieval on CPU for speed.
    compute_device = 'cpu' if device == 'cuda' else device

    # Create base retrievers (used for feature extraction and coarse distances)
    coarse_retriever = create_method_m1(
        device=compute_device,
        sr=sr,
        n_mfcc=feat_cfg.get('n_mfcc', 20),
        n_mels=feat_cfg.get('n_mels', 128),
        n_fft=feat_cfg.get('n_fft', 2048),
        hop_length=feat_cfg.get('hop_length', 512),
    )

    stage2_method = str(twostage_cfg.get('stage2', 'dtw')).lower()
    if stage2_method not in {'dtw', 'ssim'}:
        raise ValueError("twostage.stage2 must be one of: 'dtw', 'ssim'")

    k_values = yaml_config.get('evaluation', {}).get('k_values', [1, 5, 10, 20])

    # Precompute labels + coarse features for all samples once.
    console.print("\n[bold]Precomputing coarse features...[/bold]")
    targets = torch.empty(len(dataset), dtype=torch.long)
    coarse_features: List[torch.Tensor] = [None] * len(dataset)
    for idx in range(len(dataset)):
        s = get_sample(idx)
        targets[idx] = int(s['target'])
        coarse_features[idx] = coarse_retriever.extract_features(s['waveform'], s.get('sr', sr))
    coarse_features_tensor = torch.stack(coarse_features, dim=0).to(device='cpu')
    console.print("[green]✓[/green] Coarse features cached")

    # Precompute fine features for all samples once.
    fine_sequences: Optional[List[np.ndarray]] = None
    ssim_features: Optional[List[torch.Tensor]] = None

    dtw_constraint = str(dtw_cfg.get('constraint', 'none')).lower()
    sakoe_radius = int(dtw_cfg.get('sakoe_chiba_radius', -1))
    use_delta = bool(dtw_cfg.get('use_delta', False))

    if stage2_method == 'dtw':
        console.print("\n[bold]Precomputing DTW MFCC sequences...[/bold]")
        fine_extractor = create_method_m5(
            device='cpu',
            sr=sr,
            n_mfcc=dtw_cfg.get('n_mfcc', 13),
            n_mels=dtw_cfg.get('n_mels', 64),
            n_fft=feat_cfg.get('n_fft', 2048),
            hop_length=feat_cfg.get('hop_length', 512),
            sakoe_chiba_radius=sakoe_radius,
            constraint=dtw_constraint,
            use_delta=use_delta,
        )

        fine_sequences = [None] * len(dataset)
        for idx in range(len(dataset)):
            s = get_sample(idx)
            seq = fine_extractor.extract_features(s['waveform'], s.get('sr', sr)).numpy()
            fine_sequences[idx] = np.ascontiguousarray(seq, dtype=np.float32)
        console.print("[green]✓[/green] DTW sequences cached")
    else:
        # SSIM stage2 (CPU by default)
        console.print("\n[bold]Precomputing SSIM features...[/bold]")
        ssim_device = compute_device
        ssim_extractor = create_ssim_retriever(
            device=ssim_device,
            sr=sr,
            n_mels=feat_cfg.get('n_mels', 128),
            n_fft=feat_cfg.get('n_fft', 2048),
            hop_length=feat_cfg.get('hop_length', 512),
        )
        ssim_features = [None] * len(dataset)
        for idx in range(len(dataset)):
            s = get_sample(idx)
            ssim_features[idx] = ssim_extractor.extract_features(s['waveform'], s.get('sr', sr))
        console.print("[green]✓[/green] SSIM features cached")

    results = {}
    queries_per_fold = max(1, max(len(fold_splits[f][2]) for f in folds))

    # Run baselines + two-stage sweep in one pass per fold using cached features.
    console.print("\n[bold cyan]Running Baseline + Two-Stage Sweep[/bold cyan]")

    from src.retrieval.dtw_retriever import (
        _dtw_distance_batch_numba,
        _dtw_distance_itakura,
        _fastdtw_recursive,
    )

    m1_fold_metrics: List[Dict[str, float]] = []
    fine_fold_metrics: List[Dict[str, float]] = []
    twostage_fold_metrics: Dict[int, List[Dict[str, float]]] = {n: [] for n in n_values}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total_queries = sum(len(fold_splits[f][2]) for f in folds)
        total = total_queries * (len(n_values) + 2)
        task = progress.add_task("[cyan]Two-Stage Sweep", total=total)

        for fold in folds:
            query_indices, gallery_indices, query_samples, gallery_samples = fold_splits[fold]

            gallery_features = coarse_features_tensor[gallery_indices]
            gallery_labels = targets[gallery_indices]
            gallery_size = len(gallery_indices)
            gallery_indices_np = np.asarray(gallery_indices, dtype=np.int64)

            if stage2_method == 'dtw':
                assert fine_sequences is not None
                gallery_fine = [fine_sequences[i] for i in gallery_indices]
                gallery_tensor = None
            else:
                assert ssim_features is not None
                gallery_fine = [ssim_features[i] for i in gallery_indices]
                gallery_tensor = torch.stack(gallery_fine, dim=0)

            # Per-fold accumulators
            m1_all: List[Dict[str, float]] = []
            fine_all: List[Dict[str, float]] = []
            twostage_all: Dict[int, List[Dict[str, float]]] = {n: [] for n in n_values}

            stage1_time = 0.0
            stage2_full_time = 0.0

            for query in query_samples:
                q_idx = int(query.get('idx'))
                q_label = int(query.get('target'))
                num_relevant = int((gallery_labels == q_label).sum().item())

                # Stage 1: coarse ranking
                t0 = time.perf_counter()
                coarse_distances = coarse_retriever.compute_distance(
                    coarse_features_tensor[q_idx],
                    gallery_features,
                )
                coarse_order = torch.argsort(coarse_distances, stable=True)
                stage1_time += time.perf_counter() - t0

                coarse_order_np = coarse_order.detach().cpu().numpy()

                # Baseline (coarse only)
                retrieved_labels_m1 = gallery_labels[coarse_order]
                m1_all.append(
                    compute_all_metrics(
                        retrieved_labels_m1,
                        q_label,
                        num_relevant=num_relevant,
                        k_values=k_values,
                    )
                )
                progress.update(task, advance=1)

                # Stage 2: fine distances over the full gallery (computed once per query)
                t1 = time.perf_counter()
                if stage2_method == 'dtw':
                    query_seq = fine_sequences[q_idx]
                    if dtw_constraint == 'itakura':
                        fine_distances = np.array(
                            [_dtw_distance_itakura(query_seq, g) for g in gallery_fine],
                            dtype=np.float32,
                        )
                    elif dtw_constraint == 'fastdtw':
                        fastdtw_radius = int(dtw_cfg.get('fastdtw_radius', 10))
                        fine_distances = np.array(
                            [_fastdtw_recursive(query_seq, g, fastdtw_radius) for g in gallery_fine],
                            dtype=np.float32,
                        )
                    else:
                        radius = sakoe_radius if dtw_constraint == 'sakoe_chiba' else -1
                        fine_distances = _dtw_distance_batch_numba(query_seq, gallery_fine, radius).astype(np.float32)
                else:
                    # SSIM
                    query_feat = ssim_features[q_idx]
                    distances_t = ssim_extractor.compute_distance(query_feat, gallery_tensor)
                    fine_distances = distances_t.detach().cpu().numpy().astype(np.float32)

                stage2_full_time += time.perf_counter() - t1

                # Fine-only baseline
                fine_order = np.argsort(fine_distances)
                retrieved_labels_fine = gallery_labels[torch.from_numpy(fine_order).long()]
                fine_all.append(
                    compute_all_metrics(
                        retrieved_labels_fine,
                        q_label,
                        num_relevant=num_relevant,
                        k_values=k_values,
                    )
                )
                progress.update(task, advance=1)

                # Two-stage: re-rank coarse top-N using fine distances
                for n in n_values:
                    cand_pos = coarse_order_np[:n]
                    cand_dist = fine_distances[cand_pos]
                    rerank = np.argsort(cand_dist)
                    reranked_pos = cand_pos[rerank]

                    # Build a full-length ranking: reranked top-N, then the remaining items
                    # in their original coarse order. Metrics like AP/mAP require the full
                    # gallery ranking so the denominator uses the true #relevant in gallery.
                    if n < gallery_size:
                        full_order = np.concatenate([reranked_pos, coarse_order_np[n:]], axis=0)
                    else:
                        full_order = reranked_pos

                    retrieved_labels_twostage = gallery_labels[torch.from_numpy(full_order).long()]
                    twostage_all[n].append(
                        compute_all_metrics(
                            retrieved_labels_twostage,
                            q_label,
                            num_relevant=num_relevant,
                            k_values=k_values,
                        )
                    )
                    progress.update(task, advance=1)

            # Aggregate per fold
            def agg_mean(all_m: List[Dict[str, float]]) -> Dict[str, float]:
                agg = aggregate_metrics(all_m)
                return {k: v['mean'] for k, v in agg.items() if isinstance(v, dict) and 'mean' in v}

            fold_m1 = agg_mean(m1_all)
            fold_fine = agg_mean(fine_all)

            n_queries = max(1, len(query_samples))
            avg_stage1_ms = stage1_time / n_queries * 1000
            avg_stage2_full_ms = stage2_full_time / n_queries * 1000

            fold_m1['avg_query_time_ms'] = avg_stage1_ms
            fold_fine['avg_query_time_ms'] = avg_stage2_full_ms

            m1_fold_metrics.append(fold_m1)
            fine_fold_metrics.append(fold_fine)

            for n in n_values:
                fold_twostage = agg_mean(twostage_all[n])
                # Estimated time scales ~linearly with #candidates
                fold_twostage['avg_query_time_ms'] = avg_stage1_ms + avg_stage2_full_ms * (n / max(1, gallery_size))
                twostage_fold_metrics[n].append(fold_twostage)

    # Final aggregation across folds
    def mean_std(metrics_per_fold: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        keys = metrics_per_fold[0].keys() if metrics_per_fold else []
        return {
            'mean': {k: float(np.mean([m.get(k, 0.0) for m in metrics_per_fold])) for k in keys},
            'std': {k: float(np.std([m.get(k, 0.0) for m in metrics_per_fold])) for k in keys},
        }

    m1_results = mean_std(m1_fold_metrics)
    results['M1_baseline'] = m1_results

    fine_results = mean_std(fine_fold_metrics)
    results['M5_baseline' if stage2_method == 'dtw' else 'SSIM_baseline'] = fine_results

    for n in n_values:
        fold_metrics = twostage_fold_metrics[n]
        results[f'TwoStage_N{n}'] = {
            'n': n,
            'mean': {k: float(np.mean([m.get(k, 0.0) for m in fold_metrics])) for k in fold_metrics[0].keys()},
            'std': {k: float(np.std([m.get(k, 0.0) for m in fold_metrics])) for k in fold_metrics[0].keys()},
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
        "DTW (Fine only)" if stage2_method == 'dtw' else "SSIM (Fine only)",
        "-",
        f"{fine_results['mean'].get('hit@10', 0):.4f}",
        f"{fine_results['mean'].get('map@20', 0):.4f}",
        f"{fine_results['mean'].get('avg_query_time_ms', 0):.2f}",
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
