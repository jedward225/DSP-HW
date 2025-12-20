"""
Data loader for experiment results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Handle both module and direct execution imports
try:
    from .config import RESULTS_DIR
except ImportError:
    from config import RESULTS_DIR


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_latest_results(subdir: str) -> Optional[Path]:
    """
    Find the latest timestamped results directory.

    Args:
        subdir: subdirectory name (e.g., 'pretrained', 'ablations')

    Returns:
        Path to the latest results directory, or None if not found
    """
    base_path = RESULTS_DIR / subdir
    if not base_path.exists():
        return None

    # Find all timestamped directories
    dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('20')]
    if not dirs:
        return None

    # Return the latest one
    return sorted(dirs)[-1]


# =============================================================================
# Baseline Results
# =============================================================================

def load_baseline_results() -> Dict[str, Any]:
    """
    Load baseline retrieval results (M1-M7 traditional methods).

    Returns:
        Dictionary with method results
    """
    # Find the baseline results directory
    base_dirs = [d for d in RESULTS_DIR.iterdir()
                 if d.is_dir() and d.name.startswith('20')]

    if not base_dirs:
        raise FileNotFoundError("No baseline results found")

    latest_dir = sorted(base_dirs)[-1]
    results_file = latest_dir / "results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return load_json(results_file)


# =============================================================================
# Pretrained Model Results
# =============================================================================

def load_pretrained_results() -> Dict[str, Any]:
    """
    Load pretrained model results (CLAP, BEATs, Hybrid).

    Returns:
        Dictionary with pretrained method results
    """
    results_dir = find_latest_results('pretrained')
    if results_dir is None:
        raise FileNotFoundError("No pretrained results found")

    results_file = results_dir / "results.json"
    return load_json(results_file)


# =============================================================================
# Deep Retriever Results
# =============================================================================

def load_deep_results() -> Dict[str, Any]:
    """
    Load deep learning retriever results (Autoencoder, CNN, Contrastive).

    Returns:
        Dictionary with deep learning method results
    """
    results_dir = find_latest_results('deep_retrievers')
    if results_dir is None:
        raise FileNotFoundError("No deep retriever results found")

    results_file = results_dir / "results.json"
    return load_json(results_file)


# =============================================================================
# Grid Search Results
# =============================================================================

def load_grid_search_results() -> Dict[str, Any]:
    """
    Load grid search results for frame/hop length optimization.

    Returns:
        Dictionary with grid search results
    """
    results_dir = find_latest_results('grid_search')
    if results_dir is None:
        raise FileNotFoundError("No grid search results found")

    results = {}

    # Load step 1: frame_hop
    step1_file = results_dir / "step1_frame_hop.json"
    if step1_file.exists():
        results['frame_hop'] = load_json(step1_file)

    # Load step 2: mfcc_params
    step2_file = results_dir / "step2_mfcc_params.json"
    if step2_file.exists():
        results['mfcc_params'] = load_json(step2_file)

    # Load step 3: window
    step3_file = results_dir / "step3_window.json"
    if step3_file.exists():
        results['window'] = load_json(step3_file)

    return results


# =============================================================================
# Ablation Results
# =============================================================================

def load_ablation_results() -> Dict[str, Any]:
    """
    Load ablation study results.

    Returns:
        Dictionary with ablation results (preemphasis, cmvn, mel_formula)
    """
    results_dir = find_latest_results('ablations')
    if results_dir is None:
        raise FileNotFoundError("No ablation results found")

    results_file = results_dir / "all_ablations.json"
    return load_json(results_file)


# =============================================================================
# Robustness Results
# =============================================================================

def load_robustness_results() -> Dict[str, Any]:
    """
    Load robustness test results.

    Returns:
        Dictionary with robustness results (noise, volume, speed)
    """
    results_dir = find_latest_results('robustness')
    if results_dir is None:
        raise FileNotFoundError("No robustness results found")

    results = {}

    # Load noise results
    noise_file = results_dir / "noise.json"
    if noise_file.exists():
        results['noise'] = load_json(noise_file)

    # Load volume results
    volume_file = results_dir / "volume.json"
    if volume_file.exists():
        results['volume'] = load_json(volume_file)

    # Load speed results
    speed_file = results_dir / "speed.json"
    if speed_file.exists():
        results['speed'] = load_json(speed_file)

    # Load all robustness results
    all_file = results_dir / "all_robustness.json"
    if all_file.exists():
        results['all'] = load_json(all_file)

    return results


# =============================================================================
# Fusion Results
# =============================================================================

def load_fusion_results() -> Dict[str, Any]:
    """
    Load fusion experiment results.

    Returns:
        Dictionary with fusion results (late_fusion, rank_fusion)
    """
    results_dir = find_latest_results('fusion')
    if results_dir is None:
        raise FileNotFoundError("No fusion results found")

    results_file = results_dir / "all_fusion.json"
    return load_json(results_file)


# =============================================================================
# Efficiency Results
# =============================================================================

def load_efficiency_results() -> Dict[str, Any]:
    """
    Load efficiency analysis results.

    Returns:
        Dictionary with timing, memory, and throughput results
    """
    results_dir = find_latest_results('efficiency')
    if results_dir is None:
        raise FileNotFoundError("No efficiency results found")

    results_file = results_dir / "timing.json"
    return load_json(results_file)


# =============================================================================
# Two-Stage Results
# =============================================================================

def load_twostage_results() -> Dict[str, Any]:
    """
    Load two-stage retrieval results.

    Returns:
        Dictionary with N sweep results
    """
    results_dir = find_latest_results('twostage')
    if results_dir is None:
        raise FileNotFoundError("No two-stage results found")

    results_file = results_dir / "n_sweep.json"
    return load_json(results_file)


# =============================================================================
# Partial Query Results
# =============================================================================

def load_partial_results() -> Dict[str, Any]:
    """
    Load partial query experiment results.

    Returns:
        Dictionary with partial query results for different durations
    """
    results_dir = find_latest_results('partial')
    if results_dir is None:
        raise FileNotFoundError("No partial query results found")

    results_file = results_dir / "partial_query.json"
    return load_json(results_file)


# =============================================================================
# Combined Data Loading
# =============================================================================

def load_all_method_results() -> Dict[str, Dict[str, Any]]:
    """
    Load all method results and combine them.

    Returns:
        Dictionary with all methods and their mean/std metrics
    """
    all_results = {}

    # Load baseline results
    try:
        baseline = load_baseline_results()
        for method, data in baseline.get('methods', {}).items():
            all_results[method] = {
                'mean': data.get('mean', {}),
                'std': data.get('std', {}),
                'ci': data.get('ci', {}),
                'folds': data.get('folds', {}),
                'category': 'traditional'
            }
    except FileNotFoundError:
        print("Warning: Baseline results not found")

    # Load pretrained results
    try:
        pretrained = load_pretrained_results()
        for method, data in pretrained.get('methods', {}).items():
            all_results[method] = {
                'mean': data.get('mean', {}),
                'std': data.get('std', {}),
                'ci': data.get('ci', {}),
                'folds': data.get('folds', {}),
                'category': 'pretrained'
            }
    except FileNotFoundError:
        print("Warning: Pretrained results not found")

    # Load deep retriever results
    try:
        deep = load_deep_results()
        for method, data in deep.get('methods', {}).items():
            all_results[method] = {
                'mean': data.get('mean', {}),
                'std': data.get('std', {}),
                'ci': data.get('ci', {}),
                'folds': data.get('folds', {}),
                'category': 'deep'
            }
    except FileNotFoundError:
        print("Warning: Deep retriever results not found")

    return all_results


def results_to_dataframe(results: Dict[str, Dict[str, Any]],
                         metrics: list = None) -> pd.DataFrame:
    """
    Convert results dictionary to a pandas DataFrame.

    Args:
        results: Dictionary of method results
        metrics: List of metrics to include (default: all)

    Returns:
        DataFrame with methods as rows and metrics as columns
    """
    if metrics is None:
        metrics = ['hit@1', 'hit@5', 'hit@10', 'hit@20',
                   'precision@1', 'precision@5', 'precision@10', 'precision@20',
                   'mrr@10', 'mrr@20', 'map@10', 'map@20', 'ndcg@10', 'ndcg@20']

    data = []
    for method, method_data in results.items():
        row = {'method': method, 'category': method_data.get('category', 'unknown')}
        mean_data = method_data.get('mean', {})
        std_data = method_data.get('std', {})

        for metric in metrics:
            row[f'{metric}_mean'] = mean_data.get(metric, 0)
            row[f'{metric}_std'] = std_data.get(metric, 0)

        data.append(row)

    return pd.DataFrame(data)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    # Test loading all results
    print("Testing data loader...")

    print("\n1. Loading baseline results...")
    try:
        baseline = load_baseline_results()
        print(f"   Found {len(baseline.get('methods', {}))} methods")
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    print("\n2. Loading pretrained results...")
    try:
        pretrained = load_pretrained_results()
        print(f"   Found {len(pretrained.get('methods', {}))} methods")
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    print("\n3. Loading deep retriever results...")
    try:
        deep = load_deep_results()
        print(f"   Found {len(deep.get('methods', {}))} methods")
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    print("\n4. Loading all method results...")
    all_methods = load_all_method_results()
    print(f"   Total methods: {len(all_methods)}")

    print("\n5. Converting to DataFrame...")
    df = results_to_dataframe(all_methods)
    print(df[['method', 'category', 'hit@10_mean', 'hit@10_std']].to_string())
