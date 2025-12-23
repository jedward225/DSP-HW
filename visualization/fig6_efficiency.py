"""
Figure 6: Efficiency Analysis (Pareto Front)

This figure contains:
(a) Accuracy vs Latency Pareto Front
(b) Feature Extraction Time
(c) Throughput (QPS)
(d) Memory Footprint
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import (
        setup_style, save_figure, METHOD_COLORS, METHOD_NAMES,
        CATEGORY_COLORS, get_method_color, get_method_name
    )
    from .data_loader import load_efficiency_results, load_all_method_results
except ImportError:
    from config import (
        setup_style, save_figure, METHOD_COLORS, METHOD_NAMES,
        CATEGORY_COLORS, get_method_color, get_method_name
    )
    from data_loader import load_efficiency_results, load_all_method_results


def is_pareto_efficient(costs):
    """
    Find Pareto-efficient points.
    costs: array of shape (n_points, n_costs) - lower is better for all costs
    Returns: boolean array indicating Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep points that are not dominated by c
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def create_figure():
    """Create the efficiency analysis figure."""

    setup_style()

    # Load data
    try:
        efficiency_raw = load_efficiency_results()
    except FileNotFoundError:
        print("Warning: Efficiency results not found")
        return None

    # Convert methods list to dict format for easier access
    efficiency_results = {}
    for item in efficiency_raw.get('methods', []):
        method_name = item.get('method_name', '')
        efficiency_results[method_name] = item

    # Also load method performance for Pareto plot
    try:
        method_results = load_all_method_results()
    except:
        method_results = {}

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Efficiency Analysis', fontsize=14, fontweight='bold', y=0.98)

    # =========================================================================
    # (a) Accuracy vs Latency Pareto Front
    # =========================================================================
    ax1 = axes[0, 0]

    methods = []
    hit10_values = []
    latency_values = []

    for method, data in efficiency_results.items():
        methods.append(method)
        # Get hit@10 from method results if available
        if method in method_results:
            hit10 = method_results[method]['mean'].get('hit@10', 0) * 100
        else:
            hit10 = data.get('hit@10_mean', data.get('mean', {}).get('hit@10', 0)) * 100
        hit10_values.append(hit10)

        # Get retrieval time
        retrieval_time = data.get('retrieval_time_ms', 0)
        if isinstance(retrieval_time, dict):
            retrieval_time = retrieval_time.get('mean', 0)
        latency_values.append(retrieval_time)

    hit10_array = np.array(hit10_values)
    latency_array = np.array(latency_values)

    # Find Pareto-efficient points (higher accuracy, lower latency is better)
    # Convert to minimization problem: minimize (-accuracy, latency)
    costs = np.column_stack([-hit10_array, latency_array])
    pareto_mask = is_pareto_efficient(costs)

    # Plot all points
    for i, method in enumerate(methods):
        color = get_method_color(method)
        marker = '^' if pareto_mask[i] else 'o'
        size = 150 if pareto_mask[i] else 80
        ax1.scatter(latency_array[i], hit10_array[i], c=color, s=size,
                   marker=marker, edgecolors='black' if pareto_mask[i] else 'none',
                   linewidths=2 if pareto_mask[i] else 0, zorder=3 if pareto_mask[i] else 2)

    # Connect Pareto-efficient points
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) > 1:
        # Sort by latency for line connection
        sorted_idx = pareto_indices[np.argsort(latency_array[pareto_indices])]
        ax1.plot(latency_array[sorted_idx], hit10_array[sorted_idx],
                'k--', alpha=0.5, linewidth=1.5, label='Pareto Front')

    # Add method labels
    for i, method in enumerate(methods):
        offset = (5, 5) if not pareto_mask[i] else (8, 8)
        ax1.annotate(get_method_name(method), (latency_array[i], hit10_array[i]),
                    xytext=offset, textcoords='offset points', fontsize=8,
                    alpha=0.8)

    ax1.set_xlabel('Retrieval Latency (ms)')
    ax1.set_ylabel('Hit@10 (%)')
    ax1.set_title('(a) Accuracy-Latency Pareto Front', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # (b) Feature Extraction Time
    # =========================================================================
    ax2 = axes[0, 1]

    feat_times = []
    method_names_list = []

    for method, data in efficiency_results.items():
        feat_time = data.get('feature_extract_time_ms', 0)
        if isinstance(feat_time, dict):
            feat_time = feat_time.get('mean', 0)
        feat_times.append(feat_time)
        method_names_list.append(get_method_name(method))

    # Sort by time
    sorted_idx = np.argsort(feat_times)
    feat_times = [feat_times[i] for i in sorted_idx]
    method_names_list = [method_names_list[i] for i in sorted_idx]
    methods_sorted = [methods[i] for i in sorted_idx]

    y_pos = np.arange(len(method_names_list))
    colors = [get_method_color(m) for m in methods_sorted]

    bars = ax2.barh(y_pos, feat_times, color=colors)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(method_names_list, fontsize=9)
    ax2.set_xlabel('Feature Extraction Time (ms)')
    ax2.set_title('(b) Feature Extraction Time', fontweight='bold')

    # Add value labels
    for i, val in enumerate(feat_times):
        ax2.text(val + 0.2, i, f'{val:.1f}', va='center', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='x')

    # =========================================================================
    # (c) Throughput (QPS)
    # =========================================================================
    ax3 = axes[1, 0]

    throughputs = []
    method_names_qps = []

    for method, data in efficiency_results.items():
        qps = data.get('throughput_qps', 0)
        if isinstance(qps, dict):
            qps = qps.get('mean', 0)
        throughputs.append(qps)
        method_names_qps.append(get_method_name(method))

    # Sort by QPS (descending)
    sorted_idx = np.argsort(throughputs)[::-1]
    throughputs = [throughputs[i] for i in sorted_idx]
    method_names_qps = [method_names_qps[i] for i in sorted_idx]
    methods_sorted_qps = [methods[i] for i in sorted_idx]

    y_pos = np.arange(len(method_names_qps))
    colors = [get_method_color(m) for m in methods_sorted_qps]

    bars = ax3.barh(y_pos, throughputs, color=colors)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(method_names_qps, fontsize=9)
    ax3.set_xlabel('Throughput (Queries/Second)')
    ax3.set_title('(c) Query Throughput', fontweight='bold')

    # Add value labels
    for i, val in enumerate(throughputs):
        ax3.text(val + 1, i, f'{val:.1f}', va='center', fontsize=9)

    ax3.grid(True, alpha=0.3, axis='x')

    # =========================================================================
    # (d) Memory Footprint
    # =========================================================================
    ax4 = axes[1, 1]

    memories = []
    method_names_mem = []

    for method, data in efficiency_results.items():
        mem = data.get('gallery_memory_mb', 0)
        if isinstance(mem, dict):
            mem = mem.get('mean', 0)
        memories.append(mem)
        method_names_mem.append(get_method_name(method))

    # Sort by memory
    sorted_idx = np.argsort(memories)
    memories = [memories[i] for i in sorted_idx]
    method_names_mem = [method_names_mem[i] for i in sorted_idx]
    methods_sorted_mem = [methods[i] for i in sorted_idx]

    y_pos = np.arange(len(method_names_mem))
    colors = [get_method_color(m) for m in methods_sorted_mem]

    bars = ax4.barh(y_pos, memories, color=colors)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(method_names_mem, fontsize=9)
    ax4.set_xlabel('Gallery Memory (MB)')
    ax4.set_title('(d) Memory Footprint', fontweight='bold')

    # Add value labels
    for i, val in enumerate(memories):
        ax4.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9)

    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 6: Efficiency...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig6_efficiency')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
