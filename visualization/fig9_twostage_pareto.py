"""
Figure 9: Two-Stage Retrieval Pareto Analysis

This figure shows the accuracy-latency trade-off for two-stage retrieval
with different candidate sizes (N), compared to baseline methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS
    from .data_loader import load_twostage_results, load_efficiency_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS
    from data_loader import load_twostage_results, load_efficiency_results


def create_figure():
    """Create the two-stage Pareto analysis figure."""

    setup_style()

    # Load data
    try:
        twostage_results = load_twostage_results()
    except FileNotFoundError:
        print("Warning: Two-stage results not found")
        return None

    try:
        efficiency_results = load_efficiency_results()
    except FileNotFoundError:
        print("Warning: Efficiency results not found")
        efficiency_results = {'methods': []}

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Two-Stage Retrieval: Accuracy-Latency Trade-off', fontsize=14, fontweight='bold', y=1.02)

    # Color scheme
    color_twostage = '#4C72B0'  # Blue
    color_m1 = '#55A868'  # Green
    color_m5 = '#C44E52'  # Red
    color_pareto = '#8172B2'  # Purple

    # =========================================================================
    # (a) Accuracy vs Latency Pareto Front
    # =========================================================================
    ax1 = axes[0]

    # Collect all points
    points = []
    labels = []
    colors = []
    markers = []

    # Add two-stage points
    for key, data in twostage_results.items():
        if key.startswith('TwoStage_N'):
            n = data.get('n', int(key.replace('TwoStage_N', '')))
            hit10 = data.get('mean', {}).get('hit@10', 0) * 100
            time_ms = data.get('mean', {}).get('avg_query_time_ms', 0)
            points.append((time_ms, hit10, n))
            labels.append(f'N={n}')
            colors.append(color_twostage)
            markers.append('o')

    # Add baselines
    if 'M1_baseline' in twostage_results:
        m1 = twostage_results['M1_baseline']
        hit10 = m1.get('mean', {}).get('hit@10', 0) * 100
        time_ms = m1.get('mean', {}).get('avg_query_time_ms', 0)
        points.append((time_ms, hit10, 0))
        labels.append('M1 (MFCC)')
        colors.append(color_m1)
        markers.append('s')

    if 'M5_baseline' in twostage_results:
        m5 = twostage_results['M5_baseline']
        hit10 = m5.get('mean', {}).get('hit@10', 0) * 100
        time_ms = m5.get('mean', {}).get('avg_query_time_ms', 0)
        points.append((time_ms, hit10, 1600))
        labels.append('M5 (DTW)')
        colors.append(color_m5)
        markers.append('^')

    # Plot all points
    for i, (time_ms, hit10, n) in enumerate(points):
        ax1.scatter(time_ms, hit10, c=colors[i], s=150, marker=markers[i],
                   edgecolors='black', linewidths=1.5, zorder=3, label=labels[i])

    # Connect two-stage points with line
    twostage_points = [(p[0], p[1]) for p, l in zip(points, labels) if l.startswith('N=')]
    if twostage_points:
        twostage_points.sort(key=lambda x: x[0])  # Sort by time
        times, hits = zip(*twostage_points)
        ax1.plot(times, hits, '--', color=color_twostage, alpha=0.5, linewidth=2)

    # Add labels with N values
    for i, (time_ms, hit10, n) in enumerate(points):
        offset_y = 1.5 if n not in [0, 1600] else -2
        offset_x = 2 if n not in [0, 1600] else 0
        ax1.annotate(labels[i], (time_ms, hit10),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, ha='left' if n not in [0, 1600] else 'center')

    ax1.set_xlabel('Query Time (ms)', fontsize=11)
    ax1.set_ylabel('Hit@10 (%)', fontsize=11)
    ax1.set_title('(a) Accuracy vs Latency', fontweight='bold')

    # Add regions annotation
    ax1.axhline(y=70, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=30, color='gray', linestyle=':', alpha=0.5)

    # Add annotations for regions
    ax1.text(25, 72, 'Target Region:\nHigh Accuracy, Low Latency', fontsize=9,
            color='green', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(15, 90)
    ax1.set_ylim(63, 75)

    # =========================================================================
    # (b) Speedup vs Accuracy Retention
    # =========================================================================
    ax2 = axes[1]

    # Calculate speedup and accuracy retention relative to M5 baseline
    m5_time = twostage_results.get('M5_baseline', {}).get('mean', {}).get('avg_query_time_ms', 61.19)
    m5_hit10 = twostage_results.get('M5_baseline', {}).get('mean', {}).get('hit@10', 0.7045) * 100

    n_values = []
    speedups = []
    retentions = []

    for key, data in twostage_results.items():
        if key.startswith('TwoStage_N'):
            n = data.get('n', int(key.replace('TwoStage_N', '')))
            hit10 = data.get('mean', {}).get('hit@10', 0) * 100
            time_ms = data.get('mean', {}).get('avg_query_time_ms', 0)

            speedup = m5_time / time_ms if time_ms > 0 else 0
            retention = hit10 / m5_hit10 * 100 if m5_hit10 > 0 else 0

            n_values.append(n)
            speedups.append(speedup)
            retentions.append(retention)

    # Sort by N
    sorted_idx = np.argsort(n_values)
    n_values = [n_values[i] for i in sorted_idx]
    speedups = [speedups[i] for i in sorted_idx]
    retentions = [retentions[i] for i in sorted_idx]

    # Create bar chart
    x = np.arange(len(n_values))
    width = 0.35

    bars1 = ax2.bar(x - width/2, speedups, width, label='Speedup (×)', color=color_twostage)
    ax2.set_ylabel('Speedup (×)', color=color_twostage)
    ax2.tick_params(axis='y', labelcolor=color_twostage)

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, retentions, width, label='Accuracy Retention (%)',
                        color=color_pareto, alpha=0.7)
    ax2_twin.set_ylabel('Accuracy Retention (%)', color=color_pareto)
    ax2_twin.tick_params(axis='y', labelcolor=color_pareto)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'N={n}' for n in n_values], fontsize=9, rotation=45)
    ax2.set_xlabel('Candidate Size (N)')
    ax2.set_title('(b) Speedup vs Accuracy Retention (vs M5 DTW)', fontweight='bold')

    # Add value labels
    for i, (sp, ret) in enumerate(zip(speedups, retentions)):
        ax2.text(i - width/2, sp + 0.05, f'{sp:.2f}×', ha='center', fontsize=8, fontweight='bold')
        ax2_twin.text(i + width/2, ret + 0.5, f'{ret:.1f}%', ha='center', fontsize=8)

    # Add baseline reference lines
    ax2.axhline(y=1, color=color_m5, linestyle='--', alpha=0.5, label='M5 Baseline (1×)')
    ax2_twin.axhline(y=100, color=color_pareto, linestyle='--', alpha=0.5)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='y')
    ax2_twin.set_ylim(95, 105)

    # Add insight box
    best_n_idx = np.argmax([s * r/100 for s, r in zip(speedups, retentions)])
    best_n = n_values[best_n_idx]
    best_speedup = speedups[best_n_idx]
    best_retention = retentions[best_n_idx]

    insight_text = (f"Best Trade-off: N={best_n}\n"
                   f"Speedup: {best_speedup:.2f}×\n"
                   f"Accuracy: {best_retention:.1f}%")
    ax2.text(0.02, 0.98, insight_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 9: Two-Stage Pareto...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig9_twostage_pareto')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
