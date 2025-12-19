"""
Figure 7: Fusion Methods & Two-Stage Retrieval (Enhanced)

This figure contains:
(a) Late Fusion Weight Analysis
(b) Fusion Methods Comparison (RRF vs Late Fusion)
(c) Two-Stage N Sweep
(d) Individual vs Fused Performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from .data_loader import load_fusion_results, load_twostage_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from data_loader import load_fusion_results, load_twostage_results


def create_figure():
    """Create the enhanced fusion and two-stage retrieval figure."""

    setup_style()

    # Load data
    try:
        fusion_results = load_fusion_results()
    except FileNotFoundError:
        print("Warning: Fusion results not found")
        fusion_results = {}

    try:
        twostage_results = load_twostage_results()
    except FileNotFoundError:
        print("Warning: Two-stage results not found")
        twostage_results = {}

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Fusion Methods & Two-Stage Retrieval', fontsize=14, fontweight='bold', y=0.98)

    # Color scheme
    colors_fusion = '#4C72B0'  # Blue
    colors_rrf = '#55A868'  # Green
    colors_individual = '#C44E52'  # Red
    colors_twostage = '#8172B2'  # Purple

    # =========================================================================
    # (a) Late Fusion Weight Analysis
    # =========================================================================
    ax1 = axes[0, 0]

    if 'late_fusion' in fusion_results:
        late_fusion = fusion_results['late_fusion']
        all_weights = late_fusion.get('all_weights', [])

        if all_weights:
            # Extract data
            weight_labels = []
            hit10_values = []

            for item in all_weights:
                weights = item.get('weights', {})
                # Create label from weights
                w_strs = [f"{k.split('_')[0]}:{v:.2f}" for k, v in weights.items()]
                label = ', '.join(w_strs)
                weight_labels.append(label)
                hit10_values.append(item.get('mean', {}).get('hit@10', 0) * 100)

            # Sort by hit@10
            sorted_idx = np.argsort(hit10_values)[::-1]
            weight_labels = [weight_labels[i] for i in sorted_idx[:10]]  # Top 10
            hit10_values = [hit10_values[i] for i in sorted_idx[:10]]

            y_pos = np.arange(len(weight_labels))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weight_labels)))

            bars = ax1.barh(y_pos, hit10_values, color=colors)

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(weight_labels, fontsize=8)
            ax1.set_xlabel('Hit@10 (%)')
            ax1.set_title('(a) Late Fusion Weight Combinations', fontweight='bold')

            # Add value labels
            for i, val in enumerate(hit10_values):
                ax1.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)

            # Highlight best
            bars[0].set_edgecolor('green')
            bars[0].set_linewidth(2)

            ax1.set_xlim(0, max(hit10_values) + 5)
            ax1.grid(True, alpha=0.3, axis='x')

        else:
            ax1.text(0.5, 0.5, 'No fusion weight data', ha='center', va='center',
                    transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'No late fusion data available', ha='center', va='center',
                transform=ax1.transAxes)
        ax1.set_title('(a) Late Fusion Weight Analysis', fontweight='bold')

    # =========================================================================
    # (b) Fusion Methods Comparison (RRF vs Late Fusion vs Individual)
    # =========================================================================
    ax2 = axes[0, 1]

    methods_compare = []
    hit1_values = []
    hit10_values = []
    mrr_values = []
    bar_colors = []

    # Get best individual method
    individual = fusion_results.get('individual_methods', {})
    if individual:
        best_ind_hit10 = 0
        best_ind_name = ''
        for name, data in individual.items():
            h10 = data.get('mean', {}).get('hit@10', 0) * 100
            if h10 > best_ind_hit10:
                best_ind_hit10 = h10
                best_ind_name = name
                best_ind_hit1 = data.get('mean', {}).get('hit@1', 0) * 100
                best_ind_mrr = data.get('mean', {}).get('mrr@10', 0) * 100

        methods_compare.append(f'Best Individual\n({best_ind_name})')
        hit1_values.append(best_ind_hit1)
        hit10_values.append(best_ind_hit10)
        mrr_values.append(best_ind_mrr)
        bar_colors.append(colors_individual)

    # Get late fusion best
    if 'late_fusion' in fusion_results:
        best_fusion = fusion_results['late_fusion'].get('best', {})
        if best_fusion:
            methods_compare.append('Late Fusion\n(Best Weights)')
            hit1_values.append(best_fusion.get('mean', {}).get('hit@1', 0) * 100)
            hit10_values.append(best_fusion.get('mean', {}).get('hit@10', 0) * 100)
            mrr_values.append(best_fusion.get('mean', {}).get('mrr@10', 0) * 100)
            bar_colors.append(colors_fusion)

    # Get RRF results
    if 'rank_fusion' in fusion_results:
        rrf = fusion_results['rank_fusion']
        methods_compare.append('Rank Fusion\n(RRF, k=60)')
        hit1_values.append(rrf.get('mean', {}).get('hit@1', 0) * 100)
        hit10_values.append(rrf.get('mean', {}).get('hit@10', 0) * 100)
        mrr_values.append(rrf.get('mean', {}).get('mrr@10', 0) * 100)
        bar_colors.append(colors_rrf)

    if methods_compare:
        x = np.arange(len(methods_compare))
        width = 0.25

        bars1 = ax2.bar(x - width, hit1_values, width, label='Hit@1', color=bar_colors, alpha=0.6)
        bars2 = ax2.bar(x, hit10_values, width, label='Hit@10', color=bar_colors)
        bars3 = ax2.bar(x + width, mrr_values, width, label='MRR@10', color=bar_colors, alpha=0.4)

        ax2.set_xticks(x)
        ax2.set_xticklabels(methods_compare, fontsize=9)
        ax2.set_ylabel('Score (%)')
        ax2.set_title('(b) Fusion Methods Comparison', fontweight='bold')

        # Add value labels on Hit@10 bars
        for i, val in enumerate(hit10_values):
            ax2.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_ylim(0, max(hit10_values) + 10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Annotate improvement
        if len(hit10_values) >= 2:
            improvement = hit10_values[-1] - hit10_values[0]
            if improvement > 0:
                ax2.annotate(f'+{improvement:.1f}%',
                            xy=(len(methods_compare)-1, hit10_values[-1]),
                            xytext=(len(methods_compare)-0.5, hit10_values[-1] + 5),
                            fontsize=10, color='green', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='green'))
    else:
        ax2.text(0.5, 0.5, 'No fusion comparison data', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('(b) Fusion Methods Comparison', fontweight='bold')

    # =========================================================================
    # (c) Two-Stage N Sweep
    # =========================================================================
    ax3 = axes[1, 0]

    if twostage_results:
        # Extract N values and corresponding metrics
        n_values = []
        hit10_values = []
        hit10_stds = []
        query_times = []

        # Parse results
        for key, data in twostage_results.items():
            if key.startswith('TwoStage_N'):
                n = data.get('n', int(key.replace('TwoStage_N', '')))
                n_values.append(n)
                hit10_values.append(data.get('mean', {}).get('hit@10', 0) * 100)
                hit10_stds.append(ci_half_width_from_std(data.get('std', {}).get('hit@10', 0)) * 100)
                query_times.append(data.get('mean', {}).get('avg_query_time_ms', 0))

        # Also add baseline if available
        baseline_hit10 = None
        baseline_time = None
        if 'M5_baseline' in twostage_results:
            baseline = twostage_results['M5_baseline']
            baseline_hit10 = baseline.get('mean', {}).get('hit@10', 0) * 100
            baseline_time = baseline.get('mean', {}).get('avg_query_time_ms', 0)

        # Sort by N
        sorted_idx = np.argsort(n_values)
        n_values = [n_values[i] for i in sorted_idx]
        hit10_values = [hit10_values[i] for i in sorted_idx]
        hit10_stds = [hit10_stds[i] for i in sorted_idx]
        query_times = [query_times[i] for i in sorted_idx]

        # Create dual y-axis plot
        color1 = CATEGORY_COLORS['traditional']
        color2 = colors_twostage

        # Plot Hit@10
        line1 = ax3.errorbar(n_values, hit10_values, yerr=hit10_stds,
                            fmt='o-', color=color1, linewidth=2, markersize=8,
                            capsize=4, label='Hit@10')
        ax3.fill_between(n_values,
                        np.array(hit10_values) - np.array(hit10_stds),
                        np.array(hit10_values) + np.array(hit10_stds),
                        alpha=0.2, color=color1)

        ax3.set_xlabel('Number of Candidates (N)')
        ax3.set_ylabel('Hit@10 (%)', color=color1)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.set_xscale('log')

        # Add baseline line if available
        if baseline_hit10 is not None:
            ax3.axhline(y=baseline_hit10, color=color1, linestyle='--', alpha=0.5,
                       label=f'DTW Baseline ({baseline_hit10:.1f}%)')

        # Second y-axis for query time
        ax3_twin = ax3.twinx()
        line2 = ax3_twin.plot(n_values, query_times, 's--', color=color2,
                             linewidth=2, markersize=8, label='Query Time')
        ax3_twin.set_ylabel('Query Time (ms)', color=color2)
        ax3_twin.tick_params(axis='y', labelcolor=color2)

        # Add baseline time line
        if baseline_time is not None:
            ax3_twin.axhline(y=baseline_time, color=color2, linestyle='--', alpha=0.5,
                            label=f'DTW Baseline ({baseline_time:.1f}ms)')

        ax3.set_title('(c) Two-Stage Retrieval: N Candidates Sweep', fontweight='bold')

        # Combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

        ax3.grid(True, alpha=0.3)

    else:
        ax3.text(0.5, 0.5, 'No two-stage data available', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('(c) Two-Stage N Sweep', fontweight='bold')

    # =========================================================================
    # (d) Individual Methods vs Fused Performance
    # =========================================================================
    ax4 = axes[1, 1]

    if individual:
        method_names = []
        ind_hit10 = []
        ind_hit1 = []

        for name, data in individual.items():
            method_names.append(name.replace('_', '\n'))
            ind_hit10.append(data.get('mean', {}).get('hit@10', 0) * 100)
            ind_hit1.append(data.get('mean', {}).get('hit@1', 0) * 100)

        x = np.arange(len(method_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, ind_hit1, width, label='Hit@1', color=colors_individual, alpha=0.7)
        bars2 = ax4.bar(x + width/2, ind_hit10, width, label='Hit@10', color=colors_individual)

        # Add fusion lines for reference
        if 'late_fusion' in fusion_results:
            best_fusion = fusion_results['late_fusion'].get('best', {})
            if best_fusion:
                fusion_hit10 = best_fusion.get('mean', {}).get('hit@10', 0) * 100
                fusion_hit1 = best_fusion.get('mean', {}).get('hit@1', 0) * 100
                ax4.axhline(y=fusion_hit10, color=colors_fusion, linestyle='-', linewidth=2,
                           label=f'Late Fusion Hit@10 ({fusion_hit10:.1f}%)')
                ax4.axhline(y=fusion_hit1, color=colors_fusion, linestyle='--', linewidth=2,
                           label=f'Late Fusion Hit@1 ({fusion_hit1:.1f}%)', alpha=0.7)

        if 'rank_fusion' in fusion_results:
            rrf = fusion_results['rank_fusion']
            rrf_hit10 = rrf.get('mean', {}).get('hit@10', 0) * 100
            ax4.axhline(y=rrf_hit10, color=colors_rrf, linestyle='-', linewidth=2,
                       label=f'RRF Hit@10 ({rrf_hit10:.1f}%)')

        ax4.set_xticks(x)
        ax4.set_xticklabels(method_names, fontsize=9)
        ax4.set_ylabel('Score (%)')
        ax4.set_title('(d) Individual Methods vs Fusion Baselines', fontweight='bold')

        # Add value labels
        for i, (h1, h10) in enumerate(zip(ind_hit1, ind_hit10)):
            ax4.text(i - width/2, h1 + 1, f'{h1:.0f}', ha='center', fontsize=8)
            ax4.text(i + width/2, h10 + 1, f'{h10:.0f}', ha='center', fontsize=8)

        ax4.legend(loc='upper right', fontsize=8)
        ax4.set_ylim(0, max(ind_hit10) + 15)
        ax4.grid(True, alpha=0.3, axis='y')

    else:
        ax4.text(0.5, 0.5, 'No individual method data', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('(d) Individual Methods vs Fused', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 7: Fusion & Two-Stage (Enhanced)...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig7_fusion_twostage')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
