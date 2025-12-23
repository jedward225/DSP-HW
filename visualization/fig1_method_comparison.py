"""
Figure 1: Overall Method Performance Comparison

This figure contains:
(a) Hit@K Curves - Line plot showing Hit@1, Hit@5, Hit@10, Hit@20 for all methods
(b) Precision@K Curves - Line plot for Precision metrics
(c) Bar Chart with Error Bars - Hit@10 comparison with 95% CI
(d) Radar Chart - Multi-metric comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import (
        setup_style, save_figure, FIGURE_SIZES, METHOD_COLORS,
        METHOD_NAMES, CATEGORY_COLORS, get_method_color, get_method_name,
        ci_half_width_from_ci, ci_half_width_from_std
    )
    from .data_loader import load_all_method_results
except ImportError:
    from config import (
        setup_style, save_figure, FIGURE_SIZES, METHOD_COLORS,
        METHOD_NAMES, CATEGORY_COLORS, get_method_color, get_method_name,
        ci_half_width_from_ci, ci_half_width_from_std
    )
    from data_loader import load_all_method_results


def create_figure():
    """Create the method comparison figure with 4 subplots."""

    setup_style()

    # Load data
    results = load_all_method_results()

    # Sort methods by category and performance
    methods_by_category = {'traditional': [], 'deep': [], 'pretrained': []}
    for method, data in results.items():
        cat = data.get('category', 'traditional')
        methods_by_category[cat].append((method, data))

    # Sort within each category by hit@10
    for cat in methods_by_category:
        methods_by_category[cat].sort(key=lambda x: x[1]['mean'].get('hit@10', 0), reverse=True)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Overall Method Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

    # =========================================================================
    # (a) Hit@K Curves
    # =========================================================================
    ax1 = axes[0, 0]
    k_values = [1, 5, 10, 20]

    # Plot each method
    for cat_order, cat in enumerate(['traditional', 'deep', 'pretrained']):
        for method, data in methods_by_category[cat]:
            hit_values = [data['mean'].get(f'hit@{k}', 0) * 100 for k in k_values]
            color = get_method_color(method)
            linestyle = '-' if cat == 'traditional' else ('--' if cat == 'deep' else '-.')
            marker = 'o' if cat == 'traditional' else ('s' if cat == 'deep' else '^')
            ax1.plot(k_values, hit_values, marker=marker, linestyle=linestyle,
                    color=color, label=get_method_name(method), markersize=6, linewidth=1.5)

    ax1.set_xlabel('K')
    ax1.set_ylabel('Hit@K (%)')
    ax1.set_title('(a) Hit@K Curves', fontweight='bold')
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # (b) Precision@K Curves
    # =========================================================================
    ax2 = axes[0, 1]

    for cat_order, cat in enumerate(['traditional', 'deep', 'pretrained']):
        for method, data in methods_by_category[cat]:
            prec_values = [data['mean'].get(f'precision@{k}', 0) * 100 for k in k_values]
            color = get_method_color(method)
            linestyle = '-' if cat == 'traditional' else ('--' if cat == 'deep' else '-.')
            marker = 'o' if cat == 'traditional' else ('s' if cat == 'deep' else '^')
            ax2.plot(k_values, prec_values, marker=marker, linestyle=linestyle,
                    color=color, label=get_method_name(method), markersize=6, linewidth=1.5)

    ax2.set_xlabel('K')
    ax2.set_ylabel('Precision@K (%)')
    ax2.set_title('(b) Precision@K Curves', fontweight='bold')
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # (c) Bar Chart with Error Bars - Hit@10
    # =========================================================================
    ax3 = axes[1, 0]

    # Collect all methods sorted by hit@10
    all_methods = []
    for cat in ['traditional', 'deep', 'pretrained']:
        all_methods.extend(methods_by_category[cat])

    # Sort by hit@10
    all_methods.sort(key=lambda x: x[1]['mean'].get('hit@10', 0))

    method_names_list = [get_method_name(m) for m, _ in all_methods]
    hit10_means = [d['mean'].get('hit@10', 0) * 100 for _, d in all_methods]

    # 95% CI half-width (prefer precomputed CI when available)
    hit10_stds = []
    for _, d in all_methods:
        ci_entry = d.get('ci', {}).get('hit@10')
        if ci_entry:
            hit10_stds.append(ci_half_width_from_ci(ci_entry) * 100)
        else:
            hit10_stds.append(ci_half_width_from_std(d.get('std', {}).get('hit@10', 0)) * 100)
    colors = [get_method_color(m) for m, _ in all_methods]

    y_pos = np.arange(len(method_names_list))
    bars = ax3.barh(y_pos, hit10_means, xerr=hit10_stds, color=colors,
                   capsize=3, error_kw={'linewidth': 1})

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(method_names_list)
    ax3.set_xlabel('Hit@10 (%)')
    ax3.set_title('(c) Hit@10 Comparison (95% CI)', fontweight='bold')
    ax3.set_xlim(0, 105)

    # Add value labels
    for i, (mean, std) in enumerate(zip(hit10_means, hit10_stds)):
        ax3.text(mean + std + 1, i, f'{mean:.1f}%', va='center', fontsize=8)

    # Add category legend
    cat_patches = [
        Patch(facecolor=CATEGORY_COLORS['traditional'], label='Traditional'),
        Patch(facecolor=CATEGORY_COLORS['deep'], label='Deep Learning'),
        Patch(facecolor=CATEGORY_COLORS['pretrained'], label='Pretrained'),
    ]
    ax3.legend(handles=cat_patches, loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')

    # =========================================================================
    # (d) Radar Chart - Multi-metric comparison (Top 5 methods)
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.remove()  # Remove regular axes
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')

    # Select top methods from each category
    top_methods = []
    if methods_by_category['traditional']:
        top_methods.append(methods_by_category['traditional'][0])  # Best traditional
    if methods_by_category['deep']:
        top_methods.append(methods_by_category['deep'][0])  # Best deep
    if methods_by_category['pretrained']:
        top_methods.append(methods_by_category['pretrained'][0])  # Best pretrained
        if len(methods_by_category['pretrained']) > 1:
            top_methods.append(methods_by_category['pretrained'][1])  # 2nd best pretrained

    # Metrics for radar chart
    radar_metrics = ['hit@10', 'mrr@10', 'map@10', 'ndcg@10']
    radar_labels = ['Hit@10', 'MRR@10', 'MAP@10', 'NDCG@10']
    num_vars = len(radar_metrics)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Plot each method
    for method, data in top_methods:
        values = [data['mean'].get(m, 0) * 100 for m in radar_metrics]
        values += values[:1]  # Complete the loop

        color = get_method_color(method)
        ax4.plot(angles, values, 'o-', linewidth=2, label=get_method_name(method),
                color=color, markersize=5)
        ax4.fill(angles, values, alpha=0.15, color=color)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_labels, fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.set_title('(d) Multi-Metric Comparison (Top Methods)', fontweight='bold', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 1: Method Comparison...")
    fig = create_figure()
    save_figure(fig, 'fig1_method_comparison')
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
