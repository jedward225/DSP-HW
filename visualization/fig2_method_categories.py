"""
Figure 2: Traditional vs Deep Learning vs Pretrained Comparison

This figure contains:
(a) Grouped Bar Chart - Compare 3 categories by metrics
(b) Performance Gap Analysis - Improvement from Traditional → Deep → Pretrained
(c) Method Ranking Heatmap - All methods ranked by different metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import (
        setup_style, save_figure, CATEGORY_COLORS, METHOD_NAMES,
        get_method_name, get_method_color,
        ci_half_width_from_ci, ci_half_width_from_std
    )
    from .data_loader import load_all_method_results
except ImportError:
    from config import (
        setup_style, save_figure, CATEGORY_COLORS, METHOD_NAMES,
        get_method_name, get_method_color,
        ci_half_width_from_ci, ci_half_width_from_std
    )
    from data_loader import load_all_method_results


def create_figure():
    """Create the method categories comparison figure."""

    setup_style()

    # Load data
    results = load_all_method_results()

    # Organize by category
    categories = {'traditional': [], 'deep': [], 'pretrained': []}
    for method, data in results.items():
        cat = data.get('category', 'traditional')
        categories[cat].append((method, data))

    # Sort by hit@10 within each category
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1]['mean'].get('hit@10', 0), reverse=True)

    # Get best method from each category
    best_methods = {}
    for cat in categories:
        if categories[cat]:
            best_methods[cat] = categories[cat][0]

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Method Category Comparison', fontsize=14, fontweight='bold', y=1.02)

    # =========================================================================
    # (a) Grouped Bar Chart - Best from each category
    # =========================================================================
    ax1 = axes[0]

    metrics = ['hit@1', 'hit@10', 'mrr@10', 'map@10']
    metric_labels = ['Hit@1', 'Hit@10', 'MRR@10', 'MAP@10']
    x = np.arange(len(metrics))
    width = 0.25

    cat_order = ['traditional', 'deep', 'pretrained']
    cat_labels = ['Traditional\n(M5: DTW)', 'Deep Learning\n(Contrastive)', 'Pretrained\n(CLAP)']

    for i, cat in enumerate(cat_order):
        if cat in best_methods:
            method, data = best_methods[cat]
            values = [data['mean'].get(m, 0) * 100 for m in metrics]
            errors = []
            for m in metrics:
                ci_entry = data.get('ci', {}).get(m)
                if ci_entry:
                    errors.append(ci_half_width_from_ci(ci_entry) * 100)
                else:
                    errors.append(ci_half_width_from_std(data.get('std', {}).get(m, 0)) * 100)
            bars = ax1.bar(x + i * width, values, width, yerr=errors,
                          label=cat_labels[i], color=CATEGORY_COLORS[cat],
                          capsize=3, error_kw={'linewidth': 1})

    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('(a) Best Method per Category', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (b) Performance Gap Analysis
    # =========================================================================
    ax2 = axes[1]

    # Get hit@10 for best methods
    hit10_values = {}
    for cat in cat_order:
        if cat in best_methods:
            hit10_values[cat] = best_methods[cat][1]['mean'].get('hit@10', 0) * 100

    # Calculate gaps
    trad_val = hit10_values.get('traditional', 0)
    deep_val = hit10_values.get('deep', 0)
    pretrained_val = hit10_values.get('pretrained', 0)

    gap_deep = deep_val - trad_val
    gap_pretrained = pretrained_val - deep_val

    # Stacked bar showing cumulative improvement
    x_pos = [0]
    bars1 = ax2.bar(x_pos, [trad_val], width=0.5, label='Traditional Baseline',
                    color=CATEGORY_COLORS['traditional'])
    bars2 = ax2.bar(x_pos, [gap_deep], width=0.5, bottom=[trad_val],
                    label=f'Deep Learning (+{gap_deep:.1f}%)',
                    color=CATEGORY_COLORS['deep'])
    bars3 = ax2.bar(x_pos, [gap_pretrained], width=0.5, bottom=[trad_val + gap_deep],
                    label=f'Pretrained (+{gap_pretrained:.1f}%)',
                    color=CATEGORY_COLORS['pretrained'])

    ax2.set_ylabel('Hit@10 (%)')
    ax2.set_title('(b) Performance Improvement', fontweight='bold')
    ax2.set_xticks([])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper left', fontsize=9)

    # Add annotations
    ax2.annotate(f'{trad_val:.1f}%', xy=(0, trad_val/2), ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    if gap_deep > 5:
        ax2.annotate(f'+{gap_deep:.1f}%', xy=(0, trad_val + gap_deep/2), ha='center',
                    va='center', fontsize=11, fontweight='bold', color='white')
    if gap_pretrained > 5:
        ax2.annotate(f'+{gap_pretrained:.1f}%', xy=(0, trad_val + gap_deep + gap_pretrained/2),
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (c) Method Ranking Heatmap
    # =========================================================================
    ax3 = axes[2]

    # Collect all methods and metrics
    all_methods = []
    for cat in cat_order:
        all_methods.extend(categories[cat])

    # Sort by hit@10
    all_methods.sort(key=lambda x: x[1]['mean'].get('hit@10', 0), reverse=True)

    ranking_metrics = ['hit@1', 'hit@10', 'mrr@10', 'map@10', 'ndcg@10']
    ranking_labels = ['Hit@1', 'Hit@10', 'MRR@10', 'MAP@10', 'NDCG@10']

    # Create heatmap data
    heatmap_data = []
    method_labels = []
    for method, data in all_methods:
        row = [data['mean'].get(m, 0) * 100 for m in ranking_metrics]
        heatmap_data.append(row)
        method_labels.append(get_method_name(method))

    heatmap_array = np.array(heatmap_data)

    # Plot heatmap
    im = ax3.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax3.set_xticks(np.arange(len(ranking_labels)))
    ax3.set_yticks(np.arange(len(method_labels)))
    ax3.set_xticklabels(ranking_labels, fontsize=9)
    ax3.set_yticklabels(method_labels, fontsize=9)

    # Add text annotations
    for i in range(len(method_labels)):
        for j in range(len(ranking_labels)):
            value = heatmap_array[i, j]
            text_color = 'white' if value > 50 else 'black'
            ax3.text(j, i, f'{value:.1f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    ax3.set_title('(c) Method Performance Heatmap', fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Score (%)', fontsize=10)

    # Add category indicators on the left
    cat_colors = []
    for method, data in all_methods:
        cat_colors.append(CATEGORY_COLORS[data.get('category', 'traditional')])

    for i, color in enumerate(cat_colors):
        ax3.add_patch(plt.Rectangle((-0.7, i - 0.5), 0.15, 1,
                                     facecolor=color, edgecolor='none',
                                     clip_on=False))

    plt.tight_layout()

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 2: Method Categories...")
    fig = create_figure()
    save_figure(fig, 'fig2_method_categories')
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
