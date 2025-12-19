"""
Figure 10: Cross-Fold Variance Analysis

This figure shows the performance variance across different folds
for all retrieval methods.
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
        get_method_color, get_method_name
    )
    from .data_loader import load_all_method_results
except ImportError:
    from config import (
        setup_style, save_figure, METHOD_COLORS, METHOD_NAMES,
        get_method_color, get_method_name
    )
    from data_loader import load_all_method_results


def create_figure():
    """Create the fold variance analysis figure."""

    setup_style()

    # Deterministic jitter for reproducible scatter placement
    rng = np.random.default_rng(42)

    # Load data
    try:
        method_results = load_all_method_results()
    except FileNotFoundError:
        print("Warning: Method results not found")
        return None

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Cross-Fold Variance Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Filter methods with per-fold data
    methods_with_folds = {}
    for method, data in method_results.items():
        if 'folds' in data and len(data['folds']) > 0:
            methods_with_folds[method] = data

    if not methods_with_folds:
        print("Warning: No per-fold data available")
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No per-fold data available',
                   ha='center', va='center', transform=ax.transAxes)
        return fig

    # =========================================================================
    # (a) Hit@10 Box Plot Across Folds
    # =========================================================================
    ax1 = axes[0, 0]

    method_names_list = []
    fold_data_hit10 = []
    colors = []

    for method, data in methods_with_folds.items():
        folds = data.get('folds', {})
        if folds:
            fold_values = [folds[f].get('hit@10', 0) * 100 for f in sorted(folds.keys())]
            method_names_list.append(get_method_name(method))
            fold_data_hit10.append(fold_values)
            colors.append(get_method_color(method))

    # Create box plot
    bp = ax1.boxplot(fold_data_hit10, labels=method_names_list, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add scatter points for individual folds
    for i, (data, color) in enumerate(zip(fold_data_hit10, colors)):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax1.scatter(x, data, alpha=0.6, c=color, edgecolors='black', s=50, zorder=3)

    ax1.set_ylabel('Hit@10 (%)')
    ax1.set_title('(a) Hit@10 Variance Across Folds', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (b) Hit@1 Box Plot Across Folds
    # =========================================================================
    ax2 = axes[0, 1]

    fold_data_hit1 = []
    for method, data in methods_with_folds.items():
        folds = data.get('folds', {})
        if folds:
            fold_values = [folds[f].get('hit@1', 0) * 100 for f in sorted(folds.keys())]
            fold_data_hit1.append(fold_values)

    bp2 = ax2.boxplot(fold_data_hit1, labels=method_names_list, patch_artist=True)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (data, color) in enumerate(zip(fold_data_hit1, colors)):
        x = rng.normal(i + 1, 0.04, size=len(data))
        ax2.scatter(x, data, alpha=0.6, c=color, edgecolors='black', s=50, zorder=3)

    ax2.set_ylabel('Hit@1 (%)')
    ax2.set_title('(b) Hit@1 Variance Across Folds', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (c) Coefficient of Variation Comparison
    # =========================================================================
    ax3 = axes[1, 0]

    cv_hit10 = []
    cv_hit1 = []

    for method, data in methods_with_folds.items():
        mean_h10 = data.get('mean', {}).get('hit@10', 0) * 100
        std_h10 = data.get('std', {}).get('hit@10', 0) * 100
        cv_h10 = (std_h10 / mean_h10 * 100) if mean_h10 > 0 else 0
        cv_hit10.append(cv_h10)

        mean_h1 = data.get('mean', {}).get('hit@1', 0) * 100
        std_h1 = data.get('std', {}).get('hit@1', 0) * 100
        cv_h1 = (std_h1 / mean_h1 * 100) if mean_h1 > 0 else 0
        cv_hit1.append(cv_h1)

    x = np.arange(len(method_names_list))
    width = 0.35

    bars1 = ax3.bar(x - width/2, cv_hit10, width, label='Hit@10 CV', color=colors)
    bars2 = ax3.bar(x + width/2, cv_hit1, width, label='Hit@1 CV', color=colors, alpha=0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(method_names_list, fontsize=9, rotation=45)
    ax3.set_ylabel('Coefficient of Variation (%)')
    ax3.set_title('(c) Stability: Coefficient of Variation', fontweight='bold')

    # Add value labels
    for i, (cv10, cv1) in enumerate(zip(cv_hit10, cv_hit1)):
        ax3.text(i - width/2, cv10 + 0.2, f'{cv10:.1f}%', ha='center', fontsize=8)
        ax3.text(i + width/2, cv1 + 0.2, f'{cv1:.1f}%', ha='center', fontsize=8)

    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    avg_cv = np.mean(cv_hit10)
    ax3.axhline(y=avg_cv, color='red', linestyle='--', alpha=0.5, label=f'Avg CV: {avg_cv:.1f}%')

    # =========================================================================
    # (d) Fold-by-Fold Performance Heatmap
    # =========================================================================
    ax4 = axes[1, 1]

    # Create heatmap data
    fold_names = sorted(list(methods_with_folds.values())[0].get('folds', {}).keys())
    heatmap_data = np.zeros((len(methods_with_folds), len(fold_names)))

    for i, (method, data) in enumerate(methods_with_folds.items()):
        folds = data.get('folds', {})
        for j, fold in enumerate(fold_names):
            if fold in folds:
                heatmap_data[i, j] = folds[fold].get('hit@10', 0) * 100

    # Plot heatmap
    im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax4.set_xticks(np.arange(len(fold_names)))
    ax4.set_yticks(np.arange(len(methods_with_folds)))
    ax4.set_xticklabels([f'Fold {f}' for f in fold_names], fontsize=10)
    ax4.set_yticklabels(method_names_list, fontsize=9)

    ax4.set_xlabel('Fold')
    ax4.set_title('(d) Hit@10 Heatmap by Method × Fold', fontweight='bold')

    # Add text annotations
    for i in range(len(methods_with_folds)):
        for j in range(len(fold_names)):
            value = heatmap_data[i, j]
            text_color = 'white' if value > np.mean(heatmap_data) else 'black'
            ax4.text(j, i, f'{value:.1f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Hit@10 (%)', fontsize=10)

    # Highlight best fold for each method
    best_folds = np.argmax(heatmap_data, axis=1)
    for i, bf in enumerate(best_folds):
        ax4.add_patch(plt.Rectangle((bf - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='green', linewidth=2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 10: Fold Variance...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig10_fold_variance')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
