"""
Figure 4: Ablation Studies

This figure contains:
(a) Pre-emphasis Effect
(b) CMVN Mode Comparison
(c) Mel Formula Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from .data_loader import load_ablation_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from data_loader import load_ablation_results


def create_figure():
    """Create the ablation studies figure."""

    setup_style()

    # Load data
    try:
        ablation_results = load_ablation_results()
    except FileNotFoundError:
        print("Warning: Ablation results not found")
        return None

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ablation Studies: Component Impact Analysis', fontsize=14, fontweight='bold', y=1.02)

    # Define colors for ablation plots
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    # =========================================================================
    # (a) Pre-emphasis Effect
    # =========================================================================
    ax1 = axes[0]

    if 'preemphasis' in ablation_results:
        preemph_data = ablation_results['preemphasis']

        labels = []
        hit10_means = []
        hit10_stds = []

        for item in preemph_data:
            config = item.get('config', {})
            name = config.get('name', 'Unknown')
            if 'no' in name.lower():
                labels.append('No Pre-emphasis')
            else:
                coef = config.get('pre_emphasis_coef', 0.97)
                labels.append(f'Pre-emphasis ({coef})')

            hit10_means.append(item.get('mean', {}).get('hit@10', 0) * 100)
            hit10_stds.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)

        x = np.arange(len(labels))
        bars = ax1.bar(x, hit10_means, yerr=hit10_stds, color=colors[:len(labels)],
                      capsize=5, error_kw={'linewidth': 1.5})

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel('Hit@10 (%)')
        ax1.set_title('(a) Pre-emphasis Effect', fontweight='bold')

        # Add value labels
        for i, (mean, std) in enumerate(zip(hit10_means, hit10_stds)):
            ax1.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Calculate and show delta
        if len(hit10_means) >= 2:
            delta = hit10_means[1] - hit10_means[0]
            ax1.annotate(f'$\\Delta$ = {delta:+.1f}%',
                        xy=(0.5, max(hit10_means) * 0.5),
                        fontsize=12, ha='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax1.set_ylim(0, max(hit10_means) + max(hit10_stds) + 5)
        ax1.grid(True, alpha=0.3, axis='y')

    else:
        ax1.text(0.5, 0.5, 'No pre-emphasis data', ha='center', va='center',
                transform=ax1.transAxes)
        ax1.set_title('(a) Pre-emphasis Effect', fontweight='bold')

    # =========================================================================
    # (b) CMVN Mode Comparison
    # =========================================================================
    ax2 = axes[1]

    if 'cmvn' in ablation_results:
        cmvn_data = ablation_results['cmvn']

        labels = []
        hit10_means = []
        hit10_stds = []

        for item in cmvn_data:
            config = item.get('config', {})
            mode = config.get('cmvn_mode', 'none')
            labels.append(mode.capitalize())
            hit10_means.append(item.get('mean', {}).get('hit@10', 0) * 100)
            hit10_stds.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)

        x = np.arange(len(labels))
        bars = ax2.bar(x, hit10_means, yerr=hit10_stds, color=colors[:len(labels)],
                      capsize=5, error_kw={'linewidth': 1.5})

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel('Hit@10 (%)')
        ax2.set_title('(b) CMVN Mode Comparison', fontweight='bold')

        # Add value labels
        for i, (mean, std) in enumerate(zip(hit10_means, hit10_stds)):
            ax2.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Highlight best
        best_idx = np.argmax(hit10_means)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(2)

        ax2.set_ylim(0, max(hit10_means) + max(hit10_stds) + 5)
        ax2.grid(True, alpha=0.3, axis='y')

    else:
        ax2.text(0.5, 0.5, 'No CMVN data', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('(b) CMVN Mode Comparison', fontweight='bold')

    # =========================================================================
    # (c) Mel Formula Comparison
    # =========================================================================
    ax3 = axes[2]

    if 'mel_formula' in ablation_results:
        mel_data = ablation_results['mel_formula']

        labels = []
        hit10_means = []
        hit10_stds = []

        for item in mel_data:
            config = item.get('config', {})
            name = config.get('name', 'Unknown')
            htk = config.get('htk', False)
            if htk:
                labels.append('HTK')
            else:
                labels.append('Slaney')

            hit10_means.append(item.get('mean', {}).get('hit@10', 0) * 100)
            hit10_stds.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)

        x = np.arange(len(labels))
        bars = ax3.bar(x, hit10_means, yerr=hit10_stds, color=colors[:len(labels)],
                      capsize=5, error_kw={'linewidth': 1.5})

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=10)
        ax3.set_ylabel('Hit@10 (%)')
        ax3.set_title('(c) Mel-scale Formula', fontweight='bold')

        # Add value labels
        for i, (mean, std) in enumerate(zip(hit10_means, hit10_stds)):
            ax3.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')

        # Calculate and show delta
        if len(hit10_means) >= 2:
            delta = hit10_means[1] - hit10_means[0]
            ax3.annotate(f'$\\Delta$ = {delta:+.1f}%',
                        xy=(0.5, max(hit10_means) * 0.5),
                        fontsize=12, ha='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax3.set_ylim(0, max(hit10_means) + max(hit10_stds) + 5)
        ax3.grid(True, alpha=0.3, axis='y')

    else:
        ax3.text(0.5, 0.5, 'No mel formula data', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('(c) Mel-scale Formula', fontweight='bold')

    plt.tight_layout()

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 4: Ablations...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig4_ablations')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
