"""
Figure 8: Partial Query Analysis

This figure shows the impact of query duration on retrieval performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from .data_loader import load_partial_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from data_loader import load_partial_results


def create_figure():
    """Create the partial query analysis figure."""

    setup_style()

    # Load data
    try:
        partial_results = load_partial_results()
    except FileNotFoundError:
        print("Warning: Partial query results not found")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Partial Query Analysis: Impact of Query Duration', fontsize=14, fontweight='bold', y=0.98)

    # Extract data
    durations = []
    hit1_values = []
    hit5_values = []
    hit10_values = []
    hit10_stds = []
    hit20_values = []
    mrr_values = []

    for key, data in partial_results.items():
        duration = data.get('duration_s', float(key.replace('s', '')))
        durations.append(duration)
        hit1_values.append(data.get('mean', {}).get('hit@1', 0) * 100)
        hit5_values.append(data.get('mean', {}).get('hit@5', 0) * 100)
        hit10_values.append(data.get('mean', {}).get('hit@10', 0) * 100)
        hit10_stds.append(ci_half_width_from_std(data.get('std', {}).get('hit@10', 0)) * 100)
        hit20_values.append(data.get('mean', {}).get('hit@20', 0) * 100)
        mrr_values.append(data.get('mean', {}).get('mrr@10', 0) * 100)

    # Sort by duration
    sorted_idx = np.argsort(durations)
    durations = [durations[i] for i in sorted_idx]
    hit1_values = [hit1_values[i] for i in sorted_idx]
    hit5_values = [hit5_values[i] for i in sorted_idx]
    hit10_values = [hit10_values[i] for i in sorted_idx]
    hit10_stds = [hit10_stds[i] for i in sorted_idx]
    hit20_values = [hit20_values[i] for i in sorted_idx]
    mrr_values = [mrr_values[i] for i in sorted_idx]

    # Define colors
    colors = {
        'hit@1': '#C44E52',
        'hit@5': '#FF7F0E',
        'hit@10': '#4C72B0',
        'hit@20': '#55A868',
        'mrr@10': '#8172B2',
    }

    # Plot lines
    ax.plot(durations, hit1_values, 'o-', color=colors['hit@1'], linewidth=2,
            markersize=10, label='Hit@1')
    ax.plot(durations, hit5_values, 's-', color=colors['hit@5'], linewidth=2,
            markersize=10, label='Hit@5')
    ax.errorbar(durations, hit10_values, yerr=hit10_stds, fmt='^-',
               color=colors['hit@10'], linewidth=2, markersize=10,
               capsize=4, label='Hit@10 (95% CI)')
    ax.plot(durations, hit20_values, 'D-', color=colors['hit@20'], linewidth=2,
            markersize=10, label='Hit@20')
    ax.plot(durations, mrr_values, 'v--', color=colors['mrr@10'], linewidth=2,
            markersize=10, label='MRR@10', alpha=0.8)

    # Fill area under hit@10 curve
    ax.fill_between(durations,
                   np.array(hit10_values) - np.array(hit10_stds),
                   np.array(hit10_values) + np.array(hit10_stds),
                   alpha=0.15, color=colors['hit@10'])

    ax.set_xlabel('Query Duration (seconds)', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Performance vs. Query Audio Length', fontweight='bold')

    # Add vertical line at full audio duration (5s)
    ax.axvline(x=5.0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    ax.annotate('Full Audio\n(5 sec)', xy=(5.0, 10), fontsize=10,
               ha='center', color='gray')

    # Add percentage of full query annotations
    for i, dur in enumerate(durations):
        pct = dur / 5.0 * 100
        if dur < 5.0:
            ax.annotate(f'{pct:.0f}%', xy=(dur, hit10_values[i] + hit10_stds[i] + 2),
                       ha='center', fontsize=9, color='gray')

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, max(hit20_values) + 10)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add insight box
    if len(hit10_values) >= 2:
        full_hit10 = hit10_values[-1]  # 5s
        min_hit10 = hit10_values[0]    # 0.5s
        retention = min_hit10 / full_hit10 * 100 if full_hit10 > 0 else 0

        insight_text = (f"Using only 10% of audio (0.5s):\n"
                       f"Hit@10 = {min_hit10:.1f}% ({retention:.0f}% of full)")

        ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 8: Partial Query...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig8_partial_query')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
