"""
Figure 3: Hyperparameter Grid Search

This figure contains:
(a) Frame Length × Hop Length Heatmap
(b) n_mels × n_mfcc Heatmap
(c) Top Configurations Comparison
(d) Window Function Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from .data_loader import load_grid_search_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from data_loader import load_grid_search_results


def create_figure():
    """Create the grid search visualization figure."""

    setup_style()

    # Load data
    try:
        grid_results = load_grid_search_results()
    except FileNotFoundError:
        print("Warning: Grid search results not found")
        return None

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Hyperparameter Grid Search Analysis', fontsize=14, fontweight='bold', y=0.98)

    # =========================================================================
    # (a) Frame Length × Hop Length Heatmap
    # =========================================================================
    ax1 = axes[0, 0]

    if 'frame_hop' in grid_results:
        frame_hop_data = grid_results['frame_hop']

        # Extract unique frame lengths and hop lengths
        # Data has separate 'configs' and 'metrics' arrays
        configs_list = frame_hop_data.get('configs', [])
        metrics_list = frame_hop_data.get('metrics', [])

        configs = []
        for i, config in enumerate(configs_list):
            if i < len(metrics_list):
                frame_ms = config.get('frame_length_ms', config.get('n_fft', 0) / 22.050)
                hop_ms = config.get('hop_length_ms', config.get('hop_length', 0) / 22.050)
                hit10 = metrics_list[i].get('hit@10', 0) * 100
                configs.append((frame_ms, hop_ms, hit10))

        # Get unique values
        frame_lengths = sorted(set(c[0] for c in configs))
        hop_lengths = sorted(set(c[1] for c in configs))

        # Create 2D array for heatmap
        heatmap_data = np.zeros((len(frame_lengths), len(hop_lengths)))
        heatmap_data[:] = np.nan  # Fill with NaN for missing values

        for frame_ms, hop_ms, hit10 in configs:
            if frame_ms in frame_lengths and hop_ms in hop_lengths:
                i = frame_lengths.index(frame_ms)
                j = hop_lengths.index(hop_ms)
                heatmap_data[i, j] = hit10

        # Plot heatmap
        im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto',
                        origin='lower', vmin=np.nanmin(heatmap_data) * 0.95,
                        vmax=np.nanmax(heatmap_data) * 1.02)

        # Set ticks
        ax1.set_xticks(np.arange(len(hop_lengths)))
        ax1.set_yticks(np.arange(len(frame_lengths)))
        ax1.set_xticklabels([f'{h:.0f}' for h in hop_lengths], fontsize=9)
        ax1.set_yticklabels([f'{f:.0f}' for f in frame_lengths], fontsize=9)

        ax1.set_xlabel('Hop Length (ms)')
        ax1.set_ylabel('Frame Length (ms)')
        ax1.set_title('(a) Frame × Hop Length Grid Search', fontweight='bold')

        # Add text annotations
        for i in range(len(frame_lengths)):
            for j in range(len(hop_lengths)):
                value = heatmap_data[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > np.nanmean(heatmap_data) else 'black'
                    ax1.text(j, i, f'{value:.1f}', ha='center', va='center',
                            fontsize=8, color=text_color, fontweight='bold')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Hit@10 (%)', fontsize=10)

        # Mark best configuration
        best_idx = np.nanargmax(heatmap_data)
        best_i, best_j = np.unravel_index(best_idx, heatmap_data.shape)
        ax1.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                     fill=False, edgecolor='green', linewidth=3))

    else:
        ax1.text(0.5, 0.5, 'No frame/hop grid search data available',
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('(a) Frame × Hop Length Grid Search', fontweight='bold')

    # =========================================================================
    # (b) n_mels × n_mfcc Heatmap
    # =========================================================================
    ax2 = axes[0, 1]

    if 'mfcc_params' in grid_results:
        mfcc_data = grid_results['mfcc_params']

        configs_list = mfcc_data.get('configs', [])
        metrics_list = mfcc_data.get('metrics', [])

        configs = []
        for i, config in enumerate(configs_list):
            if i < len(metrics_list):
                n_mels = config.get('n_mels', 64)
                n_mfcc = config.get('n_mfcc', 20)
                hit10 = metrics_list[i].get('hit@10', 0) * 100
                configs.append((n_mels, n_mfcc, hit10))

        # Get unique values
        n_mels_values = sorted(set(c[0] for c in configs))
        n_mfcc_values = sorted(set(c[1] for c in configs))

        # Create 2D array for heatmap (aggregate by averaging if multiple entries)
        heatmap_data = np.zeros((len(n_mels_values), len(n_mfcc_values)))
        count_data = np.zeros((len(n_mels_values), len(n_mfcc_values)))

        for n_mels, n_mfcc, hit10 in configs:
            if n_mels in n_mels_values and n_mfcc in n_mfcc_values:
                i = n_mels_values.index(n_mels)
                j = n_mfcc_values.index(n_mfcc)
                heatmap_data[i, j] += hit10
                count_data[i, j] += 1

        # Average where we have multiple values
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_data = np.where(count_data > 0, heatmap_data / count_data, np.nan)

        # Plot heatmap
        im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto',
                        origin='lower', vmin=np.nanmin(heatmap_data) * 0.95,
                        vmax=np.nanmax(heatmap_data) * 1.02)

        # Set ticks
        ax2.set_xticks(np.arange(len(n_mfcc_values)))
        ax2.set_yticks(np.arange(len(n_mels_values)))
        ax2.set_xticklabels([str(int(n)) for n in n_mfcc_values], fontsize=9)
        ax2.set_yticklabels([str(int(n)) for n in n_mels_values], fontsize=9)

        ax2.set_xlabel('n_mfcc')
        ax2.set_ylabel('n_mels')
        ax2.set_title('(b) n_mels × n_mfcc Grid Search', fontweight='bold')

        # Add text annotations
        for i in range(len(n_mels_values)):
            for j in range(len(n_mfcc_values)):
                value = heatmap_data[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value > np.nanmean(heatmap_data) else 'black'
                    ax2.text(j, i, f'{value:.1f}', ha='center', va='center',
                            fontsize=9, color=text_color, fontweight='bold')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Hit@10 (%)', fontsize=10)

        # Mark best configuration
        best_idx = np.nanargmax(heatmap_data)
        best_i, best_j = np.unravel_index(best_idx, heatmap_data.shape)
        ax2.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                     fill=False, edgecolor='green', linewidth=3))

    else:
        ax2.text(0.5, 0.5, 'No MFCC params grid search data available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) n_mels × n_mfcc Grid Search', fontweight='bold')

    # =========================================================================
    # (c) Top Configurations Comparison
    # =========================================================================
    ax3 = axes[1, 0]

    if 'frame_hop' in grid_results:
        frame_hop_data = grid_results['frame_hop']

        # Data has separate 'configs' and 'metrics' arrays
        configs_list = frame_hop_data.get('configs', [])
        metrics_list = frame_hop_data.get('metrics', [])

        # Combine configs with their metrics
        combined = []
        for i, config in enumerate(configs_list):
            if i < len(metrics_list):
                frame_ms = config.get('frame_length_ms', config.get('n_fft', 0) / 22.050)
                hop_ms = config.get('hop_length_ms', config.get('hop_length', 0) / 22.050)
                hit10 = metrics_list[i].get('hit@10', 0)
                combined.append({
                    'frame_ms': frame_ms,
                    'hop_ms': hop_ms,
                    'hit10': hit10
                })

        # Sort by hit@10 and get top 8
        sorted_configs = sorted(combined, key=lambda x: x['hit10'], reverse=True)[:8]

        # Prepare data for bar chart
        config_labels = []
        hit10_values = []
        hit10_errors = []

        for item in sorted_configs:
            config_labels.append(f"F={item['frame_ms']:.0f}, H={item['hop_ms']:.0f}")
            hit10_values.append(item['hit10'] * 100)
            hit10_errors.append(0)  # No std available in this format

        y_pos = np.arange(len(config_labels))
        colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(config_labels)))

        bars = ax3.barh(y_pos, hit10_values, xerr=hit10_errors, color=colors,
                       capsize=3, error_kw={'linewidth': 1})

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(config_labels, fontsize=9)
        ax3.set_xlabel('Hit@10 (%)')
        ax3.set_title('(c) Top Frame/Hop Configurations', fontweight='bold')

        # Add value labels
        for i, (val, err) in enumerate(zip(hit10_values, hit10_errors)):
            ax3.text(val + err + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

        # Highlight best
        bars[0].set_edgecolor('green')
        bars[0].set_linewidth(2)

        ax3.set_xlim(0, max(hit10_values) + max(hit10_errors) + 5)
        ax3.grid(True, alpha=0.3, axis='x')

    else:
        ax3.text(0.5, 0.5, 'No grid search data available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) Top Configurations', fontweight='bold')

    # =========================================================================
    # (d) Window Function Comparison
    # =========================================================================
    ax4 = axes[1, 1]

    if 'window' in grid_results:
        window_data = grid_results['window']

        configs_list = window_data.get('configs', [])
        metrics_list = window_data.get('metrics', [])
        fold_results = window_data.get('fold_results', {})

        # Extract window function results
        windows = []
        hit10_means = []
        hit10_stds = []

        for i, config in enumerate(configs_list):
            if i < len(metrics_list):
                window = config.get('window', 'unknown')
                hit10 = metrics_list[i].get('hit@10', 0) * 100
                windows.append(window)
                hit10_means.append(hit10)

                # Calculate 95% CI from fold results if available
                if window in fold_results:
                    fold_vals = [fold_results[window][f].get('hit@10', 0) * 100
                                for f in fold_results[window]]
                    std_val = np.std(fold_vals)
                    # Convert std to 95% CI half-width: 1.96 * std / sqrt(n_folds)
                    hit10_stds.append(ci_half_width_from_std(std_val / 100) * 100)
                else:
                    hit10_stds.append(0)

        # Sort by hit@10
        sorted_idx = np.argsort(hit10_means)[::-1]
        windows = [windows[i] for i in sorted_idx]
        hit10_means = [hit10_means[i] for i in sorted_idx]
        hit10_stds = [hit10_stds[i] for i in sorted_idx]

        y_pos = np.arange(len(windows))
        colors = ['#4C72B0', '#55A868', '#C44E52']  # Blue, green, red

        bars = ax4.barh(y_pos, hit10_means, xerr=hit10_stds, color=colors[:len(windows)],
                       capsize=5, error_kw={'linewidth': 1.5})

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([w.capitalize() for w in windows], fontsize=10)
        ax4.set_xlabel('Hit@10 (%)')
        ax4.set_title('(d) Window Function Comparison', fontweight='bold')

        # Add value labels
        for i, (val, err) in enumerate(zip(hit10_means, hit10_stds)):
            ax4.text(val + err + 0.5, i, f'{val:.1f}±{err:.1f}%', va='center', fontsize=9)

        # Highlight best
        bars[0].set_edgecolor('green')
        bars[0].set_linewidth(2)

        # Set x-axis limits with some margin
        max_val = max(hit10_means) + max(hit10_stds) if hit10_stds else max(hit10_means)
        ax4.set_xlim(0, max_val + 8)
        ax4.grid(True, alpha=0.3, axis='x')

    else:
        ax4.text(0.5, 0.5, 'No window function data available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('(d) Window Function Comparison', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 3: Grid Search...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig3_grid_search')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
