"""
Figure 5: Robustness Analysis (Enhanced)

This figure contains:
(a) Noise Robustness Curve - Hit@10 vs SNR
(b) Volume Perturbation Impact
(c) Speed Perturbation Impact
(d) Pitch Perturbation Impact
(e) Time Shift Impact
(f) Overall Degradation Summary
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Handle both module and direct execution imports
try:
    from .config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from .data_loader import load_robustness_results
except ImportError:
    from config import setup_style, save_figure, CATEGORY_COLORS, ci_half_width_from_std
    from data_loader import load_robustness_results


def create_figure():
    """Create the enhanced robustness analysis figure."""

    setup_style()

    # Load data
    try:
        robustness_results = load_robustness_results()
    except FileNotFoundError:
        print("Warning: Robustness results not found")
        return None

    # Get all results from the 'all' key or construct from individual keys
    if 'all' in robustness_results:
        all_data = robustness_results['all']
    else:
        all_data = {}
        for key in ['noise', 'volume', 'speed']:
            if key in robustness_results:
                all_data.update(robustness_results[key])

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Robustness Analysis Under Various Perturbations', fontsize=14, fontweight='bold', y=0.98)

    # Color scheme
    colors_pos = '#55A868'  # Green for baseline/clean
    colors_neg = '#C44E52'  # Red for degradation
    colors_mid = '#4C72B0'  # Blue for moderate

    # =========================================================================
    # (a) Noise Robustness Curve
    # =========================================================================
    ax1 = axes[0, 0]

    order = ['clean', 'noise_20dB', 'noise_10dB', 'noise_0dB']
    snr_labels = ['Clean', '20 dB', '10 dB', '0 dB']

    hit10_means = []
    hit10_stds = []
    hit1_means = []

    for key in order:
        if key in all_data:
            item = all_data[key]
            hit10_means.append(item.get('mean', {}).get('hit@10', 0) * 100)
            hit10_stds.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)
            hit1_means.append(item.get('mean', {}).get('hit@1', 0) * 100)
        else:
            hit10_means.append(0)
            hit10_stds.append(0)
            hit1_means.append(0)

    x = np.arange(len(snr_labels))

    # Plot Hit@10 and Hit@1
    ax1.errorbar(x, hit10_means, yerr=hit10_stds, fmt='o-',
                color=CATEGORY_COLORS['traditional'], linewidth=2,
                markersize=10, capsize=5, capthick=2, label='Hit@10')
    ax1.plot(x, hit1_means, 's--', color=colors_neg, linewidth=2,
            markersize=8, label='Hit@1')

    ax1.fill_between(x,
                    np.array(hit10_means) - np.array(hit10_stds),
                    np.array(hit10_means) + np.array(hit10_stds),
                    alpha=0.2, color=CATEGORY_COLORS['traditional'])

    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_labels, fontsize=10)
    ax1.set_xlabel('Noise Level (SNR)')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('(a) Noise Robustness', fontweight='bold')

    # Add degradation annotation
    if len(hit10_means) >= 2:
        degradation = hit10_means[0] - hit10_means[-1]
        ax1.annotate(f'Δ = -{degradation:.1f}%',
                    xy=(len(x)-1, hit10_means[-1]),
                    xytext=(len(x)-1.3, hit10_means[-1] + 12),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax1.set_ylim(0, max(hit10_means) + max(hit10_stds) + 15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # =========================================================================
    # (b) Volume Perturbation Impact
    # =========================================================================
    ax2 = axes[0, 1]

    vol_order = ['clean', 'volume_+6dB', 'volume_-6dB']
    vol_labels = ['Clean', '+6 dB', '-6 dB']
    vol_colors = [colors_pos, colors_mid, colors_neg]

    vol_hit10 = []
    vol_hit10_std = []
    for key in vol_order:
        if key in all_data:
            item = all_data[key]
            vol_hit10.append(item.get('mean', {}).get('hit@10', 0) * 100)
            vol_hit10_std.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)
        else:
            vol_hit10.append(0)
            vol_hit10_std.append(0)

    x = np.arange(len(vol_labels))
    bars = ax2.bar(x, vol_hit10, yerr=vol_hit10_std, color=vol_colors,
                  capsize=5, error_kw={'linewidth': 1.5}, width=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(vol_labels, fontsize=10)
    ax2.set_ylabel('Hit@10 (%)')
    ax2.set_title('(b) Volume Perturbation', fontweight='bold')

    # Add value labels and delta
    for i, (mean, std) in enumerate(zip(vol_hit10, vol_hit10_std)):
        ax2.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            delta = mean - vol_hit10[0]
            color = 'red' if delta < 0 else 'green'
            ax2.annotate(f'{delta:+.1f}%', xy=(i, mean - 4),
                        ha='center', fontsize=9, color=color, fontweight='bold')

    ax2.set_ylim(0, max(vol_hit10) + max(vol_hit10_std) + 8)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (c) Speed Perturbation Impact
    # =========================================================================
    ax3 = axes[0, 2]

    speed_order = ['clean', 'speed_0.9x', 'speed_1.1x']
    speed_labels = ['Clean', '0.9× (Slower)', '1.1× (Faster)']
    speed_colors = [colors_pos, colors_mid, colors_neg]

    speed_hit10 = []
    speed_hit10_std = []
    speed_hit1 = []
    for key in speed_order:
        if key in all_data:
            item = all_data[key]
            speed_hit10.append(item.get('mean', {}).get('hit@10', 0) * 100)
            speed_hit10_std.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)
            speed_hit1.append(item.get('mean', {}).get('hit@1', 0) * 100)
        else:
            speed_hit10.append(0)
            speed_hit10_std.append(0)
            speed_hit1.append(0)

    x = np.arange(len(speed_labels))
    width = 0.35
    bars1 = ax3.bar(x - width/2, speed_hit10, width, yerr=speed_hit10_std,
                   label='Hit@10', color=[colors_pos, colors_mid, colors_neg],
                   capsize=4, error_kw={'linewidth': 1})
    bars2 = ax3.bar(x + width/2, speed_hit1, width, label='Hit@1',
                   color=[colors_pos, colors_mid, colors_neg], alpha=0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(speed_labels, fontsize=9)
    ax3.set_ylabel('Score (%)')
    ax3.set_title('(c) Speed Perturbation', fontweight='bold')

    # Add delta for speed 1.1x
    delta_slow = speed_hit10[1] - speed_hit10[0]
    delta_fast = speed_hit10[2] - speed_hit10[0]
    ax3.annotate(f'{delta_slow:+.1f}%', xy=(1, speed_hit10[1] + 2),
                ha='center', fontsize=9, color='orange', fontweight='bold')
    ax3.annotate(f'{delta_fast:+.1f}%', xy=(2, speed_hit10[2] + 2),
                ha='center', fontsize=9, color='red', fontweight='bold')

    ax3.set_ylim(0, max(speed_hit10) + max(speed_hit10_std) + 10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (d) Pitch Perturbation Impact
    # =========================================================================
    ax4 = axes[1, 0]

    pitch_order = ['clean', 'pitch_-1', 'pitch_+1']
    pitch_labels = ['Clean', '-1 Semitone', '+1 Semitone']
    pitch_colors = [colors_pos, colors_mid, colors_neg]

    pitch_hit10 = []
    pitch_hit10_std = []
    pitch_hit1 = []
    for key in pitch_order:
        if key in all_data:
            item = all_data[key]
            pitch_hit10.append(item.get('mean', {}).get('hit@10', 0) * 100)
            pitch_hit10_std.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)
            pitch_hit1.append(item.get('mean', {}).get('hit@1', 0) * 100)
        else:
            pitch_hit10.append(0)
            pitch_hit10_std.append(0)
            pitch_hit1.append(0)

    x = np.arange(len(pitch_labels))
    width = 0.35
    bars1 = ax4.bar(x - width/2, pitch_hit10, width, yerr=pitch_hit10_std,
                   label='Hit@10', color=pitch_colors,
                   capsize=4, error_kw={'linewidth': 1})
    bars2 = ax4.bar(x + width/2, pitch_hit1, width, label='Hit@1',
                   color=pitch_colors, alpha=0.5)

    ax4.set_xticks(x)
    ax4.set_xticklabels(pitch_labels, fontsize=9)
    ax4.set_ylabel('Score (%)')
    ax4.set_title('(d) Pitch Perturbation', fontweight='bold')

    # Add delta annotations
    for i in range(1, len(pitch_hit10)):
        delta = pitch_hit10[i] - pitch_hit10[0]
        color = 'red' if delta < -2 else 'orange'
        ax4.annotate(f'{delta:+.1f}%', xy=(i, pitch_hit10[i] + 2),
                    ha='center', fontsize=9, color=color, fontweight='bold')

    ax4.set_ylim(0, max(pitch_hit10) + max(pitch_hit10_std) + 10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (e) Time Shift Impact
    # =========================================================================
    ax5 = axes[1, 1]

    time_order = ['clean', 'time_shift_0.1', 'time_shift_0.2']
    time_labels = ['Clean', '10% Shift', '20% Shift']
    time_colors = [colors_pos, colors_mid, colors_mid]

    time_hit10 = []
    time_hit10_std = []
    for key in time_order:
        if key in all_data:
            item = all_data[key]
            time_hit10.append(item.get('mean', {}).get('hit@10', 0) * 100)
            time_hit10_std.append(ci_half_width_from_std(item.get('std', {}).get('hit@10', 0)) * 100)
        else:
            time_hit10.append(0)
            time_hit10_std.append(0)

    x = np.arange(len(time_labels))
    bars = ax5.bar(x, time_hit10, yerr=time_hit10_std, color=time_colors,
                  capsize=5, error_kw={'linewidth': 1.5}, width=0.6)

    ax5.set_xticks(x)
    ax5.set_xticklabels(time_labels, fontsize=10)
    ax5.set_ylabel('Hit@10 (%)')
    ax5.set_title('(e) Time Shift Impact', fontweight='bold')

    # Add value labels
    for i, (mean, std) in enumerate(zip(time_hit10, time_hit10_std)):
        ax5.text(i, mean + std + 0.5, f'{mean:.1f}%', ha='center', fontsize=9, fontweight='bold')
        if i > 0:
            delta = mean - time_hit10[0]
            color = 'green' if delta >= 0 else 'orange'
            ax5.annotate(f'{delta:+.1f}%', xy=(i, mean - 4),
                        ha='center', fontsize=9, color=color, fontweight='bold')

    ax5.set_ylim(0, max(time_hit10) + max(time_hit10_std) + 8)
    ax5.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # (f) Overall Degradation Summary
    # =========================================================================
    ax6 = axes[1, 2]

    # Collect all perturbations and their degradation
    perturbations = []
    degradations = []
    bar_colors = []

    clean_hit10 = all_data.get('clean', {}).get('mean', {}).get('hit@10', 0) * 100

    perturb_map = {
        'noise_0dB': ('Noise 0dB', colors_neg),
        'speed_1.1x': ('Speed 1.1×', colors_neg),
        'noise_10dB': ('Noise 10dB', '#DD8452'),
        'speed_0.9x': ('Speed 0.9×', '#DD8452'),
        'volume_-6dB': ('Volume -6dB', colors_mid),
        'volume_+6dB': ('Volume +6dB', colors_mid),
        'pitch_+1': ('Pitch +1', colors_mid),
        'pitch_-1': ('Pitch -1', colors_mid),
        'noise_20dB': ('Noise 20dB', '#7FB285'),
        'time_shift_0.2': ('Time 20%', '#7FB285'),
        'time_shift_0.1': ('Time 10%', '#7FB285'),
    }

    for key, (label, color) in perturb_map.items():
        if key in all_data:
            hit10 = all_data[key].get('mean', {}).get('hit@10', 0) * 100
            deg = clean_hit10 - hit10
            perturbations.append(label)
            degradations.append(deg)
            bar_colors.append(color)

    # Sort by degradation
    sorted_idx = np.argsort(degradations)[::-1]
    perturbations = [perturbations[i] for i in sorted_idx]
    degradations = [degradations[i] for i in sorted_idx]
    bar_colors = [bar_colors[i] for i in sorted_idx]

    y_pos = np.arange(len(perturbations))
    bars = ax6.barh(y_pos, degradations, color=bar_colors, height=0.7)

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(perturbations, fontsize=9)
    ax6.set_xlabel('Hit@10 Degradation (%)')
    ax6.set_title('(f) Degradation Summary', fontweight='bold')

    # Add value labels
    for i, val in enumerate(degradations):
        ax6.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    ax6.axvline(x=0, color='black', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    """Generate and save the figure."""
    print("Generating Figure 5: Robustness (Enhanced)...")
    fig = create_figure()
    if fig is not None:
        save_figure(fig, 'fig5_robustness')
        plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
