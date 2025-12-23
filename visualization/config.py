"""
Configuration and style settings for visualization.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import math

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "retrieval" / "results"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Experiment Constants
# =============================================================================

# ESC-50 retrieval experiments use 5-fold cross-validation by default.
NUM_FOLDS = 5

# Normal approximation for 95% CI half-width (used when per-metric CI is not provided).
Z_95 = 1.96

# =============================================================================
# Color Schemes
# =============================================================================

# Category colors
CATEGORY_COLORS = {
    'traditional': '#4C72B0',    # Blue
    'deep': '#55A868',           # Green
    'pretrained': '#C44E52',     # Red
    'baseline': '#8172B2',       # Purple
    'highlight': '#CCB974',      # Gold
    'grid': '#E5E5E5',           # Light gray
    'text': '#333333',           # Dark gray
}

# Individual method colors
METHOD_COLORS = {
    # Traditional methods (blues/purples)
    'M1_MFCC_Pool_Cos': '#1f77b4',
    'M2_MFCC_Delta_Pool': '#aec7e8',
    'M3_LogMel_Pool': '#ff7f0e',
    'M4_Spectral_Stat': '#ffbb78',
    'M5_MFCC_DTW': '#9467bd',
    'M6_BoAW_ChiSq': '#c5b0d5',
    'M7_MultiRes_Fusion': '#e377c2',
    # Pretrained methods (teals/greens)
    'M8_CLAP': '#17becf',
    'M9_Hybrid': '#2ca02c',
    'BEATs': '#98df8a',
    # Deep learning methods (reds/oranges)
    'Deep_Autoencoder': '#d62728',
    'Deep_CNN': '#ff9896',
    'Deep_Contrastive': '#8c564b',
}

# Method display names (shorter, cleaner)
METHOD_NAMES = {
    'M1_MFCC_Pool_Cos': 'M1: MFCC+Pool',
    'M2_MFCC_Delta_Pool': 'M2: MFCC+Delta',
    'M3_LogMel_Pool': 'M3: LogMel',
    'M4_Spectral_Stat': 'M4: Spectral',
    'M5_MFCC_DTW': 'M5: DTW',
    'M6_BoAW_ChiSq': 'M6: BoAW',
    'M7_MultiRes_Fusion': 'M7: MultiRes',
    'M8_CLAP': 'M8: CLAP',
    'M9_Hybrid': 'M9: Hybrid',
    'BEATs': 'BEATs',
    'Deep_Autoencoder': 'Autoencoder',
    'Deep_CNN': 'CNN',
    'Deep_Contrastive': 'Contrastive',
}

# Method categories
METHOD_CATEGORIES = {
    'traditional': ['M1_MFCC_Pool_Cos', 'M2_MFCC_Delta_Pool', 'M3_LogMel_Pool',
                   'M4_Spectral_Stat', 'M5_MFCC_DTW', 'M6_BoAW_ChiSq', 'M7_MultiRes_Fusion'],
    'deep': ['Deep_Autoencoder', 'Deep_CNN', 'Deep_Contrastive'],
    'pretrained': ['M8_CLAP', 'BEATs', 'M9_Hybrid'],
}

# =============================================================================
# Typography
# =============================================================================

FONT_SIZES = {
    'title': 14,
    'subtitle': 12,
    'axis_label': 11,
    'tick_label': 10,
    'legend': 9,
    'annotation': 9,
}

FONT_FAMILY = 'DejaVu Sans'

# =============================================================================
# Figure Settings
# =============================================================================

FIGURE_SETTINGS = {
    'dpi': 300,
    'format': ['pdf', 'png'],
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
    'transparent': False,
    'facecolor': 'white',
}

# Standard figure sizes (width, height) in inches
FIGURE_SIZES = {
    'single': (8, 6),
    'double': (12, 5),
    'quad': (12, 10),
    'wide': (14, 5),
    'tall': (8, 10),
}

# =============================================================================
# Plot Style Configuration
# =============================================================================

def setup_style():
    """Configure matplotlib style for academic-clean visualizations."""

    # Use seaborn whitegrid style as base
    plt.style.use('seaborn-v0_8-whitegrid')

    # Custom parameters
    params = {
        # Font
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZES['tick_label'],

        # Axes
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZES['axis_label'],
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.axisbelow': True,

        # Grid
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',

        # Ticks
        'xtick.labelsize': FONT_SIZES['tick_label'],
        'ytick.labelsize': FONT_SIZES['tick_label'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,

        # Legend
        'legend.fontsize': FONT_SIZES['legend'],
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': True,

        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': FIGURE_SETTINGS['dpi'],
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',

        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 6,

        # Patches (bars, etc.)
        'patch.linewidth': 0.5,
        'patch.edgecolor': '#333333',
    }

    mpl.rcParams.update(params)

def save_figure(fig, filename, output_dir=None):
    """
    Save figure in both PDF and PNG formats.

    Args:
        fig: matplotlib Figure object
        filename: base filename without extension
        output_dir: output directory (defaults to OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in FIGURE_SETTINGS['format']:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            filepath,
            format=fmt,
            dpi=FIGURE_SETTINGS['dpi'],
            bbox_inches=FIGURE_SETTINGS['bbox_inches'],
            pad_inches=FIGURE_SETTINGS['pad_inches'],
            facecolor=FIGURE_SETTINGS['facecolor'],
        )
        print(f"Saved: {filepath}")

# =============================================================================
# Helper Functions
# =============================================================================

def get_method_color(method):
    """Get color for a method, with fallback."""
    return METHOD_COLORS.get(method, '#7f7f7f')

def get_method_name(method):
    """Get display name for a method."""
    return METHOD_NAMES.get(method, method)

def get_category_color(category):
    """Get color for a method category."""
    return CATEGORY_COLORS.get(category, '#7f7f7f')

def format_percentage(value, decimals=1):
    """Format a decimal value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def ci_half_width_from_ci(ci_entry):
    """Return CI half-width from a CI entry dict (lower/upper/width)."""
    if not ci_entry:
        return 0.0

    width = ci_entry.get('width', None)
    if width is not None:
        return float(width) / 2.0

    lower = ci_entry.get('lower', None)
    upper = ci_entry.get('upper', None)
    if lower is None or upper is None:
        return 0.0
    return (float(upper) - float(lower)) / 2.0


def ci_half_width_from_std(std_value: float, num_folds: int = NUM_FOLDS, z: float = Z_95) -> float:
    """Approximate 95% CI half-width of the mean from fold std."""
    if std_value is None:
        return 0.0
    if num_folds is None or num_folds <= 0:
        return 0.0
    return float(z) * float(std_value) / math.sqrt(int(num_folds))

# Initialize style when module is imported
setup_style()
