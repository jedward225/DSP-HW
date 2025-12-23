#!/usr/bin/env python3
"""
Generate All Figures

Master script to generate all visualization figures from experiment results.

Usage:
    python generate_all.py [--figure N] [--output-dir DIR]

Options:
    --figure N      Generate only figure N (1-8)
    --output-dir    Custom output directory (default: visualization/outputs)
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import setup_style, OUTPUT_DIR


def generate_figure(fig_num: int, output_dir: Path = None):
    """
    Generate a specific figure.

    Args:
        fig_num: Figure number (1-8)
        output_dir: Output directory (optional)
    """
    if output_dir is not None:
        # Update output directory in config
        import config
        config.OUTPUT_DIR = Path(output_dir)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if fig_num == 1:
            from fig1_method_comparison import create_figure
            name = "Method Comparison"
            filename = "fig1_method_comparison"
        elif fig_num == 2:
            from fig2_method_categories import create_figure
            name = "Method Categories"
            filename = "fig2_method_categories"
        elif fig_num == 3:
            from fig3_grid_search import create_figure
            name = "Grid Search"
            filename = "fig3_grid_search"
        elif fig_num == 4:
            from fig4_ablations import create_figure
            name = "Ablations"
            filename = "fig4_ablations"
        elif fig_num == 5:
            from fig5_robustness import create_figure
            name = "Robustness"
            filename = "fig5_robustness"
        elif fig_num == 6:
            from fig6_efficiency import create_figure
            name = "Efficiency"
            filename = "fig6_efficiency"
        elif fig_num == 7:
            from fig7_fusion_twostage import create_figure
            name = "Fusion & Two-Stage"
            filename = "fig7_fusion_twostage"
        elif fig_num == 8:
            from fig8_partial_query import create_figure
            name = "Partial Query"
            filename = "fig8_partial_query"
        elif fig_num == 9:
            from fig9_twostage_pareto import create_figure
            name = "Two-Stage Pareto"
            filename = "fig9_twostage_pareto"
        elif fig_num == 10:
            from fig10_fold_variance import create_figure
            name = "Fold Variance"
            filename = "fig10_fold_variance"
        else:
            print(f"Error: Invalid figure number {fig_num}. Must be 1-10.")
            return False

        print(f"\n{'='*60}")
        print(f"Generating Figure {fig_num}: {name}")
        print(f"{'='*60}")

        start_time = time.time()

        # Create figure
        fig = create_figure()

        if fig is not None:
            # Save figure
            from config import save_figure
            save_figure(fig, filename, output_dir)

            import matplotlib.pyplot as plt
            plt.close(fig)

            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds")
            return True
        else:
            print(f"Warning: Figure {fig_num} returned None (data may be missing)")
            return False

    except Exception as e:
        print(f"Error generating figure {fig_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all(output_dir: Path = None):
    """
    Generate all figures.

    Args:
        output_dir: Output directory (optional)
    """
    print("\n" + "="*60)
    print("DSP-HW Audio Retrieval - Visualization Suite")
    print("="*60)
    print(f"Output directory: {output_dir or OUTPUT_DIR}")
    print("="*60)

    # Setup matplotlib style
    setup_style()

    # Track results
    results = {}
    total_start = time.time()

    # Generate all figures
    for fig_num in range(1, 11):
        success = generate_figure(fig_num, output_dir)
        results[fig_num] = success

    # Print summary
    print("\n" + "="*60)
    print("Generation Summary")
    print("="*60)

    success_count = sum(results.values())
    total_count = len(results)

    for fig_num, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  Figure {fig_num}: {status}")

    total_time = time.time() - total_start
    print(f"\nTotal: {success_count}/{total_count} figures generated")
    print(f"Total time: {total_time:.2f} seconds")

    # List output files
    out_dir = output_dir or OUTPUT_DIR
    if out_dir.exists():
        print(f"\nOutput files in {out_dir}:")
        for f in sorted(out_dir.glob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")

    return success_count == total_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualization figures for DSP-HW Audio Retrieval"
    )
    parser.add_argument(
        "--figure", "-f", type=int, choices=range(1, 11), metavar="N",
        help="Generate only figure N (1-10)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        help="Custom output directory"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available figures"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Figures:")
        print("  1. Method Comparison (Hit@K curves, bar charts, radar)")
        print("  2. Method Categories (Traditional vs Deep vs Pretrained)")
        print("  3. Grid Search (Frame/hop length heatmap)")
        print("  4. Ablations (Pre-emphasis, CMVN, Mel formula)")
        print("  5. Robustness (Noise, Volume, Speed, Pitch, Time Shift)")
        print("  6. Efficiency (Pareto front, timing, memory)")
        print("  7. Fusion & Two-Stage (Late fusion, RRF, N sweep)")
        print("  8. Partial Query (Duration impact)")
        print("  9. Two-Stage Pareto (Accuracy vs latency trade-off)")
        print(" 10. Fold Variance (Cross-fold stability analysis)")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.figure:
        # Generate single figure
        success = generate_figure(args.figure, output_dir)
        sys.exit(0 if success else 1)
    else:
        # Generate all figures
        success = generate_all(output_dir)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
