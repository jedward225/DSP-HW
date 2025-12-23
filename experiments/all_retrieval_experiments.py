#!/usr/bin/env python3
"""
Unified runner for all retrieval experiments and trainings with Rich UI.
Runs each existing experiment script sequentially, logging output per step.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel


ROOT = Path(__file__).resolve().parents[1]
CONSOLE = Console()


def build_experiments(skip_training: bool = False, skip_pretrained: bool = False):
    cmds = [
        ("Retrieval (M1-M7)", [sys.executable, "experiments/retrieval/run_retrieval.py"]),
        ("Grid Search", [sys.executable, "experiments/retrieval/run_grid_search.py"]),
        ("Ablations", [sys.executable, "experiments/retrieval/run_ablations.py"]),
        ("Robustness", [sys.executable, "experiments/retrieval/run_robustness.py"]),
        ("Efficiency", [sys.executable, "experiments/retrieval/run_efficiency.py"]),
        ("Fusion", [sys.executable, "experiments/retrieval/run_fusion.py"]),
        ("Two-Stage", [sys.executable, "experiments/retrieval/run_twostage.py"]),
        ("Partial Query", [sys.executable, "experiments/retrieval/run_partial.py"]),
    ]

    if not skip_training:
        # Get data directory from environment or use default
        data_dir = os.environ.get('ESC50_DIR', 'ESC-50')
        cmds.extend(
            [
                ("Train Autoencoder", [sys.executable, "experiments/training/train_autoencoder.py", "--data_dir", data_dir]),
                ("Train CNN", [sys.executable, "experiments/training/train_cnn.py", "--data_dir", data_dir]),
                ("Train Contrastive", [sys.executable, "experiments/training/train_contrastive.py", "--data_dir", data_dir]),
            ]
        )
        # Deep retriever evaluation (after training)
        cmds.append(
            ("Deep Retrievers", [sys.executable, "experiments/retrieval/run_deep_retrievers.py"])
        )

    if not skip_pretrained:
        # Pretrained model evaluation (CLAP, BEATs, Hybrid)
        # Note: Requires downloading checkpoints first, will skip gracefully if unavailable
        cmds.append(
            ("Pretrained (CLAP/BEATs)", [sys.executable, "experiments/retrieval/run_pretrained.py"])
        )

    return cmds


def run_experiments(experiments, logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        for name, cmd in experiments:
            log_file = logs_dir / f"{name.lower().replace(' ', '_').replace('-', '_')}.log"
            task_id = progress.add_task(f"[cyan]{name}", start=False)
            progress.start_task(task_id)

            with log_file.open("w") as lf:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )

                while process.poll() is None:
                    time.sleep(0.2)
                    progress.advance(task_id, 0)

                returncode = process.wait()

            if returncode == 0:
                progress.update(task_id, description=f"[green]{name} (done)")
                results.append((name, "success", log_file))
            else:
                progress.update(task_id, description=f"[red]{name} (failed {returncode})")
                results.append((name, f"failed ({returncode})", log_file))

    return results


def render_summary(results, logs_dir: Path, started_at: datetime):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment")
    table.add_column("Status")
    table.add_column("Log File")

    for name, status, log_file in results:
        table.add_row(name, status, str(log_file.relative_to(ROOT)))

    CONSOLE.print()
    CONSOLE.print(Panel.fit(table, title="All Experiments", subtitle=f"Logs: {logs_dir.relative_to(ROOT)}"))
    CONSOLE.print(f"Started at: {started_at.isoformat(timespec='seconds')}")
    CONSOLE.print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run all retrieval experiments with Rich UI")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training scripts (autoencoder, CNN, contrastive) and deep retriever evaluation",
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Skip pretrained model evaluation (CLAP, BEATs, Hybrid)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=ROOT / "experiments" / "retrieval" / "results" / "all_runs",
        help="Directory to store experiment logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.logs_dir / timestamp

    experiments = build_experiments(
        skip_training=args.skip_training,
        skip_pretrained=args.skip_pretrained,
    )
    started_at = datetime.now()
    results = run_experiments(experiments, run_dir)
    render_summary(results, run_dir, started_at)


if __name__ == "__main__":
    main()
