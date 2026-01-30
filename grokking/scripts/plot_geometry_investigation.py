"""
Plot geometry metrics vs step to investigate geometric changes (e.g. at grokking).

Usage:
  1. First produce geometry CSV from a run:
     uv run python grokking/scripts/run_geometry_analysis.py RUN_DIR -o RUN_DIR/geometry_metrics.csv

  2. Then plot:
     uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR/geometry_metrics.csv -o RUN_DIR/geometry_plots/

  Or pass run_dir and step_interval to compute + plot in one go:
     uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR --from-run-dir --step-interval 500 -o RUN_DIR/geometry_plots/
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_geometry_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load geometry_metrics.csv."""
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"No 'step' column in {csv_path}")
    return df


def plot_geometry_investigation(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    grokking_step: int | None = None,
) -> None:
    """Plot main geometry metrics vs step. Optionally mark grokking_step with a vertical line."""
    if plt is None:
        print("matplotlib not installed; install with: pip install matplotlib", file=sys.stderr)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    step = df["step"]

    def vline_if(ax):
        if grokking_step is not None:
            ax.axvline(x=grokking_step, color="red", linestyle="--", alpha=0.7, label="grokking (val acc ≥ 0.9)")

    # 1. Parameter-space: curvature proxy, Fisher max eig, sharpness
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if "curvature_proxy" in df.columns:
        axes[0].plot(step, df["curvature_proxy"], label="curvature_proxy")
        vline_if(axes[0])
        axes[0].set_ylabel("curvature_proxy")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)
    if "fisher_max_eig" in df.columns:
        axes[1].plot(step, df["fisher_max_eig"], label="fisher_max_eig")
        vline_if(axes[1])
        axes[1].set_ylabel("fisher_max_eig")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)
    if "sharpness_hessian_trace" in df.columns:
        axes[2].plot(step, df["sharpness_hessian_trace"], label="sharpness_hessian_trace")
        vline_if(axes[2])
        axes[2].set_ylabel("sharpness_hessian_trace")
        axes[2].set_xlabel("step")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)
    plt.suptitle("Parameter-space geometry vs step")
    plt.tight_layout()
    fig.savefig(output_dir / "geometry_parameter_space.pdf", dpi=150)
    plt.close()

    # 2. Cumulative natural path length (if present)
    if "cumulative_natural_path_length" in df.columns:
        valid = pd.to_numeric(df["cumulative_natural_path_length"], errors="coerce").dropna()
        if len(valid) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(step, df["cumulative_natural_path_length"], label="cumulative_natural_path_length")
            vline_if(ax)
            ax.set_xlabel("step")
            ax.set_ylabel("cumulative_natural_path_length")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(output_dir / "geometry_path_length.pdf", dpi=150)
            plt.close()

    # 3. Representation spectrum (top eigenvalues)
    rep_cols = [c for c in df.columns if c.startswith("representation_spectrum_") and c[-1].isdigit()]
    if rep_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in sorted(rep_cols, key=lambda x: int(x.split("_")[-1]))[:5]:
            ax.plot(step, df[c], label=c.replace("representation_spectrum_", "rep_"))
        vline_if(ax)
        ax.set_xlabel("step")
        ax.set_ylabel("representation eigenvalue")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "geometry_representation_spectrum.pdf", dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot geometry metrics vs step for investigation (e.g. grokking transition)."
    )
    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to geometry_metrics.csv, or to run dir if --from-run-dir",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for plots (default: same as CSV dir or run_dir/geometry_plots)",
    )
    parser.add_argument(
        "--from-run-dir",
        action="store_true",
        help="If set, path is a run dir: first run run_geometry_analysis.py to produce CSV, then plot",
    )
    parser.add_argument("--step-interval", type=int, default=500, help="Used only with --from-run-dir")
    parser.add_argument(
        "--grokking-step",
        type=int,
        default=None,
        help="Step where val accuracy crossed threshold (e.g. 0.9); draw vertical line",
    )
    args = parser.parse_args()

    if args.from_run_dir:
        run_dir = args.path
        csv_path = run_dir / "geometry_metrics.csv"
        if not csv_path.exists():
            print(f"Running geometry analysis on {run_dir} ...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grokking.scripts.run_geometry_analysis",
                    str(run_dir),
                    "-o",
                    str(csv_path),
                    "--step-interval",
                    str(args.step_interval),
                ],
                check=True,
            )
        out_dir = args.output_dir or (run_dir / "geometry_plots")
        df = load_geometry_csv(csv_path)
    else:
        csv_path = args.path
        if not csv_path.exists():
            print(f"File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        out_dir = args.output_dir or csv_path.parent / "geometry_plots"
        df = load_geometry_csv(csv_path)

    plot_geometry_investigation(df, out_dir, grokking_step=args.grokking_step)


if __name__ == "__main__":
    main()
