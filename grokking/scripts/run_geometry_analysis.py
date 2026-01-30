"""Offline geometry analysis over checkpoints (sparse schedule, e.g. every 5K steps)."""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import re

import torch

from grokking.geometry.analysis import compute_geometry_at_checkpoint
from grokking.geometry.path import get_flat_params, segment_natural_length
from grokking.scripts.training_loop_state import TrainingLoopState

default_logger = logging.getLogger(__name__)


def find_checkpoint_dirs(
    run_dir: pathlib.Path,
    step_interval: int = 500,
) -> list[pathlib.Path]:
    """Find checkpoint dirs under run_dir/checkpoints/step=* with step % step_interval == 0."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    step_dirs: list[tuple[int, pathlib.Path]] = []
    pattern = re.compile(r"^step=(\d+)$")
    for p in checkpoints_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step_interval > 0 and step % step_interval != 0:
            continue
        step_dirs.append((step, p))
    step_dirs.sort(key=lambda x: x[0])
    return [p for _, p in step_dirs]


def run_geometry_analysis(
    run_dir: pathlib.Path,
    output_csv: pathlib.Path,
    *,
    step_interval: int = 500,
    n_fisher_batches: int = 20,
    n_hutchinson: int = 20,
    compute_path_length: bool = True,
    map_location: str = "cpu",
    verbosity: int = 0,
    logger: logging.Logger | None = None,
) -> None:
    """
    Load checkpoints at step_interval (e.g. 5K), compute geometry metrics, save CSV.
    If compute_path_length is True, computes natural (Fisher) path length between consecutive checkpoints.
    """
    log = logger or default_logger
    checkpoint_dirs = find_checkpoint_dirs(run_dir, step_interval=step_interval)
    if not checkpoint_dirs:
        log.warning("No checkpoint dirs found under %s with step_interval=%s", run_dir, step_interval)
        return

    log.info("Found %d checkpoints (step_interval=%d)", len(checkpoint_dirs), step_interval)
    rows: list[dict] = []
    prev_theta: torch.Tensor | None = None
    prev_step: int | None = None
    cumulative_natural_path_length: float = 0.0

    for ckpt_dir in checkpoint_dirs:
        step_str = ckpt_dir.name  # step=5000
        step = int(step_str.split("=")[1])
        log.info("Loading checkpoint %s ...", ckpt_dir)
        try:
            state = TrainingLoopState.from_checkpoints_root_dir(
                checkpoints_root_dir=ckpt_dir,
                output_dir=run_dir,
                map_location=map_location,
                verbosity=verbosity,
                logger=log,
            )
        except Exception as e:
            log.warning("Failed to load %s: %s", ckpt_dir, e)
            continue

        curr_theta = get_flat_params(state.model, device=state.device)
        if compute_path_length and prev_theta is not None and prev_step is not None:
            try:
                seg = segment_natural_length(
                    prev_theta,
                    curr_theta,
                    state.model,
                    state.train_dataloader,
                    state.device,
                    n_fisher_batches=n_fisher_batches,
                    top_k=10,
                )
                cumulative_natural_path_length += seg
                segment_natural_length_val = seg
            except Exception as e:
                log.warning("Path length at step %d: %s", step, e)
                segment_natural_length_val = ""
        else:
            segment_natural_length_val = ""

        log.info("Computing geometry at step %d ...", step)
        try:
            result = compute_geometry_at_checkpoint(
                model=state.model,
                train_dataloader=state.train_dataloader,
                device=state.device,
                step=step,
                n_fisher_batches=n_fisher_batches,
                n_hutchinson=n_hutchinson,
            )
        except Exception as e:
            log.warning("Failed geometry at step %d: %s", step, e)
            continue
        # Flatten list of eigenvalues for CSV (e.g. fisher_eig_0, fisher_eig_1, ...)
        flat: dict = {"step": result["step"]}
        if compute_path_length:
            flat["segment_natural_length"] = segment_natural_length_val if segment_natural_length_val != "" else ""
            flat["cumulative_natural_path_length"] = cumulative_natural_path_length
        for k, v in result.items():
            if k == "step":
                continue
            if isinstance(v, list):
                for i, val in enumerate(v):
                    flat[f"{k}_{i}"] = val
            else:
                flat[k] = v
        rows.append(flat)
        prev_theta = curr_theta.detach().clone()
        prev_step = step

    if not rows:
        log.warning("No rows to write")
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    for r in rows:
        for k in fieldnames:
            if k not in r:
                r[k] = ""
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geometry analysis on saved checkpoints (5K schedule).")
    parser.add_argument("run_dir", type=pathlib.Path, help="Run directory containing checkpoints/step=*")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output CSV path (default: run_dir/geometry_metrics.csv)",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=500,
        help="Analyze checkpoints every N steps (default: 500)",
    )
    parser.add_argument("--n-fisher-batches", type=int, default=20, help="Batches for empirical Fisher")
    parser.add_argument("--n-hutchinson", type=int, default=20, help="Samples for Hutchinson Hessian trace")
    parser.add_argument("--map-location", type=str, default="cpu", help="Device to load checkpoints onto")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    output = args.output or (args.run_dir / "geometry_metrics.csv")
    run_geometry_analysis(
        run_dir=args.run_dir,
        output_csv=output,
        step_interval=args.step_interval,
        n_fisher_batches=args.n_fisher_batches,
        n_hutchinson=args.n_hutchinson,
        map_location=args.map_location,
        verbosity=1 if args.verbose else 0,
    )


if __name__ == "__main__":
    main()
