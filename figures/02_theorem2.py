"""Theorem 2 summary across fractions: R^2(predicted vs observed CE) with 95% CI over seeds.

This script reads each run's `geometry/loss_scatter` W&B table (last available checkpoint),
computes per-seed R^2 for:
  x = predicted loss contribution d_L(H)
  y = observed per-sample CE

Then it aggregates by training fraction and plots one curve with 95% confidence intervals.

Example:
    uv run python figures/plot_theorem2_ci_by_fraction.py \
        --project canonical_repr_exp123 \
        --fracs 0.1,0.15,0.2,0.25,0.3,0.4,0.5 \
        --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures._style import ACCENT, COLORS, apply_style, finish_axes, save_fig
from figures._wandb_fetch import DEFAULT_ENTITY

apply_style()

SCATTER_KEY = "geometry/loss_scatter"


def _table_to_arrays(table, run, call_logs: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(table, "get_dataframe"):
        df = table.get_dataframe()
    elif hasattr(table, "data"):
        df = pd.DataFrame(table.data, columns=table.columns)
    elif isinstance(table, dict):
        # W&B history often stores tables as a file reference dict:
        # {"_type":"table-file","path":"media/table/...table.json", ...}
        if "data" in table and "columns" in table:
            df = pd.DataFrame(table["data"], columns=table["columns"])
        elif "path" in table:
            rel_path = str(table["path"])
            if call_logs is not None:
                call_logs.append(
                    {"op": "download_table_file", "run_id": run.id, "path": rel_path, "status": "start"}
                )
            with tempfile.TemporaryDirectory(prefix="theorem2_table_") as tmp:
                local = run.file(rel_path).download(root=tmp, replace=True)
                table_json = json.loads(Path(local.name).read_text())
                df = pd.DataFrame(
                    table_json.get("data", []),
                    columns=table_json.get("columns", []),
                )
            if call_logs is not None:
                call_logs.append(
                    {"op": "download_table_file", "run_id": run.id, "path": rel_path, "status": "ok"}
                )
        else:
            raise TypeError(f"Unhandled table dict keys: {list(table.keys())}")
    else:
        raise TypeError(f"Unhandled table type: {type(table)}")
    if "predicted" not in df.columns or "observed" not in df.columns:
        raise ValueError(f"Expected columns predicted, observed; got {list(df.columns)}")
    x = df["predicted"].to_numpy(dtype=float)
    y = df["observed"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _r2_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return float("nan")
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r * r


def _round_frac(v: float) -> float:
    return round(float(v), 5)


def _build_run_index(api: wandb.Api, path: str, weight_decay: float, call_logs: list[dict]) -> dict[tuple[float, int], object]:
    call_logs.append({"op": "list_runs", "path": path, "status": "start"})
    runs = list(api.runs(path, filters={"state": "finished"}, per_page=500))
    call_logs.append({"op": "list_runs", "path": path, "status": "ok", "n_runs": len(runs)})
    idx: dict[tuple[float, int], object] = {}
    for run in runs:
        cfg = dict(run.config)
        try:
            frac = _round_frac(float(cfg.get("frac_train", -1)))
            seed = int(cfg.get("seed", -9999))
            wd = float(cfg.get("weight_decay", -1))
            lam_w = float(cfg.get("lambda_canonical", 0.0))
            lam_h = float(cfg.get("lambda_canonical_H", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(wd - weight_decay) > 1e-6 or lam_w != 0.0 or lam_h != 0.0:
            continue
        key = (frac, seed)
        if key not in idx:
            idx[key] = run
    return idx


def _extract_last_scatter_table(
    run,
    *,
    max_retries: int,
    retry_sleep_sec: float,
    call_logs: list[dict],
) -> object | None:
    table = None
    for attempt in range(1, max_retries + 1):
        call_logs.append({"op": "scan_history", "run_id": run.id, "attempt": attempt, "status": "start"})
        try:
            for row in run.scan_history():
                if SCATTER_KEY in row and row[SCATTER_KEY] is not None:
                    table = row[SCATTER_KEY]
            call_logs.append({"op": "scan_history", "run_id": run.id, "attempt": attempt, "status": "ok"})
            return table
        except Exception as e:  # noqa: BLE001
            call_logs.append(
                {
                    "op": "scan_history",
                    "run_id": run.id,
                    "attempt": attempt,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_sec)
            continue
    return table


def _ci95_halfwidth(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    n = len(vals)
    if n <= 1:
        return 0.0
    sem = float(vals.std(ddof=1) / math.sqrt(n))
    return 1.96 * sem


def _frac_tag(frac: float) -> str:
    return f"{frac:.3g}".replace(".", "p")


def _seed_ci_curve(
    xys: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build mean y(x) and 95% CI by aggregating per-seed binned curves."""
    if not xys:
        return np.array([]), np.array([]), np.array([])
    all_x = np.concatenate([xy[0] for xy in xys if len(xy[0]) > 0])
    if len(all_x) < 5:
        return np.array([]), np.array([]), np.array([])
    x_min = float(np.nanmin(all_x))
    x_max = float(np.nanmax(all_x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(x_min, x_max, max(4, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])

    per_seed_means = []
    for x, y in xys:
        idx = np.digitize(x, edges) - 1
        vals = np.full(len(centers), np.nan, dtype=float)
        for b in range(len(centers)):
            m = idx == b
            if np.any(m):
                vals[b] = float(np.mean(y[m]))
        per_seed_means.append(vals)
    mtx = np.vstack(per_seed_means) if per_seed_means else np.empty((0, len(centers)))
    mean = np.nanmean(mtx, axis=0)

    ci = np.zeros_like(mean)
    for j in range(len(mean)):
        col = mtx[:, j]
        col = col[np.isfinite(col)]
        ci[j] = _ci95_halfwidth(col)
    return centers, mean, ci


def _plot_fraction_scatter_with_ci(
    *,
    frac: float,
    frac_rows: pd.DataFrame,
    n_bins: int,
    max_scatter_points: int,
) -> Path:
    """One theorem-2 scatter figure per fraction with seed-level CI overlay."""
    if frac_rows.empty:
        raise ValueError(f"No rows for fraction {frac}.")

    xys = []
    all_x = []
    all_y = []
    for _, r in frac_rows.iterrows():
        x = np.asarray(r["x"], dtype=float)
        y = np.asarray(r["y"], dtype=float)
        if len(x) == 0:
            continue
        xys.append((x, y))
        all_x.append(x)
        all_y.append(y)
    if not xys:
        raise ValueError(f"No valid scatter points for fraction {frac}.")

    x_pool = np.concatenate(all_x)
    y_pool = np.concatenate(all_y)
    if len(x_pool) > max_scatter_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(x_pool), size=max_scatter_points, replace=False)
        xs = x_pool[pick]
        ys = y_pool[pick]
    else:
        xs = x_pool
        ys = y_pool

    centers, mean_y, ci_y = _seed_ci_curve(xys, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(xs, ys, s=8, alpha=0.18, color=COLORS[0], edgecolors="none", label="Per-sample points")

    lo = min(float(np.nanmin(x_pool)), float(np.nanmin(y_pool)))
    hi = max(float(np.nanmax(x_pool)), float(np.nanmax(y_pool)))
    ax.plot([lo, hi], [lo, hi], color=ACCENT, lw=1.4, ls="--", label=r"$y=x$")

    if len(centers) > 0:
        ax.plot(centers, mean_y, color=COLORS[2], lw=2.0, label="Seed-mean trend")
        ax.fill_between(
            centers,
            mean_y - ci_y,
            mean_y + ci_y,
            color=COLORS[2],
            alpha=0.22,
            linewidth=0,
            label="95% CI across seeds",
        )

    r2_vals = frac_rows["r2"].to_numpy(dtype=float)
    r2_mean = float(np.nanmean(r2_vals))
    r2_ci = _ci95_halfwidth(r2_vals)
    ax.text(
        0.04, 0.96, rf"$R^2={r2_mean:.4f}\pm{r2_ci:.4f}$",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    finish_axes(
        ax,
        xlabel=r"Predicted loss $d_L(H)$",
        ylabel=r"Observed CE (per sample)",
        title=rf"Theorem 2 scatter ($f_{{\mathrm{{train}}}}={frac:g}$)",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out = save_fig(fig, f"theorem2_scatter_ci_frac{_frac_tag(frac)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", required=True)
    ap.add_argument("--fracs", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=5, help="number of (frac,seed) jobs per batch")
    ap.add_argument("--max-retries", type=int, default=4, help="retries per run when scan_history fails")
    ap.add_argument("--retry-sleep-sec", type=float, default=3.0, help="sleep between retries")
    ap.add_argument("--scatter-ci-per-frac", action="store_true",
                    help="also write one per-fraction scatter with seed-level CI overlay")
    ap.add_argument("--scatter-bins", type=int, default=25, help="bin count for seed-level CI trend")
    ap.add_argument("--max-scatter-points", type=int, default=6000, help="max displayed points per fraction")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    api = wandb.Api(timeout=120)
    path = f"{args.entity}/{args.project}"
    call_logs: list[dict] = []
    run_idx = _build_run_index(api, path, args.weight_decay, call_logs)
    print(f"[index] loaded {len(run_idx)} candidate baseline runs")

    rows = []
    jobs = [(frac, seed) for frac in fracs for seed in seeds]
    total = len(jobs)
    for start in range(0, total, max(1, args.batch_size)):
        end = min(total, start + max(1, args.batch_size))
        batch = jobs[start:end]
        print(f"[batch] processing {start + 1}-{end} / {total}")
        for frac, seed in batch:
            run = run_idx.get((_round_frac(frac), seed))
            if run is None:
                print(f"[skip] no finished run for frac={frac}, seed={seed}")
                continue
            table = _extract_last_scatter_table(
                run,
                max_retries=args.max_retries,
                retry_sleep_sec=args.retry_sleep_sec,
                call_logs=call_logs,
            )
            if table is None:
                print(f"[skip] run {run.id} has no {SCATTER_KEY}")
                continue
            x, y = _table_to_arrays(table, run, call_logs=call_logs)
            r2 = _r2_from_arrays(x, y)
            rows.append(
                {
                    "frac_train": frac,
                    "seed": seed,
                    "run_id": run.id,
                    "n_points": int(len(x)),
                    "r2": r2,
                    "x": x,
                    "y": y,
                }
            )
            print(f"[ok] frac={frac} seed={seed} run={run.id} R2={r2:.5f} n={len(x)}")
        print(f"[batch] done {start + 1}-{end} / {total}")

    if not rows:
        raise SystemExit("No valid theorem2 scatter tables found.")

    per_seed = pd.DataFrame(rows).sort_values(["frac_train", "seed"])
    per_seed_export = per_seed.drop(columns=["x", "y"], errors="ignore")
    summary = (
        per_seed_export.groupby("frac_train", as_index=False)["r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "r2_mean", "std": "r2_std", "count": "n"})
    )
    summary["ci95"] = [
        _ci95_halfwidth(per_seed_export.loc[per_seed_export["frac_train"] == f, "r2"].to_numpy(dtype=float))
        for f in summary["frac_train"]
    ]

    # Save tabular outputs for appendix / reproducibility.
    out_dir = Path(__file__).resolve().parent.parent / "figures" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = out_dir / "theorem2_r2_per_seed.csv"
    summary_csv = out_dir / "theorem2_r2_summary_by_fraction.csv"
    calls_jsonl = out_dir / "theorem2_api_calls.jsonl"
    per_seed_export.to_csv(per_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    calls_jsonl.write_text("".join(json.dumps(row) + "\n" for row in call_logs))

    # Plot: one graph, CI across seeds for each fraction.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = summary["frac_train"].to_numpy(dtype=float)
    y = summary["r2_mean"].to_numpy(dtype=float)
    ci = summary["ci95"].to_numpy(dtype=float)

    ax.plot(x, y, color=COLORS[0], lw=2.2, marker="o", ms=5, label=r"Mean $R^2$ across seeds")
    ax.fill_between(x, y - ci, y + ci, color=COLORS[0], alpha=0.18, linewidth=0, label="95% CI")
    # Optional seed points for transparency.
    ax.scatter(
        per_seed["frac_train"].to_numpy(dtype=float),
        per_seed["r2"].to_numpy(dtype=float),
        s=18,
        alpha=0.45,
        color=ACCENT,
        edgecolors="none",
        label="Per-seed runs",
    )
    ax.axhline(1.0, color="#444444", ls="--", lw=1.0, alpha=0.8)
    ax.set_ylim(0.0, 1.02)

    finish_axes(
        ax,
        xlabel=r"Training fraction $f_{\mathrm{train}}$",
        ylabel=r"$R^2\!\left(d_L(H),\ \mathrm{CE}\right)$",
        title="Theorem 2 validation: predicted vs observed loss",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_fig(fig, "theorem2_r2_ci_by_fraction")
    print(f"wrote {out}")
    print(f"wrote {per_seed_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {calls_jsonl}")

    if args.scatter_ci_per_frac:
        for frac in sorted(per_seed["frac_train"].unique()):
            frac_rows = per_seed[per_seed["frac_train"] == frac]
            scatter_out = _plot_fraction_scatter_with_ci(
                frac=float(frac),
                frac_rows=frac_rows,
                n_bins=args.scatter_bins,
                max_scatter_points=args.max_scatter_points,
            )
            print(f"wrote {scatter_out}")


if __name__ == "__main__":
    main()
