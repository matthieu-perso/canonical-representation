"""Regularizer comparison: baseline vs multiple WD-only and canonical H penalties.

Reads run histories, aligns on training step, and plots mean ± 95% CI of ``val/accuracy`` across seeds.

**Why the default command feels slow:** it calls the W&B API to list every run in the project,
then downloads any run not already cached. That can take **minutes** on first use (network +
``scan_history`` per run).

**Fast path:** populate ``analysis/runs_cache/<entity>__<project>/*.parquet`` once with
``figures/_wandb_fetch.py``, then pass ``--cache-only`` so this script **never touches the API**
— typically **a few seconds** for hundreds of cached runs.

Typical paper mapping (adjust flags to match your sweep):
  • Baseline — ``weight_decay_geometry`` with default WD (0.01), no λ_H.
  • Weight decay only — same experiment, larger WD (e.g. 0.1), still no λ_H.
  • Canonical — ``canonical_regularizer`` with WD=0 and λ_H = β.

Examples
--------
    # One-time sync from W&B into analysis/runs_cache/ (slow)
    uv run python figures/_wandb_fetch.py --project regularizer

    # Plot from disk only (fast)
    uv run python figures/plot_regularizer_three_way.py --cache-only --project regularizer \\
        --frac 0.3 --baseline-wd 0.01 --wd-values 0.03 0.1 --canonical-lamh-values 0.03 0.1 --seeds 0 1 2

    # Appendix grid
    uv run python figures/plot_regularizer_three_way.py --cache-only \\
        --project regularizer --fracs 0.1 0.15 0.2 0.25 0.3 0.4 0.5 \\
        --baseline-wd 0.01 --wd-only 0.1 --canonical-lamh 0.1
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures._style import ACCENT, COLORS, apply_style, finish_axes, save_fig
from figures._wandb_fetch import DEFAULT_ENTITY, fetch_run_history, _project_cache_dir


apply_style()


def _infer_group_from_run_name(name: str) -> str:
    """Infer W&B group from run name (not always stored in config rows)."""
    if not isinstance(name, str):
        return "unknown"
    if "lamH" in name or "lamW" in name:
        return "canonical_regularizer"
    if "_wd" in name:
        return "weight_decay_geometry"
    return "unknown"


def _peek_cfg(path: Path) -> Optional[dict]:
    """Lightweight first-row metadata from a cached parquet."""
    cols = ["run_name", "cfg.frac_train", "cfg.seed", "cfg.weight_decay", "cfg.lambda_canonical_H"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return None
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "run_name": row.get("run_name"),
        "frac_train": row.get("cfg.frac_train"),
        "seed": row.get("cfg.seed"),
        "weight_decay": row.get("cfg.weight_decay"),
        "lambda_canonical_H": row.get("cfg.lambda_canonical_H"),
        "path": path,
    }


def _cfg_float(cfg: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = cfg.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _pick_cached_paths(
    entity: str,
    project: str,
    *,
    frac: float,
    seeds: Optional[list],
    group: str,
    weight_decay: Optional[float],
    lam_h: Optional[float],
) -> list:
    """Select matching parquets under analysis/runs_cache (no API)."""
    cache_dir = _project_cache_dir(entity, project)
    if not cache_dir.is_dir():
        print(f"[warn] cache dir missing: {cache_dir}")
        return []
    out = []
    for path in sorted(cache_dir.glob("*.parquet")):
        meta = _peek_cfg(path)
        if meta is None:
            continue
        inferred = _infer_group_from_run_name(str(meta.get("run_name") or ""))
        if inferred != group:
            continue
        ft = _cfg_float(meta, "frac_train")
        if ft is None or abs(ft - frac) > 1e-5:
            continue
        sd = meta.get("seed", -9999)
        try:
            sd_int = int(sd if sd is not None else -9999)
        except (TypeError, ValueError):
            continue
        if seeds is not None and sd_int not in seeds:
            continue
        wd = _cfg_float(meta, "weight_decay", 0.0)
        wd = wd if wd is not None else 0.0
        lh_raw = _cfg_float(meta, "lambda_canonical_H", 0.0)
        lh = lh_raw if lh_raw is not None else 0.0
        if weight_decay is not None and abs(wd - weight_decay) > 1e-8:
            continue
        if lam_h is not None and abs(lh - lam_h) > 1e-8:
            continue
        out.append(meta["path"])
    return out


def _pick_runs_wandb(
    api,
    *,
    frac: float,
    seeds: Optional[list],
    group: str,
    weight_decay: Optional[float],
    lam_h: Optional[float],
    entity_project: str,
):
    runs_out = []
    for r in api.runs(
        entity_project,
        filters={"state": {"$in": ["finished", "running", "crashed"]}},
        per_page=500,
    ):
        if (getattr(r, "group", None) or "") != group:
            continue
        cfg = dict(r.config)
        ft = _cfg_float(cfg, "frac_train")
        if ft is None or abs(ft - frac) > 1e-5:
            continue
        sd = cfg.get("seed", -9999)
        try:
            sd_int = int(sd if sd is not None else -9999)
        except (TypeError, ValueError):
            continue
        if seeds is not None and sd_int not in seeds:
            continue
        wd = _cfg_float(cfg, "weight_decay", 0.0)
        wd = wd if wd is not None else 0.0
        lh_raw = _cfg_float(cfg, "lambda_canonical_H", 0.0)
        lh = lh_raw if lh_raw is not None else 0.0
        if weight_decay is not None and abs(wd - weight_decay) > 1e-8:
            continue
        if lam_h is not None and abs(lh - lam_h) > 1e-8:
            continue
        runs_out.append(r)
    return runs_out


def _history_cached(run, entity: str, project: str) -> pd.DataFrame:
    cache = _project_cache_dir(entity, project) / f"{run.id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    h = fetch_run_history(run)
    if not h.empty:
        h.to_parquet(cache, index=False)
    return h


def _aggregate_val_acc_from_runs(runs: list, entity: str, project: str, metric: str) -> pd.DataFrame:
    series = []
    for r in runs:
        h = _history_cached(r, entity, project)
        if h.empty or metric not in h.columns or "_step" not in h.columns:
            continue
        g = h.dropna(subset=[metric]).sort_values("_step")
        if g.empty:
            continue
        # W&B can contain duplicate rows at the same step for one run.
        s = g.groupby("_step", sort=True)[metric].last()
        if s.empty:
            continue
        series.append(s.rename(r.id))
    if not series:
        return pd.DataFrame()
    wide = pd.concat(series, axis=1)
    count = wide.count(axis=1)
    std = wide.std(axis=1)
    ci95 = 1.96 * (std / np.sqrt(np.maximum(count, 1)))
    out = pd.DataFrame({
        "_step": wide.index,
        "mean": wide.mean(axis=1),
        "std": std,
        "count": count,
        "ci95": ci95.fillna(0.0),
    }).reset_index(drop=True)
    return out.dropna(subset=["mean"])


def _aggregate_val_acc_from_paths(paths: list, metric: str) -> pd.DataFrame:
    series = []
    for path in paths:
        try:
            h = pd.read_parquet(path)
        except Exception:
            continue
        if h.empty or metric not in h.columns or "_step" not in h.columns:
            continue
        g = h.dropna(subset=[metric]).sort_values("_step")
        if g.empty:
            continue
        # W&B can contain duplicate rows at the same step for one run.
        s = g.groupby("_step", sort=True)[metric].last()
        if s.empty:
            continue
        series.append(s.rename(path.name))
    if not series:
        return pd.DataFrame()
    wide = pd.concat(series, axis=1)
    count = wide.count(axis=1)
    std = wide.std(axis=1)
    ci95 = 1.96 * (std / np.sqrt(np.maximum(count, 1)))
    out = pd.DataFrame({
        "_step": wide.index,
        "mean": wide.mean(axis=1),
        "std": std,
        "count": count,
        "ci95": ci95.fillna(0.0),
    }).reset_index(drop=True)
    return out.dropna(subset=["mean"])


def _plot_frac(
    ax,
    *,
    entity: str,
    project: str,
    frac: float,
    seeds: Optional[list],
    baseline_wd: float,
    wd_values: list,
    canonical_lamh_values: list,
    max_step: Optional[int],
    metric: str,
    cache_only: bool,
) -> None:
    import wandb

    specs = []
    specs.append(("weight_decay_geometry", baseline_wd, None, rf"Baseline (WD$={baseline_wd:g}$)", COLORS[0]))
    wd_palette = plt.get_cmap("Blues")
    can_palette = plt.get_cmap("OrRd")
    wd_vals = [float(w) for w in wd_values if abs(float(w) - float(baseline_wd)) > 1e-10]
    can_vals = [float(b) for b in canonical_lamh_values]
    for i, wd in enumerate(wd_vals):
        color = wd_palette(0.45 + 0.5 * (i / max(1, len(wd_vals) - 1)))
        specs.append(("weight_decay_geometry", wd, None, rf"WD only (WD$={wd:g}$)", color))
    for i, lam_h in enumerate(can_vals):
        color = can_palette(0.4 + 0.55 * (i / max(1, len(can_vals) - 1)))
        specs.append(("canonical_regularizer", 0.0, lam_h, rf"Canonical $H$ ($\beta={lam_h:g}$)", color))

    api = None if cache_only else wandb.Api(timeout=120)
    ep = f"{entity}/{project}"

    for group, wd, lam_h, label, color in specs:
        if cache_only:
            paths = _pick_cached_paths(
                entity, project,
                frac=frac, seeds=seeds, group=group,
                weight_decay=wd if group == "weight_decay_geometry" else 0.0,
                lam_h=lam_h if group == "canonical_regularizer" else 0.0,
            )
            n_found = len(paths)
            agg = _aggregate_val_acc_from_paths(paths, metric)
        else:
            runs = _pick_runs_wandb(
                api,
                frac=frac, seeds=seeds, group=group,
                weight_decay=wd if group == "weight_decay_geometry" else 0.0,
                lam_h=lam_h if group == "canonical_regularizer" else 0.0,
                entity_project=ep,
            )
            n_found = len(runs)
            agg = _aggregate_val_acc_from_runs(runs, entity, project, metric)

        if agg.empty:
            print(f"[warn] no data: {label} frac={frac} (matched {n_found} runs)")
            continue
        if max_step is not None:
            agg = agg[agg["_step"] <= max_step]
        ax.plot(agg["_step"], agg["mean"], color=color, lw=2.0, label=label)
        ci = agg["ci95"].fillna(0.0)
        ax.fill_between(
            agg["_step"],
            np.clip(agg["mean"] - ci, 0.0, 1.0),
            np.clip(agg["mean"] + ci, 0.0, 1.0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default="regularizer")
    p.add_argument("--frac", type=float, default=None, help="single training fraction")
    p.add_argument("--fracs", type=float, nargs="*", default=None, help="multiple fractions → appendix grid")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict to these seeds (default: all)")
    p.add_argument("--baseline-wd", type=float, default=0.01)
    p.add_argument("--wd-only", type=float, default=0.1, dest="wd_only", help="deprecated alias for one WD value")
    p.add_argument("--canonical-lamh", type=float, default=0.1, help="deprecated alias for one canonical value")
    p.add_argument("--wd-values", type=float, nargs="*", default=None, help="multiple WD-only values")
    p.add_argument("--canonical-lamh-values", type=float, nargs="*", default=None, help="multiple canonical lambda_H values")
    p.add_argument("--max-step", type=int, default=None, help="clip x-axis upper bound")
    p.add_argument("--metric", default="val/accuracy")
    p.add_argument("--out-name", default="regularizer_three_way", help="basename under figures/images/")
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="only read analysis/runs_cache parquets — no W&B API (fast). "
        "Run figures/_wandb_fetch.py once to populate the cache.",
    )
    args = p.parse_args()

    fracs = args.fracs if args.fracs else ([args.frac] if args.frac is not None else [0.3])
    if args.frac is not None and args.fracs:
        p.error("use either --frac or --fracs, not both")

    wd_values = args.wd_values if args.wd_values else [args.wd_only]
    canonical_vals = args.canonical_lamh_values if args.canonical_lamh_values else [args.canonical_lamh]

    for frac in fracs:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        _plot_frac(
            ax,
            entity=args.entity,
            project=args.project,
            frac=frac,
            seeds=args.seeds,
            baseline_wd=args.baseline_wd,
            wd_values=wd_values,
            canonical_lamh_values=canonical_vals,
            max_step=args.max_step,
            metric=args.metric,
            cache_only=args.cache_only,
        )
        finish_axes(
            ax,
            "Training step",
            "Validation accuracy",
            rf"$f_{{\mathrm{{train}}}} = {frac:.3g}$",
        )
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        fig.suptitle(r"Baseline vs.\ WD-only vs.\ canonical $H$ penalties ($p=113$)", y=1.02, fontsize=13)
        fig.tight_layout()
        frac_tag = str(frac).replace(".", "p")
        out = save_fig(fig, f"{args.out_name}_frac{frac_tag}")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
