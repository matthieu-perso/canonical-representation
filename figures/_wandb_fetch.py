"""Cache W&B run history locally as parquet for fast offline plotting.

Usage:
    from figures._wandb_fetch import load_runs

    df = load_runs(experiment="geometry_phase_transition")   # → tidy long-form DataFrame

The first call downloads from wandb (cloud) and writes one parquet per run under
analysis/runs_cache/<entity>__<project>/<run_id>.parquet.  Subsequent calls hit
the cache; pass refresh=True to force redownload.

Each per-run parquet contains the full step history (one row per logged step,
columns = config keys ∪ logged metrics) plus a `_step` index.

Run as a script to pre-warm the cache:
    python figures/_wandb_fetch.py                                          # all runs in default project
    python figures/_wandb_fetch.py --experiment geometry_phase_transition   # filter
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Defaults: the project that holds the bulk of finished runs.
# ---------------------------------------------------------------------------
DEFAULT_ENTITY = "matthieu-perso"
DEFAULT_PROJECT = "canonical_repr_grokking"

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = REPO_ROOT / "analysis" / "runs_cache"


def _project_cache_dir(entity: str, project: str) -> Path:
    d = CACHE_ROOT / f"{entity}__{project}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _config_to_dict(config) -> dict:
    """wandb.Run.config → flat dict of scalar values."""
    out = {}
    for k, v in dict(config).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v)
            except Exception:
                out[k] = str(v)
    return out


def _experiment_of(cfg: dict) -> str | None:
    """Recover the --experiment CLI flag from a run's config or args."""
    if "experiment" in cfg:
        return cfg["experiment"]
    args = cfg.get("args") or cfg.get("argv")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    if isinstance(args, list):
        for i, a in enumerate(args):
            if a == "--experiment" and i + 1 < len(args):
                return args[i + 1]
    return None


def fetch_run_history(
    run,
    *,
    samples: int | None = None,
) -> pd.DataFrame:
    """Pull full step history for one wandb Run, including config columns."""
    cfg = _config_to_dict(run.config)
    cfg.setdefault("experiment", _experiment_of(cfg))

    # scan_history streams the *full* logged table; history(samples=...) downsamples.
    if samples is None:
        rows = list(run.scan_history())
    else:
        rows = list(run.history(samples=samples, pandas=False, stream="default"))

    if not rows:
        return pd.DataFrame()
    h = pd.DataFrame(rows)
    if "_step" not in h.columns:
        h["_step"] = range(len(h))

    h["run_id"] = run.id
    h["run_name"] = run.name
    h["run_state"] = run.state
    for k, v in cfg.items():
        col = f"cfg.{k}"
        if col not in h.columns:
            h[col] = v
    return h


def cache_project(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    *,
    states: Iterable[str] = ("finished", "running", "crashed"),
    samples: int | None = None,
    refresh: bool = False,
    limit: int | None = None,
) -> list[Path]:
    """Download (or refresh) parquet cache for a wandb project.  Returns list of files."""
    cache_dir = _project_cache_dir(entity, project)
    api = wandb.Api(timeout=60)

    # Note: `experiment` is not stored in run.config for this project — runs are
    # identified post-hoc by (frac_train, weight_decay, lambda_canonical) tuples.
    # We just pull everything and let callers slice.
    filters: dict = {"state": {"$in": list(states)}}
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=500)
    paths: list[Path] = []
    n = 0
    for run in runs:
        if limit is not None and n >= limit:
            break
        out = cache_dir / f"{run.id}.parquet"
        if out.exists() and not refresh:
            paths.append(out)
            n += 1
            continue
        try:
            h = fetch_run_history(run, samples=samples)
        except Exception as e:
            print(f"[skip {run.id}] {type(e).__name__}: {e}")
            continue
        if h.empty:
            print(f"[skip {run.id}] empty history")
            continue
        h.to_parquet(out, index=False)
        cfg_summary = (
            f"exp={h.get('cfg.experiment', pd.Series([None]*len(h))).iloc[0]} "
            f"frac={h.get('cfg.frac_train', pd.Series([None]*len(h))).iloc[0]} "
            f"wd={h.get('cfg.weight_decay', pd.Series([None]*len(h))).iloc[0]} "
            f"seed={h.get('cfg.seed', pd.Series([None]*len(h))).iloc[0]}"
        )
        print(f"[ok {run.id}] {len(h):>6d} steps  {cfg_summary}")
        paths.append(out)
        n += 1
    return paths


def load_runs(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    *,
    refresh: bool = False,
    auto_fetch: bool = True,
) -> pd.DataFrame:
    """Load all cached runs in a project as one tidy DataFrame.

    If the cache is empty (and auto_fetch=True) it downloads everything once.
    """
    cache_dir = _project_cache_dir(entity, project)
    if refresh or (auto_fetch and not any(cache_dir.glob("*.parquet"))):
        cache_project(entity, project, refresh=refresh)

    frames: list[pd.DataFrame] = []
    for p in sorted(cache_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[skip {p.name}] {e}")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Canonical experiment classifier — recovers the conceptual experiment name
# from each run's (frac_train, weight_decay, lambda_canonical) tuple.
# Matches grokking/experiments/build_paper_tasks.sh.
# ---------------------------------------------------------------------------
def classify_experiment(row) -> str:
    """Return one of: phase_transition | weight_decay | regulariser | other."""
    f = row.get("cfg.frac_train")
    wd = row.get("cfg.weight_decay")
    lam = row.get("cfg.lambda_canonical") or 0
    try:
        f = float(f); wd = float(wd); lam = float(lam)
    except (TypeError, ValueError):
        return "other"
    if abs(f - 0.35) < 1e-6 and abs(wd - 0.01) < 1e-6 and lam == 0:
        return "phase_transition"
    if abs(f - 0.30) < 1e-6 and lam == 0:
        return "weight_decay"
    if f in (0.15, 0.20) and wd == 0:
        return "regulariser"
    return "other"


def add_experiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a derived `experiment` column by classifying each row."""
    if df.empty:
        return df
    df = df.copy()
    df["experiment"] = df.apply(classify_experiment, axis=1)
    return df


def dedupe_run_history_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate ``_step`` rows per ``run_id`` (W&B can log a step twice)."""
    if df.empty or "run_id" not in df.columns or "_step" not in df.columns:
        return df
    return df.sort_values(["run_id", "_step"]).drop_duplicates(["run_id", "_step"], keep="last")


def filter_geometry_phase_transition(
    df: pd.DataFrame,
    *,
    frac_train: float,
    weight_decay: float = 0.01,
) -> pd.DataFrame:
    """Keep rows from ``geometry_phase_transition``-style runs (no canonical penalty).

    Use this for paper experiments 1–3 instead of ``classify_experiment`` when your W&B
    project uses arbitrary ``frac_train`` (e.g. ``canonical_repr_exp123`` sweep).
    """
    if df.empty:
        return df
    out = df.copy()
    ft = out["cfg.frac_train"].astype(float)
    wd = out["cfg.weight_decay"].astype(float)
    lc = out.get("cfg.lambda_canonical")
    lc = lc.fillna(0).astype(float) if lc is not None else pd.Series(0.0, index=out.index)
    lh = out.get("cfg.lambda_canonical_H")
    lh = lh.fillna(0).astype(float) if lh is not None else pd.Series(0.0, index=out.index)
    m = (ft - float(frac_train)).abs() < 1e-5
    m &= (wd - float(weight_decay)).abs() < 1e-6
    m &= lc == 0
    m &= lh == 0
    return dedupe_run_history_rows(out.loc[m])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--refresh", action="store_true",
                   help="redownload even if a cache file already exists")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of runs fetched (debug)")
    p.add_argument("--samples", type=int, default=None,
                   help="downsample history (None ⇒ full scan, slower but exact)")
    return p.parse_args()


def main():
    args = _parse()
    paths = cache_project(
        args.entity, args.project,
        refresh=args.refresh,
        limit=args.limit,
        samples=args.samples,
    )
    print(f"\ncached {len(paths)} runs into {_project_cache_dir(args.entity, args.project)}")


if __name__ == "__main__":
    main()
