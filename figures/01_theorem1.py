"""Figure 2 — §4.1 Phase Transition & Spectrum Collapse.

Produces **two separate** images per call (matching the paper's requested layout):

  (a) ``02_active_irreps_<frac>`` — empirical active irrep count (left y-axis) and
      validation accuracy / loss (right y-axis) vs. training step (linear scale),
      averaged over seeds with shaded ±1 SD bands, plus a horizontal reference at
      the Theorem-1 prediction m†.

  (b) ``02_spectrum_<frac>``      — per-irrep amplitudes α_k^φ(t) for k = 1..(p-1)/2
      at the median-seed run (linear scale on both axes).  Spectrum collapse:
      ~top-K modes survive on V*, the rest stay dead.

Pass ``--frac 0.3`` (or similar) for a single fraction, or ``--all-fracs`` to loop
over every fraction present in the cache and write one pair of images per fraction.

Examples:
    # One fraction → two PNG/PDF pairs in figures/images/
    uv run python figures/02_phase_transition_spectrum.py \\
        --project canonical_repr_exp123 --frac 0.3

    # Every fraction in the cache → 2 images × N fractions
    uv run python figures/02_phase_transition_spectrum.py \\
        --project canonical_repr_exp123 --all-fracs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from figures._style import ACCENT, COLORS, apply_style, finish_axes, save_fig
from figures._wandb_fetch import DEFAULT_ENTITY, DEFAULT_PROJECT, filter_geometry_phase_transition, load_runs


apply_style()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate_over_seeds(df_runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return mean/std of ``metric`` across runs at each ``_step``."""
    g = (
        df_runs.dropna(subset=[metric])
        .groupby("_step", sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return g


def _frac_tag(frac: float) -> str:
    return f"frac{frac:.3g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Panel (a) — active irreps vs. val accuracy / loss
# ---------------------------------------------------------------------------
def plot_active_irreps(
    pt: pd.DataFrame,
    *,
    frac: float,
    left_metric: str,
    right_metric: str,
    out_name: str,
) -> Path:
    left_agg = aggregate_over_seeds(pt, left_metric)
    right_agg = aggregate_over_seeds(pt, right_metric)

    if left_agg.empty or right_agg.empty:
        raise SystemExit(
            f"Missing metric: {left_metric if left_agg.empty else right_metric} "
            "(check geometry logging)."
        )

    p_val = int(pt["cfg.p"].dropna().iloc[0])
    KMAX = (p_val - 1) // 2

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax_r = ax.twinx()
    ax_r.grid(False)

    left_color = COLORS[0]
    right_color = ACCENT

    l1, = ax.plot(left_agg["_step"], left_agg["mean"],
                  color=left_color, lw=2.0, label="Active irreps")
    ax.fill_between(
        left_agg["_step"],
        (left_agg["mean"] - left_agg["std"].fillna(0)).clip(lower=0),
        left_agg["mean"] + left_agg["std"].fillna(0),
        color=left_color, alpha=0.18, linewidth=0,
    )

    r_hi = right_agg["mean"] + right_agg["std"].fillna(0)
    r_lo = right_agg["mean"] - right_agg["std"].fillna(0)
    if right_metric == "val/accuracy":
        r_hi = r_hi.clip(upper=1.0)
        r_lo = r_lo.clip(lower=0.0)
    right_label = {"val/accuracy": "Validation accuracy", "val/loss": "Validation loss"}.get(
        right_metric, right_metric
    )
    l2, = ax_r.plot(right_agg["_step"], right_agg["mean"],
                    color=right_color, lw=2.0, ls="--", label=right_label)
    ax_r.fill_between(right_agg["_step"], r_lo, r_hi, color=right_color, alpha=0.15, linewidth=0)

    # Theorem-1 horizontal reference (m†).
    if "geometry/K_theory_n" in pt.columns:
        ks = pt["geometry/K_theory_n"].dropna()
        m_dag = int(ks.iloc[0]) if len(ks) else 6
    else:
        m_dag = 6
    ax.axhline(m_dag, color="#555555", lw=1.0, ls=":", alpha=0.8)
    # Place label on the *inside* of the left axis so it cannot overlap the right-axis ticks.
    ax.annotate(
        fr"$m^\dagger = {m_dag}$",
        xy=(0.015, m_dag), xycoords=("axes fraction", "data"),
        ha="left", va="bottom", color="#555555", fontsize=10,
    )

    smin = max(0, int(min(left_agg["_step"].min(), right_agg["_step"].min())))
    smax = int(max(left_agg["_step"].max(), right_agg["_step"].max()))
    ax.set_xlim(smin, smax)

    ax.set_ylabel("Active irreps", color=left_color)
    ax.tick_params(axis="y", colors=left_color)
    if left_metric in ("geometry/logit_n_active_irreps", "geometry/K_min_theoretical"):
        ax.set_ylim(0, KMAX + 2)

    ax_r.set_ylabel(right_label, color=right_color)
    ax_r.tick_params(axis="y", colors=right_color)
    if right_metric == "val/accuracy":
        ax_r.set_ylim(-0.02, 1.05)

    finish_axes(
        ax,
        xlabel="Training step",
        ylabel="Active irreps",
        title=rf"Irreps vs.\ generalisation ($p={p_val}$, $f_{{\mathrm{{train}}}}={frac:g}$)",
    )
    ax.legend(handles=[l1, l2], loc="center right")

    fig.tight_layout()
    out = save_fig(fig, out_name)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Panel (b) — per-irrep amplitude trajectories (linear–linear)
# ---------------------------------------------------------------------------
def plot_spectrum(
    pt: pd.DataFrame,
    *,
    frac: float,
    top_k: int,
    out_name: str,
) -> Path:
    final = (
        pt.dropna(subset=["val/accuracy"])
        .sort_values("_step")
        .groupby("run_id")
        .tail(1)[["run_id", "val/accuracy", "cfg.seed"]]
        .reset_index(drop=True)
    )
    median_run = final.sort_values("val/accuracy").iloc[len(final) // 2]["run_id"]
    seed_run = pt[pt["run_id"] == median_run].sort_values("_step").copy()

    p_val = int(seed_run["cfg.p"].dropna().iloc[0])
    n_modes = (p_val - 1) // 2

    alpha_cols = [f"geometry/phi_alpha_{k}" for k in range(1, n_modes + 1)]
    alpha_cols = [c for c in alpha_cols if c in seed_run.columns]
    if not alpha_cols:
        raise SystemExit("No geometry/phi_alpha_* columns in cache — margins not logged?")

    final_alphas = seed_run[alpha_cols].dropna().iloc[-1]
    order = np.argsort(final_alphas.values)[::-1]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cmap_top = plt.get_cmap("viridis")
    for rank, idx in enumerate(order):
        col = alpha_cols[idx]
        k = int(col.split("_")[-1])
        series = seed_run[["_step", col]].dropna()
        if rank < top_k:
            color = cmap_top(rank / max(1, top_k - 1))
            lw, alpha, zorder, label = 1.6, 0.95, 3, fr"$k={k}$"
        else:
            color = "#9aa1a8"
            lw, alpha, zorder, label = 0.6, 0.35, 1, None
        ax.plot(series["_step"], series[col],
                color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)

    ax.set_xlim(0, int(seed_run["_step"].max()))
    ax.set_ylim(bottom=0)

    finish_axes(
        ax,
        xlabel="Training step",
        ylabel=r"Per-irrep amplitude  $\alpha_k^{\phi}$",
        title=(
            fr"Spectrum collapse on $V^*$: top-{top_k} modes survive  "
            fr"($f_{{\mathrm{{train}}}}={frac:g}$, seed $={int(seed_run['cfg.seed'].iloc[0])}$)"
        ),
    )
    ax.legend(title=fr"top-{top_k} modes", loc="upper right",
              ncol=2, fontsize=9, title_fontsize=9)

    fig.tight_layout()
    out = save_fig(fig, out_name)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project (e.g. canonical_repr_exp123)")
    ap.add_argument("--frac", type=float, default=None, help="training fraction to plot (skip if --all-fracs)")
    ap.add_argument("--all-fracs", action="store_true",
                    help="loop over every frac_train present in the cache")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--left-metric", default="geometry/logit_n_active_irreps")
    ap.add_argument("--right-metric", default="val/accuracy", choices=["val/accuracy", "val/loss"])
    ap.add_argument("--top-k", type=int, default=7)
    args = ap.parse_args()

    if not args.all_fracs and args.frac is None:
        ap.error("Pass either --frac <F> or --all-fracs.")

    df = load_runs(entity=args.entity, project=args.project)
    if df.empty:
        raise SystemExit(
            "Cache empty. Run: "
            f"uv run python figures/_wandb_fetch.py --entity {args.entity} --project {args.project}"
        )

    avail = sorted({round(float(x), 4) for x in df["cfg.frac_train"].dropna().unique()})
    fracs = avail if args.all_fracs else [args.frac]

    for frac in fracs:
        pt = filter_geometry_phase_transition(df, frac_train=frac, weight_decay=args.weight_decay)
        if pt.empty:
            print(f"[skip] frac={frac}: no rows matching wd={args.weight_decay}")
            continue
        print(f"[frac={frac}] {pt['run_id'].nunique()} runs, {len(pt)} rows")

        tag = _frac_tag(frac)
        a_out = plot_active_irreps(
            pt,
            frac=frac,
            left_metric=args.left_metric,
            right_metric=args.right_metric,
            out_name=f"02_active_irreps_{tag}",
        )
        b_out = plot_spectrum(
            pt,
            frac=frac,
            top_k=args.top_k,
            out_name=f"02_spectrum_{tag}",
        )
        print(f"  wrote {a_out}")
        print(f"  wrote {b_out}")


if __name__ == "__main__":
    main()
