"""Theorem 3 — explicit gap decomposition.

Main decomposition plot (same step axis):
  • ΔCE_matched(t) = E[CE(z)] - E_S[CE(z)]                    (matched-set CE gap)
  • Δd_L(t)        = E[d_L(H)] - E_S[d_L(H)]                 (deviation-gap term)
  • ΔCE*(t)        = E[CE(z*)] - E_S[CE(z*)]                 (canonical-loss-gap term)

With CE(z*) reconstructed from logged means:
  CE*(full)  = CE(full)  - d_L(full)
  CE*(train) = CE(train) - d_L(train)
  so ΔCE*(t) = (CE_full - dL_full) - (CE_train - dL_train)

Also overlays the split-based empirical gap:
  • gap_val-train(t) = val/loss - train/loss

Optional component curves can be shown with ``--show-components``:
  • E_S[d_L(H)] and E[d_L(H)]

By default, this script aggregates over selected seeds and plots mean ± 95% CI.
It can render one fraction or multiple fractions in one command, saving one file
per fraction directly into ``figures/images``.

Examples:
    # one fraction, selected seeds
    uv run python figures/plot_theorem3_deviation_costs.py \\
        --project canonical_repr_exp123 --frac 0.3 --seeds 0 1 2 3 4

    # all fractions found in cache, selected seeds
    uv run python figures/plot_theorem3_deviation_costs.py \\
        --project canonical_repr_exp123 --all-fracs --seeds 0 1 2 3 4
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


def _frac_tag(frac: float) -> str:
    return str(frac).replace(".", "p")


def _agg_ci(pt: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = (
        pt.dropna(subset=[metric])
        .groupby("_step", sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if g.empty:
        return g
    g["ci95"] = 1.96 * (g["std"].fillna(0.0) / np.sqrt(np.maximum(g["count"], 1)))
    return g


def _plot_one_fraction(
    df_all: pd.DataFrame,
    frac: float,
    weight_decay: float,
    seeds: list[int] | None,
    *,
    show_components: bool,
) -> None:
    df = filter_geometry_phase_transition(df_all, frac_train=frac, weight_decay=weight_decay)
    if seeds is not None:
        df = df[df["cfg.seed"].isin(seeds)]

    need = [
        "geometry/d_L_train_mean",
        "geometry/d_L_full_mean",
        "geometry/ce_train_mean",
        "geometry/ce_full_mean",
        "train/loss",
        "val/loss",
    ]
    pt = df.dropna(subset=need).copy()
    if pt.empty:
        print(f"[skip] frac={frac}: no rows after filtering.")
        return

    pt["gap"] = pt["val/loss"] - pt["train/loss"]
    pt["gap_ce_matched"] = pt["geometry/ce_full_mean"] - pt["geometry/ce_train_mean"]
    pt["delta_dL"] = pt["geometry/d_L_full_mean"] - pt["geometry/d_L_train_mean"]
    if "geometry/d_L_gap" in pt.columns:
        # Use directly logged value when available (same object as delta_dL by definition).
        m = pt["geometry/d_L_gap"].notna()
        pt.loc[m, "delta_dL"] = pt.loc[m, "geometry/d_L_gap"]
    pt["canonical_full_mean"] = pt["geometry/ce_full_mean"] - pt["geometry/d_L_full_mean"]
    pt["canonical_train_mean"] = pt["geometry/ce_train_mean"] - pt["geometry/d_L_train_mean"]
    pt["delta_canonical"] = pt["canonical_full_mean"] - pt["canonical_train_mean"]
    train_agg = _agg_ci(pt, "geometry/d_L_train_mean")
    full_agg = _agg_ci(pt, "geometry/d_L_full_mean")
    delta_agg = _agg_ci(pt, "delta_dL")
    canon_delta_agg = _agg_ci(pt, "delta_canonical")
    gap_ce_agg = _agg_ci(pt, "gap_ce_matched")
    gap_agg = _agg_ci(pt, "gap")

    if delta_agg.empty or canon_delta_agg.empty or gap_ce_agg.empty:
        print(f"[skip] frac={frac}: missing required metrics.")
        return

    n_runs = int(pt["run_id"].nunique())
    used_seeds = sorted({int(s) for s in pt["cfg.seed"].dropna().astype(int).unique()})

    fig, ax = plt.subplots(figsize=(8.2, 4.9))

    # Main theorem-3 decomposition.
    ax.plot(
        gap_ce_agg["_step"],
        gap_ce_agg["mean"],
        color="#111111",
        lw=2.3,
        label=r"$\Delta \mathrm{CE}_{\mathrm{matched}}$",
    )
    ax.fill_between(
        gap_ce_agg["_step"],
        gap_ce_agg["mean"] - gap_ce_agg["ci95"],
        gap_ce_agg["mean"] + gap_ce_agg["ci95"],
        color="#444444",
        alpha=0.10,
        linewidth=0,
    )

    ax.plot(
        delta_agg["_step"],
        delta_agg["mean"],
        color=COLORS[2],
        lw=2.2,
        ls="--",
        label=r"$\Delta d_L = \mathbb{E}[d_L]-\mathbb{E}_S[d_L]$",
    )
    ax.fill_between(
        delta_agg["_step"],
        delta_agg["mean"] - delta_agg["ci95"],
        delta_agg["mean"] + delta_agg["ci95"],
        color=COLORS[2],
        alpha=0.17,
        linewidth=0,
    )

    ax.plot(
        canon_delta_agg["_step"],
        canon_delta_agg["mean"],
        color=COLORS[0],
        lw=2.0,
        label=r"$\Delta \mathrm{CE}^{*} = \mathbb{E}[\mathrm{CE}(z^*)]-\mathbb{E}_S[\mathrm{CE}(z^*)]$",
    )
    ax.fill_between(
        canon_delta_agg["_step"],
        canon_delta_agg["mean"] - canon_delta_agg["ci95"],
        canon_delta_agg["mean"] + canon_delta_agg["ci95"],
        color=COLORS[0],
        alpha=0.12,
        linewidth=0,
    )

    # Optional split-based (val-train) gap, shown as contextual reference.
    if not gap_agg.empty:
        ax.plot(
            gap_agg["_step"],
            gap_agg["mean"],
            color=ACCENT,
            lw=1.5,
            alpha=0.9,
            ls=":",
            label=r"$\mathcal{R}_{\mathrm{val}}-\mathcal{R}_{\mathrm{train}}$ (split gap)",
        )

    if show_components and not train_agg.empty and not full_agg.empty:
        ax.plot(
            train_agg["_step"], train_agg["mean"],
            color=COLORS[0], lw=1.4, alpha=0.9,
            label=r"$\mathbb{E}_S[d_L]$",
        )
        ax.plot(
            full_agg["_step"], full_agg["mean"],
            color="#6f8f9d", lw=1.4, alpha=0.9,
            label=r"$\mathbb{E}[d_L]$",
        )

    ax.axhline(0.0, color="#999999", lw=0.8, alpha=0.7)
    finish_axes(
        ax,
        "Training step",
        r"Gap / deviation-gap",
        rf"Theorem 3 decomposition ($f_{{\mathrm{{train}}}}={frac:g}$, runs={n_runs})",
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = save_fig(fig, f"theorem3_decomposition_frac{_frac_tag(frac)}")
    print(f"wrote {out}  (seeds={used_seeds})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Theorem 3 — deviation costs + gap.")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--frac", type=float, default=None, help="single fraction")
    ap.add_argument("--fracs", type=float, nargs="*", default=None, help="multiple fractions")
    ap.add_argument("--all-fracs", action="store_true", help="use all fractions found in cache")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict to these seeds (e.g. 0 1 2 3 4)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--show-components", action="store_true", help="also draw E_S[d_L] and E[d_L] component curves")
    args = ap.parse_args()

    if args.frac is not None and args.fracs:
        ap.error("use either --frac or --fracs, not both")
    if args.all_fracs and (args.frac is not None or args.fracs):
        ap.error("use --all-fracs alone, or set --frac/--fracs")

    df_all = load_runs(entity=args.entity, project=args.project)
    if df_all.empty:
        raise SystemExit("No cached rows — run figures/_wandb_fetch.py first.")

    if args.all_fracs:
        fracs = sorted({float(x) for x in df_all["cfg.frac_train"].dropna().unique()})
    elif args.fracs:
        fracs = [float(f) for f in args.fracs]
    elif args.frac is not None:
        fracs = [float(args.frac)]
    else:
        fracs = [0.3]

    for frac in fracs:
        _plot_one_fraction(
            df_all,
            frac=float(frac),
            weight_decay=args.weight_decay,
            seeds=args.seeds,
            show_components=args.show_components,
        )


if __name__ == "__main__":
    main()
