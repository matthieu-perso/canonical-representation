"""Shared matplotlib style for all paper figures.

Matches `figures/irreps_cost.py` so every figure in the paper looks like a family.
Import:
    from figures._style import apply_style, COLORS, MARKERS, finish_axes, save_fig
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Project-relative output dir (figures/images/).
IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    """Apply the elegant serif rcParams used across the paper."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d0d0",
    })


# Muted blue-gray palette + accent red for "off" / failure states.
COLORS = ["#3B7DDD", "#5E9BD3", "#9BB7C9", "#6E737D", "#2F3338"]
ACCENT = "#C44E52"   # contrast color for ε / off-canonical energy
MARKERS = ["o", "s", "D", "^", "v"]


def finish_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#666666")
    ax.margins(x=0.03, y=0.08)


def save_fig(fig, name: str, *, dpi: int = 300) -> Path:
    """Save figure to figures/images/<name>.{png,pdf}.  Returns the PNG path."""
    name = name.removesuffix(".png").removesuffix(".pdf")
    png = IMAGES_DIR / f"{name}.png"
    pdf = IMAGES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png
