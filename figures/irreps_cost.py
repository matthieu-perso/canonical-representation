import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────

def required_loss(gamma: np.ndarray) -> float:
    return float(np.log1p(np.sum(np.exp(-gamma))))

def precompute_basis(p: int) -> np.ndarray:
    ks = np.arange(1, (p - 1) // 2 + 1)[:, None]
    deltas = np.arange(1, p)[None, :]
    return 1.0 - np.cos(2.0 * np.pi * ks * deltas / p)

def minimal_irreps(B: np.ndarray, p: int, A: float, delta: float, max_m: int = 56):
    n_irreps = B.shape[0]
    available = np.ones(n_irreps, dtype=bool)
    running = np.zeros(p - 1)

    for m in range(1, min(max_m, n_irreps) + 1):
        candidates = np.where(available)[0]
        best_idx, best_loss = None, np.inf

        for idx in candidates:
            gamma = (A / m) * (running + B[idx])
            loss = required_loss(gamma)
            if loss < best_loss:
                best_loss = loss
                best_idx = idx

        available[best_idx] = False
        running += B[best_idx]

        if required_loss((A / m) * running) <= delta:
            return m

    return None

def wd_cost(A, m, lam=1.0):
    return lam * (A**2 / m)

def eff_cost(A, m, c=5.0):
    # Matches: C(A,K) ≈ A^2 / |K| + c |K|
    return (A**2 / m) + c * m


# ─────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────

p = 113
A_vals = np.arange(6, 26)
deltas = [0.1, 0.05, 0.02, 0.01, 0.005]
c_irrep = 5.0  # choose > 4 as in your text

B = precompute_basis(p)


# ─────────────────────────────────────────────────────────────
# Compute results
# ─────────────────────────────────────────────────────────────

results = {}
for d in deltas:
    ms = [minimal_irreps(B, p, A, d) for A in A_vals]
    results[d] = ms


# ─────────────────────────────────────────────────────────────
# Elegant plotting style
# ─────────────────────────────────────────────────────────────

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

# Elegant muted blue-gray palette, close to your screenshot
colors = ["#3B7DDD", "#5E9BD3", "#9BB7C9", "#6E737D", "#2F3338"]
markers = ["o", "s", "D", "^", "v"]


# ─────────────────────────────────────────────────────────────
# Helper: common curve plotting
# ─────────────────────────────────────────────────────────────

def finish_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#666666")
    ax.margins(x=0.03, y=0.08)


# ─────────────────────────────────────────────────────────────
# Plot 1: threshold frontier, A on x-axis
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.0, 5.0))

for color, marker, delta in zip(colors, markers, deltas):
    ms = results[delta]
    xs = [A for A, m in zip(A_vals, ms) if m is not None]
    ys = [m for m in ms if m is not None]

    ax.plot(
        xs, ys,
        color=color,
        linewidth=1.7,
        marker=marker,
        markersize=4.2,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=rf"$\delta = {delta}$"
    )

ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_xlim(A_vals[0] - 0.5, A_vals[-1] + 0.5)

finish_axes(
    ax,
    xlabel=r"Amplitude $A$",
    ylabel="Minimal Irreps",
    title=rf"Irrep threshold  —  $p = {p}$"
)

ax.legend(loc="upper right")
fig.tight_layout()
plt.savefig("threshold_vs_amplitude_elegant.png", dpi=300, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# Plot 2: effective cost with minimal irreps on x-axis
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.0, 5.0))

for color, marker, delta in zip(colors, markers, deltas):
    ms = results[delta]

    x_m = []
    y_cost = []

    for A, m in zip(A_vals, ms):
        if m is not None:
            x_m.append(m)
            y_cost.append(eff_cost(A, m, c=c_irrep))

    # sort by m for cleaner lines
    order = np.argsort(x_m)
    x_m = np.array(x_m)[order]
    y_cost = np.array(y_cost)[order]

    ax.plot(
        x_m, y_cost,
        color=color,
        linewidth=1.7,
        marker=marker,
        markersize=4.2,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=rf"$\delta = {delta}$"
    )

ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

finish_axes(
    ax,
    xlabel=r"Minimal number of irreps $m(A,\delta)$",
    ylabel=rf"Effective cost  $A^2/m + c\,m$",
    title=rf"Effective cost over feasible minimal irreps  —  $p = {p}$"
)

ax.legend(loc="upper right")
fig.tight_layout()
plt.savefig("cost_vs_minimal_irreps_elegant.png", dpi=300, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────
# Plot 3: effective cost vs amplitude, for comparison
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8.0, 5.0))

for color, marker, delta in zip(colors, markers, deltas):
    ms = results[delta]
    xs = []
    ys = []

    for A, m in zip(A_vals, ms):
        if m is not None:
            xs.append(A)
            ys.append(eff_cost(A, m, c=c_irrep))

    ax.plot(
        xs, ys,
        color=color,
        linewidth=1.7,
        marker=marker,
        markersize=4.2,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=rf"$\delta = {delta}$"
    )

ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

finish_axes(
    ax,
    xlabel=r"Amplitude $A$",
    ylabel=rf"Effective cost  $A^2/m(A,\delta) + c\,m(A,\delta)$",
    title=rf"Effective cost curves  —  $p = {p}$"
)

ax.legend(loc="upper left")
fig.tight_layout()
plt.savefig("cost_vs_amplitude_elegant.png", dpi=300, bbox_inches="tight")
plt.show()