# The Canonical Representation of a Task: The Case of Grokking

Code for the paper *"The Canonical Representation of a Task: The Case of Grokking"*.

> **Note:** Significant portions of this codebase reuse or adapt code from [Less Is More](https://github.com/aidos-lab/grokking-via-lid) repository (by Ben Ruppik et al.), which itself is a reimplementation of the original grokking paper by Power et al.


We study grokking on modular addition by deriving, from the group structure of the task, the **canonical representation** that a model must approximate to generalize, and the **representational deviation** that controls the generalization gap. The experiments here empirically validate the three theorems of the paper and show that directly penalizing representational deviation accelerates grokking.

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Repository structure

- `grokking/grokk_replica/` — transformer model and modular-arithmetic datasets.
- `grokking/geometry/canonical_alignment.py` — Fourier/irrep projection, canonical subspace $V^*$, representational deviation $H$, margin and excess-loss decompositions.
- `grokking/scripts/train_grokk.py` — baseline Hydra-configured grokking training run.
- `grokking/scripts/canonical_geometry_experiments.py` — the main paper experiments (see below).
- `grokking/scripts/fast_grokking_geometry_wandb.py` — short training run logging canonical alignment to Weights & Biases.
- `grokking/scripts/irrep_ce_analysis.py` — closed-form computation of Theorem 1 (minimal number of irreps for a target loss).
- `config/` — Hydra configs for datasets, model, and training.

## Running the experiments

All paper experiments are exposed as a single CLI:

```bash
# Sec. 5.1 — irreps appear in lock-step with validation-loss drops
uv run canonical_geometry_experiments --experiment geometry_phase_transition

# Sec. 5.3 — generalization gap vs. data fraction
uv run canonical_geometry_experiments --experiment data_threshold \
    --fracs 0.1,0.15,0.2,0.25,0.3,0.4,0.5

# Sec. 5.4 — weight decay drives representational deviation -> 0
uv run canonical_geometry_experiments --experiment weight_decay_geometry \
    --weight-decays 0,1e-4,1e-3,1e-2,1e-1,1.0

# Sec. 5.5 — penalizing ||H||^2 directly accelerates grokking
uv run canonical_geometry_experiments --experiment canonical_regularizer \
    --lambdas 0,0.01,0.1,1.0 --fracs 0.15,0.3

# Sec. 3.1 — robustness of a grokked model to logit noise
uv run canonical_geometry_experiments --experiment noise_robustness \
    --sigmas 0,0.5,1,2,5,10
```

Theorem 1 (minimal irreps $m^\dagger$ for target loss $\delta$, Table 1):

```bash
uv run python grokking/scripts/irrep_ce_analysis.py
```

A short W&B-logged grokking run (useful as a smoke test):

```bash
uv run python grokking/scripts/fast_grokking_geometry_wandb.py \
    --wandb-project canonical-representation --p 113 --max-steps 4000
```

Standard training (Hydra):

```bash
uv run train_grokk
```

W&B is optional — set `WANDB_MODE=offline` or `WANDB_MODE=disabled` to run locally.


