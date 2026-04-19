#!/usr/bin/env bash
# Generate one CLI command per line for the SLURM array launcher.
# Output: tasks_paper_experiments.txt (overwritten on each run).
#
# Usage: ./grokking/experiments/build_paper_tasks.sh
#        sbatch --array=1-$(wc -l < tasks_paper_experiments.txt) \
#               grokking/experiments/slurm_paper_experiments.sh
set -euo pipefail

OUT=tasks_paper_experiments.txt
: > "$OUT"

SEEDS=(0 1 2 3 4)            # 5 seeds → 95% CIs with t-distribution (df=4, t=2.78)
P=113
MAX_STEPS=30000
EVAL_EVERY=200
MARGINS_EVERY=5
PROJECT=canonical_repr_grokking

# Common flags shared across experiments
COMMON="--p $P --max-steps $MAX_STEPS --eval-every $EVAL_EVERY \
--margins-every-n-evals $MARGINS_EVERY --wandb-project $PROJECT"

# ---------------------------------------------------------------------------
# Exp 1 — geometry phase transition (1 frac × 5 seeds = 5 runs)
# ---------------------------------------------------------------------------
for s in "${SEEDS[@]}"; do
  echo "canonical_geometry_experiments --experiment geometry_phase_transition \
$COMMON --frac-train 0.35 --weight-decay 0.01 --seed $s" >> "$OUT"
done

# ---------------------------------------------------------------------------
# Exp 4 — weight-decay sweep (8 WD × 5 seeds = 40 runs)
# ---------------------------------------------------------------------------
WDS=(0 0.0001 0.001 0.01 0.03 0.1 0.3 1.0)
for s in "${SEEDS[@]}"; do
  for wd in "${WDS[@]}"; do
    echo "canonical_geometry_experiments --experiment weight_decay_geometry \
$COMMON --frac-train 0.30 --seed $s --weight-decays $wd" >> "$OUT"
  done
done

# ---------------------------------------------------------------------------
# Exp 5a — weight-side regulariser ‖Π_⊥ W_L‖²  (2 fracs × 5 λ_W × 5 seeds = 50)
# ---------------------------------------------------------------------------
LAMW=(0 0.01 0.1 1.0 10.0)
FRACS=(0.15 0.20)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMW[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks 1 --seed $s \
--lambdas $lam --lambdas-h 0" >> "$OUT"
    done
  done
done

# ---------------------------------------------------------------------------
# Exp 5b — logit-side regulariser β‖H‖² (2 fracs × 4 λ_H × 5 seeds = 40)
# ---------------------------------------------------------------------------
LAMH=(0.001 0.01 0.1 1.0)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMH[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks 1 --seed $s \
--lambdas 0 --lambdas-h $lam" >> "$OUT"
    done
  done
done

# ---------------------------------------------------------------------------
# Exp 5c (optional) — alternative irrep V* = V_17, tests universality of
# §2.3 claim "any single non-trivial irrep suffices"  (2 × 4 × 5 = 40)
# ---------------------------------------------------------------------------
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMH[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks 17 --seed $s \
--lambdas 0 --lambdas-h $lam" >> "$OUT"
    done
  done
done

N=$(wc -l < "$OUT")
echo "Wrote $N tasks to $OUT"
echo "Submit with:"
echo "  sbatch --array=1-${N}%32 grokking/experiments/slurm_paper_experiments.sh"
echo "(the %32 caps concurrent jobs; raise/lower for your fairshare)"
