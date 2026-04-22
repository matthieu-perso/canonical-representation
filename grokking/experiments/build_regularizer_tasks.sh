#!/usr/bin/env bash
# Build the task list comparing weight-decay runs vs canonical-H-penalty runs.
# All runs share p, max_steps, and seeds, differing only in the regulariser.
# Sweeps frac_train over all paper fractions. W&B project = "regularizer".
#
# Usage:
#   ./grokking/experiments/build_regularizer_tasks.sh
#   sbatch --array=1-$(wc -l < tasks_regularizer.txt)%32 \
#          --export=ALL,TASKS_FILE=tasks_regularizer.txt \
#          grokking/experiments/slurm_paper_experiments.sh
set -euo pipefail

OUT=tasks_regularizer.txt
: > "$OUT"

SEEDS=(0 1 2)
FRACS=(0.1 0.15 0.2 0.25 0.3 0.4 0.5)
P=113
MAX_STEPS=30000
EVAL_EVERY=200
MARGINS_EVERY=5
PROJECT=regularizer

COMMON="--p $P --max-steps $MAX_STEPS --eval-every $EVAL_EVERY \
--margins-every-n-evals $MARGINS_EVERY --wandb-project $PROJECT"

# --- Arm A: weight-decay sweep (no canonical penalty) -----------------------
WDS=(0 0.001 0.01 0.1 1.0)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for wd in "${WDS[@]}"; do
      echo "canonical_geometry_experiments --experiment weight_decay_geometry \
$COMMON --frac-train $f --seed $s --weight-decays $wd" >> "$OUT"
    done
  done
done

# --- Arm B: canonical H-penalty sweep (canonical_regularizer hardcodes WD=0) -
LAMH=(0 0.001 0.01 0.1 1.0)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMH[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks 1 --seed $s \
--lambdas 0 --lambdas-h $lam" >> "$OUT"
    done
  done
done

N=$(wc -l < "$OUT")
echo "Wrote $N tasks to $OUT"
echo "Submit with:"
echo "  sbatch --array=1-${N}%32 --export=ALL,TASKS_FILE=$OUT \\"
echo "         grokking/experiments/slurm_paper_experiments.sh"
