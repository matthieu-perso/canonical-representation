#!/usr/bin/env bash
# Build the task list for paper experiments 1, 2, 3.
# All three come from geometry_phase_transition runs — no regularizer, no noise.
# Sweeps frac_train over the original paper fractions, 5 seeds each.
#
# Usage:
#   ./grokking/experiments/build_exp123_tasks.sh
#   sbatch --array=1-$(wc -l < tasks_exp123.txt)%32 \
#          --export=ALL,TASKS_FILE=tasks_exp123.txt \
#          grokking/experiments/slurm_paper_experiments.sh
set -euo pipefail

OUT=tasks_exp123.txt
: > "$OUT"

SEEDS=(0 1 2 3 4)
FRACS=(0.1 0.15 0.2 0.25 0.3 0.4 0.5)
P=113
WD=0.01
MAX_STEPS=30000
EVAL_EVERY=200
MARGINS_EVERY=5
PROJECT=canonical_repr_exp123

COMMON="--p $P --max-steps $MAX_STEPS --eval-every $EVAL_EVERY \
--margins-every-n-evals $MARGINS_EVERY --wandb-project $PROJECT \
--weight-decay $WD"

for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    echo "canonical_geometry_experiments --experiment geometry_phase_transition \
$COMMON --frac-train $f --seed $s" >> "$OUT"
  done
done

N=$(wc -l < "$OUT")
echo "Wrote $N tasks to $OUT"
echo "Submit with:"
echo "  sbatch --array=1-${N}%32 --export=ALL,TASKS_FILE=$OUT \\"
echo "         grokking/experiments/slurm_paper_experiments.sh"
