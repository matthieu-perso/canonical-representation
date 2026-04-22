#!/usr/bin/env bash
# Build the V*=top-K canonical-regulariser rerun (replaces the broken V*=V_1 sweep).
#
# Why top-K?  The cost minimum K* for p=113, delta=0.02 is K≈5-7 (Theorem 1).
# Constraining V*=V_1 in the previous sweep killed the network's capacity
# (val_acc ≤ 0.56). Setting V*=V_1 ⊕ ... ⊕ V_7 gives the network enough room
# to find the cost-minimum representation while still penalising off-canonical
# energy.
#
# Output: tasks_regulariser_topk.txt  (separate from tasks_paper_experiments.txt)
#
# Submit with:
#   N=$(wc -l < tasks_regulariser_topk.txt)
#   sbatch --array=1-${N}%32 \
#          --export=ALL,TASKS_FILE=tasks_regulariser_topk.txt \
#          grokking/experiments/slurm_paper_experiments.sh

set -euo pipefail

OUT=tasks_regulariser_topk.txt
: > "$OUT"

SEEDS=(0 1 2 3 4)
P=113
MAX_STEPS=30000
EVAL_EVERY=200
MARGINS_EVERY=5
PROJECT=canonical_representation_paper
KS="1,2,3,4,5,6,7"            # V* = ⊕_{k=1}^{7} V_k (covers cost-minimum K*)
FRACS=(0.15 0.20)             # below n* (0.15) and above (0.20)

COMMON="--p $P --max-steps $MAX_STEPS --eval-every $EVAL_EVERY \
--margins-every-n-evals $MARGINS_EVERY --wandb-project $PROJECT"

# ---------------------------------------------------------------------------
# Exp 5a' — weight-side regulariser  λ_W ‖Π_⊥ W_L‖²_F  (5 λ × 2 fracs × 5 seeds = 50)
# ---------------------------------------------------------------------------
LAMW=(0 0.01 0.1 1.0 10.0)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMW[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks $KS --seed $s \
--lambdas $lam --lambdas-h 0" >> "$OUT"
    done
  done
done

# ---------------------------------------------------------------------------
# Exp 5b' — logit-side regulariser  λ_H ‖H‖²  (4 λ × 2 fracs × 5 seeds = 40)
#   Note: the original 80 logit-side runs failed due to a bug in
#   _training_step_with_reg (combine_logs unpacking). Fixed; rerun here.
# ---------------------------------------------------------------------------
LAMH=(0.001 0.01 0.1 1.0)
for s in "${SEEDS[@]}"; do
  for f in "${FRACS[@]}"; do
    for lam in "${LAMH[@]}"; do
      echo "canonical_geometry_experiments --experiment canonical_regularizer \
$COMMON --fracs $f --canonical-ks $KS --seed $s \
--lambdas 0 --lambdas-h $lam" >> "$OUT"
    done
  done
done

N=$(wc -l < "$OUT")
echo "Wrote $N tasks to $OUT"
echo
echo "Submit with:"
echo "  sbatch --array=1-${N}%32 \\"
echo "         --export=ALL,TASKS_FILE=$OUT \\"
echo "         grokking/experiments/slurm_paper_experiments.sh"
