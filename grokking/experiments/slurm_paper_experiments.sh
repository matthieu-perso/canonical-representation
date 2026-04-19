#!/bin/bash
# SLURM array launcher for the canonical-geometry paper experiments.
#
# Run order:
#   1. ./grokking/experiments/build_paper_tasks.sh
#      → writes ./tasks_paper_experiments.txt (one CLI per line)
#   2. sbatch --array=1-$(wc -l < tasks_paper_experiments.txt)%32 \
#             grokking/experiments/slurm_paper_experiments.sh
#
# Each array task:
#   - claims 1 GPU + 8 CPUs + 32 GB
#   - reads the line at $SLURM_ARRAY_TASK_ID from tasks_paper_experiments.txt
#   - runs that single experiment configuration
#
# Resource sizing rationale: model is ~1M params, ~600 MB GPU mem at bsize=512.
# 30k steps complete in ~25 min on an A100. 1 h walltime is 2× safety margin.
#
#SBATCH -A BURDEN-MECHINT-SL2-GPU
#SBATCH --job-name=canon_geom
#SBATCH --output=logs/slurm/canon_geom_%A_%a.out
#SBATCH --error=logs/slurm/canon_geom_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mm2833@cam.ac.uk

set -euo pipefail
mkdir -p logs/slurm

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.8 cudnn/8.9_cuda-11.8
# NOTE: CSD3 does not provide a python/3.12 module. Python 3.12 is provided
# by uv and lives in the project-local virtual environment at $REPO/.venv
# (created once via `uv sync`).

export GROKKING_REPOSITORY_BASE_PATH="${GROKKING_REPOSITORY_BASE_PATH:-/rds/user/mm2833/hpc-work/canonical-representation}"
cd "$GROKKING_REPOSITORY_BASE_PATH"

source "$GROKKING_REPOSITORY_BASE_PATH/.venv/bin/activate"
[ -f .env ] && source .env

# Avoid wandb service-wait hangs under heavy array load
export WANDB__SERVICE_WAIT=300
# Each task gets its own wandb cache dir to avoid collision
export WANDB_DIR="${GROKKING_REPOSITORY_BASE_PATH}/wandb_dir/array_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$WANDB_DIR"

TASKS_FILE="${TASKS_FILE:-tasks_paper_experiments.txt}"
if [ ! -f "$TASKS_FILE" ]; then
  echo "ERROR: $TASKS_FILE not found. Run grokking/experiments/build_paper_tasks.sh first." >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set; submit with sbatch --array=...}"
CMD=$(sed -n "${TASK_ID}p" "$TASKS_FILE")
if [ -z "$CMD" ]; then
  echo "ERROR: no task at line $TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

echo "================================================================"
echo "[task $TASK_ID/$SLURM_ARRAY_TASK_COUNT] $(date)"
echo "host=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "cmd: $CMD"
echo "================================================================"

# Run via python (matches existing slurm_run_grokking.sh convention; uv not used on cluster)
python -m grokking.scripts.${CMD}
EXIT=$?
echo "[task $TASK_ID] exit=$EXIT  $(date)"
exit $EXIT
