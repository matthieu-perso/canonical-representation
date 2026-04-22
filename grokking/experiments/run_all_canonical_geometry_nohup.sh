#!/usr/bin/env bash
# Run all canonical-geometry experiments under nohup, one after another (single GPU friendly).
#
# Usage:
#   chmod +x grokking/experiments/run_all_canonical_geometry_nohup.sh
#   cd /path/to/grokking-via-lid
#   WANDB_ENTITY=your_entity ./grokking/experiments/run_all_canonical_geometry_nohup.sh
#
# Optional env:
#   WANDB_PROJECT   (default: canonical_repr_grokking)
#   WANDB_MODE      (default: unset → online; use "offline" to sync later)
#   CUDA_VISIBLE_DEVICES  (default: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-canonical_representation_paper}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/canonical_geometry_runs}"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
RUNNER=(uv run python grokking/scripts/canonical_geometry_experiments.py)
COMMON=(--wandb-project "$WANDB_PROJECT")
if [[ -n "${WANDB_MODE:-}" ]]; then
  COMMON+=(--wandb-mode "$WANDB_MODE")
fi

echo "Logging under $LOG_DIR (timestamp $TS)"
echo "W&B project: $WANDB_PROJECT"

run_one() {
  local name="$1"
  shift
  echo "=== START $name ==="
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${RUNNER[@]}" "${COMMON[@]}" "$@" \
    >"$LOG_DIR/${TS}_${name}.log" 2>&1
  echo "=== DONE $name (log: $LOG_DIR/${TS}_${name}.log) ==="
}

# A — single long run: geometry, margins, projection
run_one geometry_phase_transition --experiment geometry_phase_transition

# E — data fraction sweep (tune --fracs if needed)
run_one data_threshold --experiment data_threshold \
  --fracs "0.1,0.12,0.14,0.15,0.18,0.2,0.22,0.23,0.24,0.25,0.3,0.35,0.4"

# F — weight decay sweep (standard L2 on parameters)
run_one weight_decay_geometry --experiment weight_decay_geometry \
  --weight-decays "0,0.0001,0.001,0.01,0.1,1.0"

# D — noise on logits after grokking
run_one noise_robustness --experiment noise_robustness \
  --sigmas "0,0.5,1,2,5,10,20"

# G — canonical regulariser λ‖Π_{V^⊥}W‖² (WD=0 inside script); includes high λ
run_one canonical_regularizer --experiment canonical_regularizer \
  --lambdas "0,0.001,0.01,0.1,1.0,10.0,30.0,100.0" \
  --fracs "0.15,0.3" \
  --canonical-ks "1"

echo "All jobs finished sequentially. Logs: $LOG_DIR/${TS}_*.log"
