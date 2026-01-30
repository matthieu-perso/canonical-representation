#!/bin/bash

echo ">>> Running on git commit: $(git rev-parse HEAD)"

# Default settings
LAUNCHER="basic"

# Name of the environment variables holding the repository base path
THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME="GROKKING_REPOSITORY_BASE_PATH"

# Check if the environment variable is set
if [[ -z "${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}" ]]; then
    echo "@@@ Error: Environment variable ${THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME} is not set."
    exit 1
fi

echo ">>> Loading environment variables from ${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}/.env ..."
source "${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}/.env"
if [[ $? -ne 0 ]]; then
    echo "@@@ Error: Failed to load environment variables from ${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}/.env"
    exit 1
fi
echo ">>> Environment variables loaded successfully."

# Change the working directory to the repository base path
cd "${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}" || {
    echo "@@@ Error: Failed to change directory to ${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}"
    exit 1
}
echo ">>> Changed directory to ${!THIS_REPOSITORY_BASE_PATH_ENV_VAR_NAME}"
echo ">>> Current working directory: $(pwd)"

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
    --launcher)
        LAUNCHER="$2"
        shift 2
        ;;
    --help | -h)
        echo "Usage: $0 [--launcher hpc|basic|basic_4gpu]"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage."
        exit 1
        ;;
    esac
done

# Use an array for launcher-specific args
LAUNCHER_ARGS=()
case "$LAUNCHER" in
basic)
    LAUNCHER_ARGS+=("hydra/launcher=basic")
    ;;
basic_4gpu)
    # Run N jobs in parallel, one per allocated GPU (job index % N).
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
    else
        NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    fi
    if [[ -z "$NGPUS" ]] || [[ "$NGPUS" -lt 1 ]]; then
        NGPUS=1
    fi
    echo ">>> Detected $NGPUS GPU(s), running up to $NGPUS jobs in parallel."
    LAUNCHER_ARGS+=(
        "hydra/launcher=joblib"
        "hydra.launcher.n_jobs=$NGPUS"
    )
    ;;
hpc)
    LAUNCHER_ARGS+=(
        "hydra/sweeper=basic"
        "hydra/launcher=hpc_submission"
        "hydra.launcher.queue=CUDA"
        # "hydra.launcher.template=GTX1080"
        "hydra.launcher.template=RTX6000"
        "hydra.launcher.memory=32"
        "hydra.launcher.ncpus=2"
        "hydra.launcher.ngpus=1"
        "hydra.launcher.walltime=59:59:59"
    )
    ;;
*)
    echo "Unknown launcher: $LAUNCHER. Valid options: hpc, basic, basic_4gpu"
    exit 1
    ;;
esac

echo ">>> Using launcher: $LAUNCHER"
echo ">>> Launcher arguments: ${LAUNCHER_ARGS[@]}"

# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Notes:
# - Important: SCRIPT_PATH is relative to the repository base path, otherwise the hpc submission script will not work.
SCRIPT_PATH="grokking/scripts/train_grokk.py"
echo ">>> Running script: $SCRIPT_PATH"

uv run python3 "$SCRIPT_PATH" \
    --multirun \
    hydra/sweeper=basic \
    "${LAUNCHER_ARGS[@]}" \
    dataset=mod_sum_dataset \
    dataset.p=197 \
    dataset.dataset_seed=46,47,48,49,50 \
    dataset.frac_train=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
    train.optimizer.weight_decay=0.01 \
    train.optimizer.eps=1e-6 \
    train.optimizer.clip_grad_norm_max_norm=1.0 \
    train.lr_scheduler.lr_scheduler_type=linear \
    train.max_steps=60000 \
    train.save_checkpoints_every=500 \
    logging.training.create_plot_of_model_output_parameters_every=10000 \
    topological_analysis.compute_estimates_every=500 \
    topological_analysis.absolute_n_neighbors_choices=[64] \
    topological_analysis.create_projection_plot_every=20000 \
    wandb.use_wandb=true \
    wandb.wandb_project=grokking_replica_HPC_cluster_runs_different_dataset_portions_long_large_p

echo ">>> Finished running the script."
echo ">>> Exiting with status code 0."
exit 0
