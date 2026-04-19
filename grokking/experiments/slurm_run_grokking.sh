#!/bin/bash
#SBATCH -A BURDEN-MECHINT-SL2-GPU
#SBATCH --job-name=grokking_multirun
#SBATCH --output=logs/grokking_%j.out
#SBATCH --error=logs/grokking_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm2833@cam.ac.uk

mkdir -p logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
# This project requires Python 3.12. Create venv once: module load python/3.12; python3 -m venv /rds/user/mm2833/hpc-work/venv_grokking; source .../venv_grokking/bin/activate; pip install -e .
module load python/3.12 cuda/11.8 cudnn/8.9_cuda-11.8
source /rds/user/mm2833/hpc-work/venv_grokking/bin/activate

# Repo path on the cluster – change if you clone elsewhere
export GROKKING_REPOSITORY_BASE_PATH="${GROKKING_REPOSITORY_BASE_PATH:-/rds/user/mm2833/hpc-work/grokking-via-lid}"
cd "$GROKKING_REPOSITORY_BASE_PATH" || { echo "Failed to cd to repo"; exit 1; }
[ -f .env ] && source .env

# Use your venv Python (no uv)
export GROKKING_PYTHON_CMD="python3"

echo "Python: $(python3 --version)"
echo "Repo: $GROKKING_REPOSITORY_BASE_PATH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

./grokking/experiments/run_with_multiple_dataset.frac_train.sh --launcher basic_4gpu
