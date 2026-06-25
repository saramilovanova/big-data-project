#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# T10: City-Wide Demand Forecasting - GPU
#
# GPU allocation strategy:
#   Request --gres=gpu:4 to target a 4-GPU node (gpu:4 line in sinfo).
#   If the scheduler assigns a 2-GPU node instead, the Python script
#   auto-detects the actual count via cp.cuda.runtime.getDeviceCount()
#   and adjusts the sweep accordingly (1->2 instead of 1->2->4).
#
# Submit: sbatch run_t10.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=t10_gpu_forecast
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=03:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/logs/t10_slurm_%j.out

set -euo pipefail

PROJECT="/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project"
VENV="/d/hpc/projects/FRI/bigdata/students/sm_bv/.venv"
SCRIPT="${PROJECT}/t10_demand_forecast.py"

mkdir -p "${PROJECT}/logs"

echo "============================================================"
echo "T10 GPU Demand Forecasting — Job $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Start   : $(date)"
echo "GPUs    : $(nvidia-smi --list-gpus | wc -l) × $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# CUDA toolkit must be loaded before activating the venv so that
# RAPIDS shared libraries (libcuml, libcudf, etc.) resolve correctly.
module load CUDA/12.6.0

source "${VENV}/bin/activate"
echo "Python  : $(which python) ($(python --version))"
echo "cuML    : $(python -c 'import cuml; print(cuml.__version__)')"
echo "XGBoost : $(python -c 'import xgboost; print(xgboost.__version__)')"
echo "CuPy    : $(python -c 'import cupy; print(cupy.__version__)')"

# Verify GPU is reachable from Python
python -c "
import cupy as cp
n = cp.cuda.runtime.getDeviceCount()
print(f'GPUs visible to CuPy: {n}')
for i in range(n):
    with cp.cuda.Device(i):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name  = props['name'].decode()
        total = props['totalGlobalMem'] / 1e9
        print(f'  GPU {i}: {name}  ({total:.0f} GB)')
"

echo "------------------------------------------------------------"
echo "Starting T10 experiment sweep..."
echo "------------------------------------------------------------"

# --gpus 1 2 4 : scalability sweep
#   The script auto-clips to available GPU count, so passing 4 is safe
#   even if only 2 GPUs are available on the allocated node.
python -u "${SCRIPT}" --gpus 1 2 4

echo "============================================================"
echo "End     : $(date)"
echo "============================================================"
