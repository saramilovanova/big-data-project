#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# T7: City-Wide Demand Forecasting — Arnes HPC SLURM submission script
#
# This is the *scheduler/orchestrator* job. It runs the Python script, which
# in turn submits worker sub-jobs via SLURMCluster (dask_jobqueue).
# Workers are allocated on demand as the scalability sweep progresses:
#   run 1 →  2 worker sub-jobs × 4 cores × 16 GB
#   run 2 →  4 worker sub-jobs × 4 cores × 16 GB
#   run 3 →  8 worker sub-jobs × 4 cores × 16 GB
#
# Total HPC budget (worst case, all 3 runs alive simultaneously):
#   scheduler : 1 node × 4 CPU × 32 GB
#   workers   : 8 nodes × 4 CPU × 16 GB  (only 8 live at once)
#
# Submit: sbatch run_t7.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=t7_demand_forecast
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t7_slurm_%j.out

set -euo pipefail

VENV="/d/hpc/projects/FRI/bigdata/students/sm_bv/.venv"
SCRIPT="/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t7_demand_forecast.py"

echo "============================================================"
echo "T7 Demand Forecasting — Job $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Start   : $(date)"
echo "============================================================"

# Activate virtual environment
source "${VENV}/bin/activate"
echo "Python  : $(which python) ($(python --version))"

# Verify key packages
# python -c "import dask_ml, xgboost, dask_jobqueue; print('Packages OK')"

# Run the experiment:
#   --workers 2 4 8   → scalability sweep (required by T7 spec)
#   --cores-per-worker 4   → 4 CPU per Dask worker SLURM sub-job
#   --mem-per-worker 16GB  → each worker gets 16 GB RAM
#   --walltime 01:30:00    → worker sub-job walltime
#
# The scheduler job (this script) lives for the full 6h to coordinate all runs.
# Each worker sub-job lives for at most 1.5h per scalability step.

python -u "${SCRIPT}" \
    --workers 2 4 8 \
    --cores-per-worker 4 \
    --mem-per-worker 16GB \
    --walltime 01:30:00

echo "============================================================"
echo "End     : $(date)"
echo "============================================================"
