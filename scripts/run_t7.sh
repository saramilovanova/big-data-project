#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# T7: City-Wide Demand Forecasting
#
# This is the *scheduler/orchestrator* job. It runs the Python script, which
# in turn submits worker sub-jobs via SLURMCluster (dask_jobqueue).
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

python -u "${SCRIPT}" \
    --workers 2 4 8 \
    --cores-per-worker 4 \
    --mem-per-worker 16GB \
    --walltime 01:30:00

echo "============================================================"
echo "End     : $(date)"
echo "============================================================"
