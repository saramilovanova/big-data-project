#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# T8: FHVHV Emergence Analysis
#
# Submit: sbatch run_t8.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=t8_fhvhv_emergence
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t8_slurm_%j.out

set -euo pipefail

VENV="/d/hpc/projects/FRI/bigdata/students/sm_bv/.venv"
SCRIPT="/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t8_analysis.py"

echo "============================================================"
echo "T8 FHVHV Emergence — Job $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "============================================================"

source "${VENV}/bin/activate"
echo "Python: $(which python) ($(python --version))"

python -u "${SCRIPT}"

echo "============================================================"
echo "End   : $(date)"
echo "============================================================"
