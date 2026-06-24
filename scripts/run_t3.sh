#!/bin/bash
#SBATCH --job-name=t3_aggregations
#SBATCH --partition=all
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t3_slurm_%j.out

source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate
python -u /d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t3_aggregations.py
