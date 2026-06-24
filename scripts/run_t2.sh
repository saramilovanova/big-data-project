#!/bin/bash
#SBATCH --job-name=t2_quality
#SBATCH --partition=all
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t2_slurm_%j.out

source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate
python /d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t2_quality.py
