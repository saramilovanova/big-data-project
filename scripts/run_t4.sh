#!/bin/bash
#SBATCH --job-name=t4_formats
#SBATCH --partition=all
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t4_slurm_%j.out

source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate
python -u /d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t4_formats.py
