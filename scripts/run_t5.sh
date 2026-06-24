#!/bin/bash
#SBATCH --job-name=t5_augment
#SBATCH --partition=all
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t5_slurm_%j.out

source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate
python -u /d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t5_augment.py
