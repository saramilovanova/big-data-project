#!/bin/bash
#SBATCH --job-name=t1_repartition
#SBATCH --partition=all
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t1_slurm_%j.out

source /d/hpc/projects/FRI/bigdata/students/sm_bv/.venv/bin/activate
python /d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/t1_repartition.py
