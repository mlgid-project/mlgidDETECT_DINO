#!/bin/bash
#SBATCH --job-name=freshchk
#SBATCH --partition=a100-galvani
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/freshchk-%j.out
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/image_freshness.py --epochs 4 --per-epoch 40
