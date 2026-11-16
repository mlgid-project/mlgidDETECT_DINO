#!/bin/bash
#SBATCH --job-name=bclabels
#SBATCH --partition=2080-galvani
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/bclabels-%j.out
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/boxconv1_labels.py 150
