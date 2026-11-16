#!/bin/bash
#SBATCH --job-name=nbimages
#SBATCH --partition=a100-galvani
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/nbimages-%j.out
# CPU-only (no --gres)
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/newbox_sim_images.py
