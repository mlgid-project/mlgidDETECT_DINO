#!/bin/bash
#SBATCH --job-name=newboxchk
#SBATCH --partition=a100-galvani
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/newboxchk-%j.out
# CPU-only (no --gres). Submit with -p bethge or -p v100-galvani when a100-galvani is backed up.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/newbox_check.py
