#!/bin/bash
#SBATCH --job-name=ringtgt
#SBATCH --partition=a100-galvani
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/ringtgt-%j.out
# CPU-only (no --gres). realbkg6 loads the hkl bank (~15 GB), hence 64G.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/ring_targets.py
