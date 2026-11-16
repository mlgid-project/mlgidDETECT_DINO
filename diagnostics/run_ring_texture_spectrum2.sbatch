#!/bin/bash
#SBATCH --job-name=spectrum2
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/spectrum2-%j.out
# CPU-only (no --gres). realbkg6 loads the hkl bank (~15 GB), hence 64G.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export SP_CANDS="[(1.0, (1, 64)), (1.5, (1, 64)), (2.0, (1, 64)), (3.0, (1, 64))]"
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/ring_texture_spectrum.py
