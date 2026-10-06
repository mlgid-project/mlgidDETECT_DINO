#!/bin/bash
#SBATCH --job-name=rsweep4
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/rsweep4-%j.out
# CPU-only (no --gres). realbkg6 loads the hkl bank (~15 GB), hence 64G.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export BASE=config/DINO/DINO_4scale_swin_realbkg_r7.py
export VARIANTS="$(cat /mnt/lustre/work/schreiber/szb389/tmp_diag/ring_fix/variants4.txt)"
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/ring_fix_sweep.py
