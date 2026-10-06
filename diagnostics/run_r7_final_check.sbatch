#!/bin/bash
#SBATCH --job-name=r7final
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/r7final-%j.out
# CPU-only (no --gres). Final r7: preflight through main.py, stats vs 41, review figures, raw h5.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
echo "##### PREFLIGHT"
$PY -u diagnostics/r7_preflight.py
echo "##### STATS vs 41"
BASE=config/DINO/DINO_4scale_swin_realbkg_r7.py VARIANTS="$(cat /mnt/lustre/work/schreiber/szb389/tmp_diag/ring_fix/variants_final.txt)" \
    $PY -u diagnostics/ring_fix_sweep.py
echo "##### REVIEW FIGURES"
$PY -u diagnostics/ring_fix_review.py
echo "##### RAW H5"
$PY -u diagnostics/dump_realbkg_pygid.py --config config/DINO/DINO_4scale_swin_realbkg_r7.py --frames 20 --seed 3 \
    --out /mnt/lustre/work/schreiber/szb389/datasets/realbkgsim_r7_ringfix_raw_20.h5
