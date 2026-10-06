#!/bin/bash
#SBATCH --job-name=r7verify
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/r7verify-%j.out
# CPU-only (no --gres). r7 ring fix: shape/crowding vs 41, review figures, raw h5, cross-sim check.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
echo "##### SHAPE + CROWDING (r7 vs real 41)"
$PY -u diagnostics/peak_shape_compare.py --only real41,realbkg6 --config config/DINO/DINO_4scale_swin_realbkg_r7.py --frames 150
echo "##### REVIEW FIGURES"
$PY -u diagnostics/ring_fix_review.py
echo "##### RAW H5"
$PY -u diagnostics/dump_realbkg_pygid.py --config config/DINO/DINO_4scale_swin_realbkg_r7.py --frames 20 --seed 3 \
    --out /mnt/lustre/work/schreiber/szb389/datasets/realbkgsim_r7_ringfix_raw_20.h5
echo "##### CROSS-SIM (do r7 rings read as legacy/41 rings?)"
XSIM_CONFIG=config/DINO/DINO_4scale_swin_realbkg_r7.py $PY -u diagnostics/cross_sim_probe.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    conv3_final=$R/dino_realbkg_conv3/checkpoint.pth
