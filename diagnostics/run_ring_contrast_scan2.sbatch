#!/bin/bash
#SBATCH --job-name=rcscan2
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/rcscan2-%j.out
# CPU-only (no --gres). Loads the hkl bank for realbkg (~15 GB) plus a model.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
export SCAN_SCALES=0.03,0.014,0.006,0.003,0.0015
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/ring_contrast_scan.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    conv3_final=$R/dino_realbkg_conv3/checkpoint.pth
