#!/bin/bash
#SBATCH --job-name=xsim
#SBATCH --partition=a100-galvani
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/xsim-%j.out
# CPU-only (no --gres). Loads the hkl bank for realbkg (~15 GB) plus a model.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/cross_sim_probe.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    conv3_final=$R/dino_realbkg_conv3/checkpoint.pth \
    run6_best41=$R/dino_realbkg_conv6/checkpoint_best_41.pth
