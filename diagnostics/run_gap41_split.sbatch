#!/bin/bash
#SBATCH --job-name=gap41
#SBATCH --partition=a100-galvani
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/gap41-%j.out
# CPU-only (no --gres): read-only forward passes, 49 frames per checkpoint.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export SWEEP_OUT=/mnt/lustre/work/schreiber/szb389/tmp_diag/gap41
mkdir -p $SWEEP_OUT
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/gap41_split.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    run6_best41=$R/dino_realbkg_conv6/checkpoint_best_41.pth \
    conv3_final=$R/dino_realbkg_conv3/checkpoint.pth \
    conv3_ep279=$R/dino_realbkg_conv3/checkpoint0279.pth \
    conv2_final=$R/dino_realbkg_conv2/checkpoint.pth \
    physics4_1=$R/dino_physics4_1/checkpoint.pth
