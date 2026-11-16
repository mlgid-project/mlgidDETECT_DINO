#!/bin/bash
#SBATCH --job-name=ringfp7
#SBATCH --partition=a100-galvani
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/ringfp7-%j.out
# CPU-only (no --gres): did run 7's ring texture stop the ring fragmentation on 41?
# run7_best41.pth + config_args_all.json are COPIES from dino_realbkg_conv7 (load_model reads the json next to the ckpt; training keeps rewriting the originals).
# Reference = lr4e5_1 (legacy, ~1.7 boxes/ring on 41). conv3 was 4.7 boxes/ring, ring TP score 0.30.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export RINGFP_OUT=/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/14_run7_ring_fragments
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/ring_fp_probe.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    conv3_final=$R/dino_realbkg_conv3/checkpoint.pth \
    run6_best41=$R/dino_realbkg_conv6/checkpoint_best_41.pth \
    run7_best41=/mnt/lustre/work/schreiber/szb389/tmp_diag/ringfp7/run7_best41.pth \
    run7_latest=/mnt/lustre/work/schreiber/szb389/tmp_diag/ringfp7/run7_latest.pth
