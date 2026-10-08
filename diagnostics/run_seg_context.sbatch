#!/bin/bash
#SBATCH --job-name=segctx
#SBATCH --partition=a100-galvani
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/segctx-%j.out
# CPU-only (no --gres): are run 7's low 41 segment scores caused by segments that sit on rings?
# Submit with -p bethge or -p v100-galvani when a100-galvani is backed up.
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
W=/mnt/lustre/work/schreiber/szb389/tmp_diag/ringfp7
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/seg_context_probe.py \
    lr4e5_1=$R/dino_lr4e5_1/checkpoint.pth \
    run6_best41=$R/dino_realbkg_conv6/checkpoint_best_41.pth \
    run7_best41=$W/run7_best41.pth \
    run7_latest=$W/run7_latest.pth
