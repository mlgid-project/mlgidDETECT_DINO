#!/bin/bash
#SBATCH --job-name=segctx2
#SBATCH --partition=a100-galvani
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/segctx2-%j.out
# CPU-only. seg_context_probe.py (run_seg_context.sbatch) showed run 7 ep73 scores segments ON rings 0.36 vs
# 0.58 isolated on 41, and 0.45-0.53 vs 0.89 on its own sim. Why? Hypotheses, same run 7 model, same seeds:
#   texture 0     -> ring texture bumps mimic unlabelled segments (on-ring score rises if so)
#   keep 0.5      -> segment-on-ring prior (share ~ 41's 23%)
#   keep 0.5 tex0 -> both
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export SEGCTX_SETS="default|;texture0|realbkg_ring_chi_texture=0;keep0.5|realbkg_seg_on_ring_keep=0.5;keep0.5_tex0|realbkg_seg_on_ring_keep=0.5,realbkg_ring_chi_texture=0;keep0.3|realbkg_seg_on_ring_keep=0.3"
W=/mnt/lustre/work/schreiber/szb389/tmp_diag/ringfp7
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u diagnostics/seg_context_probe.py \
    run7_latest=$W/run7_latest.pth
