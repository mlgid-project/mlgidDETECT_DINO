#!/bin/bash
#SBATCH --job-name=dino_edge
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_edge-%j.out
#
# EDGE-PEAK detector: identical to ssl1 (1 channel, SSL backbone, same lr/schedule); the only
# change is in the simulator -- peaks cut off by the dark wedge or a detector gap keep their
# FULL box instead of being clamped to the mask edge (wedge) or deleted outright (gaps).
# Measured 2026-09-06: 34.8% of 41's GT boxes and 20.1% of organic's overlap an invalid pixel,
# in 100% of images, while the simulator produced essentially none.
#
# Gate: organic/41 ap_total vs ssl1's 0.5683 / 0.7441 (reproduced on this branch).
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same
# command across 72h sessions. Per-epoch eval logs `ap_total` for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_edge1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, edge peaks ON)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_edge.py \
    --output_dir "$OUT_DIR"
