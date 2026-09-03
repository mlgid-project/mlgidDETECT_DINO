#!/bin/bash
#SBATCH --job-name=dino_mc
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_mc-%j.out
#
# Train the 2-class ring/segment DINO detector with 4-channel input [HE, B1, B2, mask]
# (util/channels.py build_channels), SSL backbone init with zero-padded patch embed ->
# step 0 == ssl1 exactly (verified by diagnostics/mc_smoke.py).
# Gate: organic/41 ap_total vs ssl1's 0.5683 / 0.7441 (reproduced on this branch).
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same
# command across 72h sessions. Per-epoch eval logs `ap_total` for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_mc1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 4-channel)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_mc.py \
    --output_dir "$OUT_DIR"
