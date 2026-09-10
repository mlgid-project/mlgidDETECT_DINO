#!/bin/bash
#SBATCH --job-name=dino_lr1e4
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_lr1e4-%j.out
#
# Learning-rate experiment: lr = 1e-4, lr_backbone = 1e-5 (the upstream DINO ratio).
# The ONLY difference vs the ssl1 recipe is those two numbers -- see DINO_4scale_swin_lr1e4.py.
# 1-channel, SSL backbone init, legacy simulation, 2.80/1.30 boxes, same 500ep/lr_drop 280.
#
# Gate: organic/41 ap_total vs ssl1's 0.5683 / 0.7441; the bar to beat is dino_lr4e5_1's
# organic 0.6102 plateau (lr 4e-5). EARLY CHECK at epoch 6-10: ssl1 reached organic 0.203 by
# epoch 6, dino_lrsweep_1 (1.6e-4) never left 0.018. Flat by epoch 10 -> kill it.
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same
# command across 72h sessions. Per-epoch eval logs `ap_total` for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_lr1e4_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, legacy sim, batch 2, lr 1e-4 / lr_backbone 1e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_lr1e4.py \
    --output_dir "$OUT_DIR"
