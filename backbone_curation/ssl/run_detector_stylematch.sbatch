#!/bin/bash
#SBATCH --job-name=dino_stylematch
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_stylematch-%j.out
#
# Style-transfer input matching Phase-0 screen (docs/STYLE_TRANSFER_INVESTIGATION.md).
# Fine-tunes the ssl1 checkpoint with synthetic training images CDF-matched to the real
# corpus intensity distribution. Training-only transform; standard pipeline otherwise.
# RESUMABLE (auto-resume checkpoint.pth). Per-epoch eval logs 41 + organic.
# READ-OUT: gate on the faint/high-q recall probe (diag_compare.py), not AP.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_stylematch1
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
STYLE_REF=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/style_ref.pt
export PYTHONPATH=$REPO

if [ ! -f "$PRETRAIN" ]; then echo "ERROR: warm-start ckpt not found at $PRETRAIN"; exit 1; fi
if [ ! -f "$STYLE_REF" ]; then echo "ERROR: style reference not found at $STYLE_REF (run datasets/style_match.py)"; exit 1; fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (ssl1 warm start)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_stylematch.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path "$PRETRAIN"
