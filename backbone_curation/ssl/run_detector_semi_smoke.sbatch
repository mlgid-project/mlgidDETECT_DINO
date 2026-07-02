#!/bin/bash
#SBATCH --job-name=semi_smoke
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/semi_smoke-%j.out
#
# SMOKE TEST for the semi-supervised path (docs/SEMI_DETR_INTEGRATION.md): warm-start from
# the ssl1 checkpoint so the teacher emits real pseudo-boxes, force the semi phase from
# epoch 0, run ~15 steps (--debug) + one labeled eval (student + teacher), then exit.
# Success = train steps log loss_unsup/pseudo_seg/pseudo_ring and both eval APs print.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_semi_smoke
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
export PYTHONPATH=$REPO

rm -rf "$OUT_DIR"            # smoke runs are throwaway; never resume one
mkdir -p "$OUT_DIR"
cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_semi.py \
    --output_dir "$OUT_DIR" \
    --amp --debug \
    --pretrain_model_path "$PRETRAIN" \
    --options epochs=1 semi_start_epoch=0 ema_epoch=0 unsup_warmup_epochs=1
echo "[smoke] OK"
