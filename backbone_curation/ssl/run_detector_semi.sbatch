#!/bin/bash
#SBATCH --job-name=dino_semi
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_semi-%j.out
#
# Semi-supervised (Semi-DETR MVP) training of the 2-class ring/segment DINO detector:
# synthetic-supervised branch + EMA-teacher pseudo-labels on the 13k real unlabeled corpus.
# Design: docs/SEMI_DETR_INTEGRATION.md. RESUMABLE: main.py auto-picks-up checkpoint.pth
# from --output_dir, so resubmit the same command across 72h sessions.
# Per-epoch eval logs ap_total for 41 + organic (student: exp_ap_<name>.txt; EMA teacher
# during the semi phase: exp_ap_<name>_teacher.txt).
# --amp is required: the semi phase runs two student forwards per step (~2x activations).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_semi1
H5=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5
# optional full-detector warm start (shortens/skips burn-in; see config header):
#PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
export PYTHONPATH=$REPO

if [ ! -f "$H5" ]; then
    echo "ERROR: real unlabeled corpus not found at $H5"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_semi.py \
    --output_dir "$OUT_DIR" \
    --amp \
    ${PRETRAIN:+--pretrain_model_path "$PRETRAIN"}
