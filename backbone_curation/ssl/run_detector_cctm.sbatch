#!/bin/bash
#SBATCH --job-name=dino_cctm2
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_cctm2-%j.out
#
# Exp B: Cross-DINO CCTM feature-enrichment block (docs/CROSS_DINO_INVESTIGATION.md S4b/S7).
# Fine-tunes the ssl1 checkpoint with the CCTM module inserted at the encoder->decoder
# boundary (identity at init -> warm-start safe). Synthetic-only training, standard
# pipeline otherwise. RESUMABLE (auto-resume from checkpoint.pth).
# Per-epoch eval logs ap_total for 41 + organic every 2 epochs.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_cctm2
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
export PYTHONPATH=$REPO

if [ ! -f "$PRETRAIN" ]; then
    echo "ERROR: warm-start checkpoint not found at $PRETRAIN"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (ssl1 warm start)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_cctm.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path "$PRETRAIN"
