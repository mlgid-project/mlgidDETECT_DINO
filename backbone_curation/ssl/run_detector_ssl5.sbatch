#!/bin/bash
#SBATCH --job-name=dino_ssl5
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_ssl5-%j.out
#
# RECIPE_v3 Idea 6: DINO-5scale detector on the ROUND-1 SSL backbone (feeds the SSL-pretrained
# stride-4 level). NO --amp -> matches ssl1/ssl2/frozen exactly, so vs dino_ssl1 (4-scale) the
# only changed variable is num_feature_levels. Smoke test: ~25GB est at batch 2 no-amp (fits 40GB).
# Pair with dino_5scale_scratch (the from-scratch 5-scale control).
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir; resubmit across 72h sessions.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl5
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: round-1 SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (simmim1 backbone, 5-scale)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_5scale_swin_ssl.py \
    --output_dir "$OUT_DIR"
