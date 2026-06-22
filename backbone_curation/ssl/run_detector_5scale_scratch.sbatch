#!/bin/bash
#SBATCH --job-name=dino_5scratch
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_5scratch-%j.out
#
# FROM-SCRATCH 5-scale control for the DINO-5 experiment (RECIPE_v3 Idea 6). Random-init backbone
# (no backbone_dir), 5 feature levels. NO --amp -> matches the other runs. Compare vs dino_ssl5 to
# separate "SSL backbone benefit" from "extra scale benefit". RESUMABLE via main.py checkpoint.pth.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_5scale_scratch
export PYTHONPATH=$REPO

mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (random init, 5-scale)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_5scale_swin_scratch.py \
    --output_dir "$OUT_DIR"
