#!/bin/bash
#SBATCH --job-name=simmim2_giwaxs
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = the partition max
#SBATCH --cpus-per-task=12             # dataloader workers
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/slurm-simmim2-%j.out
#
# ROUND-2 SimMIM pretraining (ssl/RECIPE_v2.md): harder mask (0.70), drop_path 0.25,
# richer polar-legal augmentation (--aug v2). Writes to ssl_runs/simmim2 — simmim1 untouched.
# 150 epochs (round-1 converged by ~ep40-77; cosine fully anneals over 150) -> fits ONE 72h session.
# IDEMPOTENT + RESUMABLE: auto-continues from last.pth if present; just resubmit if interrupted.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim2
export PYTHONPATH=$REPO

mkdir -p "$OUT_DIR"
# auto-resume if a checkpoint from a previous (interrupted) session exists
RESUME=""
if [ -f "$OUT_DIR/last.pth" ]; then
    RESUME="--resume $OUT_DIR/last.pth"
    echo "[run] found $OUT_DIR/last.pth -> resuming"
else
    echo "[run] no checkpoint -> fresh start"
fi

cd "$REPO/backbone_curation/ssl"
srun $PY -u train_simmim.py \
    --epochs 150 \
    --batch 16 --accum 2 \
    --lr 1.5e-4 --warmup_epochs 10 --wd 0.05 \
    --mask_ratio 0.70 --mask_patch 32 --drop_path 0.25 \
    --aug v2 \
    --workers 10 --val_frac 0.12 \
    --out_dir "$OUT_DIR" $RESUME
