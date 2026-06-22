#!/bin/bash
#SBATCH --job-name=simmim_giwaxs
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = the partition max
#SBATCH --cpus-per-task=12             # dataloader workers
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/slurm-%j.out
#
# SimMIM pretraining of the swin-L 48x6 backbone on backbone_ssl_corpus.h5.
# IDEMPOTENT + RESUMABLE: this same script starts a fresh run if none exists, and
# auto-continues from last.pth if a prior (e.g. 72h-interrupted) run is present.
# To run across multiple 72h sessions: just `sbatch run_simmim.sbatch` again each time.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1
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
    --epochs 200 \
    --batch 16 --accum 2 \
    --lr 1.5e-4 --warmup_epochs 10 --wd 0.05 \
    --mask_ratio 0.6 --mask_patch 32 --drop_path 0.2 \
    --workers 10 --val_frac 0.12 \
    --out_dir "$OUT_DIR" $RESUME

# --- optional auto-chain: uncomment to auto-resubmit until training prints DONE ---
# if ! grep -q '^199,' "$OUT_DIR/log.csv" 2>/dev/null; then
#     echo "[run] not finished -> resubmitting"; sbatch "$0"
# fi
