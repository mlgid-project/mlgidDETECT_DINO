#!/bin/bash
#SBATCH --job-name=dino_codino_scratch
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max; RESUMABLE, resubmit across sessions
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_codino_scratch-%j.out
#
# Co-DINO FROM-SCRATCH co-train (the faithful test). Mirrors run_detector_ssl.sbatch (the
# recipe that produced ssl1) EXACTLY -- SSL backbone init via backbone_dir, no --pretrain,
# no --amp, uniform LR -- but with the collaborative aux head on from epoch 0. So the only
# difference vs the ssl1 baseline (organic 0.586) is the aux head => a clean A/B.
# Per-epoch eval logs 41 + organic (eval_interval=2). RESUMABLE (auto-resume checkpoint.pth).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_codino_scratch1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init + co-heads from ep0)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_codino_scratch.py \
    --output_dir "$OUT_DIR"
