#!/bin/bash
#SBATCH --job-name=dino_ssl2
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_ssl2-%j.out
#
# Round-2 detector: 2-class ring/segment DINO with the simmim2 (RECIPE_v2) SSL backbone.
# IDENTICAL setup to run_detector_ssl.sbatch except the backbone export + output dir + config,
# so dino_ssl1 vs dino_ssl2 isolates the SSL recipe (v1 vs v2).
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir; resubmit across 72h sessions.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl2
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim2/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: simmim2 SSL backbone weights not found at $BB"
    echo "  export it first:  export_backbone.py --ckpt <simmim2 best.pth> --out $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (simmim2 backbone init)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_ssl2.py \
    --output_dir "$OUT_DIR"
