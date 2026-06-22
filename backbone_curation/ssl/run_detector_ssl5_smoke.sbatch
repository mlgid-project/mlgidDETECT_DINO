#!/bin/bash
#SBATCH --job-name=ssl5_smoke
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00                 # SHORT: smoke test only — meant to be cancelled after readout
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/ssl5_smoke-%j.out
#
# FEASIBILITY SMOKE TEST for DINO-5scale (RECIPE_v3 Idea 6), single A100 40GB + --amp.
# Goal: read real GPU memory ('max mem:') and throughput ('s / it') from the first ~20 iteration
# log lines, and confirm the model builds + loads (incl. the random-init norm0) without OOM.
# NOT a real run: throwaway output dir; cancel it (scancel <jobid>) once you've read the numbers.
#   - if 'max mem' is comfortably < 40GB at batch 2 -> launch the full run at batch 2.
#   - if it OOMs -> set batch_size=1 in DINO_5scale_swin_ssl.py and re-smoke.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl5_smoke
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: round-1 SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
echo "[smoke] DINO-5scale, --amp, batch from config. Watch 'max mem' + 's / it' then scancel."

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_5scale_swin_ssl.py \
    --output_dir "$OUT_DIR" \
    --amp
