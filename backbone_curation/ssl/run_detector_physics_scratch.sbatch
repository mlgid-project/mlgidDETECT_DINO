#!/bin/bash
#SBATCH --job-name=dino_physics_scratch
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn203
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_physics_scratch1-%j.out
#
# Track B, FROM-SCRATCH arm: ssl1's exact recipe (SSL backbone via backbone_dir, random detector
# head, uniform LR, 500 ep / drop 280, NO amp) with the 50% physics dilution present from ep0.
# Baseline = ssl1 (organic 0.586 / 41 0.762) = this recipe minus the dilution.
#
# WALLTIME: ~9.4 min/epoch measured -> 500 ep ~= 3.3 d, but the a100 partition caps at 3 d.
# This job is RESUMABLE (main.py auto-resumes from OUT_DIR/checkpoint.pth), so resubmit the
# SAME command once to finish. The decisive probe point (~ep300) lands inside the first session.
#
# Deliberately NO --amp: the 0.586 control was produced with amp=False, and matching it keeps
# the comparison single-variable.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics_scratch1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library/bank/bank.npz
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then echo "ERROR: SSL backbone weights not found at $BB"; exit 1; fi
if [ ! -f "$BANK" ]; then echo "ERROR: physics bank not found at $BANK"; exit 1; fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, detector head random)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics_scratch.py \
    --output_dir "$OUT_DIR"
