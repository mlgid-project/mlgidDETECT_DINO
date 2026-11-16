#!/bin/bash
#SBATCH --job-name=convh5
#SBATCH --partition=v100-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/convh5_%j.out

# The 25 review frames as a pyGID/NeXus h5, RAW (pre-contrast), under the agreed labelling
# convention: render iff labelled, 2.0 x local noise + SNR 6, IoU 0.30 segments / 0.10 rings,
# at most 200 reflections drawn. Same seed and recipe table as run_box_convention.sbatch, so
# the frames correspond to the PNGs in images/08_box_convention.
set -euo pipefail
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets
cd "$REPO"
srun "$PY" -u diagnostics/dump_realbkg_pygid.py \
    --frames 25 \
    --seed 7 \
    --recipe convention \
    --unified-labels \
    --contrast-min 2.0 \
    --snr-min 6.0 \
    --seg-iou-max 0.30 \
    --ring-iou-max 0.10 \
    --max-peaks 200 \
    --bank "$LIB/cif_library_organic/bank/bank_organic_hkl.npz" \
    --out  "$LIB/realbkgsim_convention_25.h5"
