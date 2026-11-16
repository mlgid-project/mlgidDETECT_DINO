#!/bin/bash
#SBATCH --job-name=boxconv
#SBATCH --partition=v100-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/boxconv_%j.out

# 25 frames under the candidate LABELLING CONVENTION, as PNGs, for visual review.
#
# WHY A JOB AND NOT THE LOGIN NODE: the hkl bank needs ~12.5 GB peak -- 5.4 GB to slurp the npz
# plus ~7 GB of float32 q/chi/intensity tensors -- and the login node's per-user cgroup kills
# anything over ~6.5 GB. `free` reports the machine, not the slice, so it looks like there is
# room when there is not.
set -euo pipefail
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets
cd "$REPO"
srun "$PY" -u diagnostics/dump_label_convention.py \
    --frames 25 \
    --seed 7 \
    --bank "$LIB/cif_library_organic/bank/bank_organic_hkl.npz" \
    --contrast-min 2.0 \
    --snr-min 6.0 \
    --seg-iou-max 0.30 \
    --ring-iou-max 0.10 \
    --max-peaks 200 \
    --cmap magma \
    --layout quad \
    --out-dir /mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/08_box_convention
