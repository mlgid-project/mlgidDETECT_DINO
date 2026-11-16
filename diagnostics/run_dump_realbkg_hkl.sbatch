#!/bin/bash
#SBATCH --job-name=dumphkl
#SBATCH --partition=v100-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/dumphkl_%j.out

# Ten RAW (pre-contrast) frames of the real-background simulator on the hkl bank, for review.
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
srun "$PY" -u diagnostics/dump_realbkg_pygid.py \
    --frames 10 \
    --seed 1 \
    --recipe diverse \
    --spots-cap 2,200 \
    --bank "$LIB/cif_library_organic/bank/bank_organic_hkl.npz" \
    --out  "$LIB/realbkgsim_hkl_raw_10.h5"
