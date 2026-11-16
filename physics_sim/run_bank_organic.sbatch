#!/bin/bash
#SBATCH --job-name=bank_org
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank_full-%j.out
#
# Build the full organic peak bank for the pygidSIM track (MODIFICATIONS.md section I).
#
# Library: 60,474 COD structures selected by physics_sim/fetch_cod_organics.py -- must contain
# C+H, only non-metals plus at most one OSC metal centre (the eval family includes CuPc/ZnPc),
# cell volume 500-8000 A^3, deduped on rounded cell. Median cell 2374 A^3.
#
# Per CIF: 1 powder entry + 8 recorded random fiber orientations, top-200 peaks each. Expect
# roughly 60k * 8.7 ~ 520k entries. Covers |q| <= hypot(3.5,3.5) = 4.95, the organic detector's
# q_max. 60k CIFs on 32 workers lands around 6-8 h; the 12 h limit is slack, not an estimate.
#
# CPU only -- pygidsim needs xrayutilities==1.7.10, which lives in mlgid_physics, NOT in the
# training env. Training reads the resulting npz and gains no new dependency.
#
# NOTE: --no-exclusions. The eval-exclusion pass (physics_sim/build_exclusion_list.py on branch
# development, via mlgidMATCH) has NOT been ported yet. This bank is therefore not eval-clean:
# a COD structure matching an eval material can still contribute. Must be rebuilt with
# exclusions before any number from a run on it is quoted as a clean result.
set -euo pipefail

PY=/home/schreiber/szb389/.conda/envs/mlgid_physics/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic

cd "$REPO"
srun "$PY" physics_sim/generate_bank.py \
    --cif-dir "$LIB/cif" \
    --out     "$LIB/bank/bank_organic.npz" \
    --k-orient 8 \
    --jobs 32 \
    --no-exclusions
