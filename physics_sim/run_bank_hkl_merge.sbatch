#!/bin/bash
#SBATCH --job-name=bankmerge
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank_hkl_merge-%j.out
#
# Concatenate the six shards into one bank. Held back until every shard has succeeded, because a
# missing shard would silently produce a bank with a sixth of the structures absent -- --expect 6
# makes that fatal rather than quiet.
set -euo pipefail
PY=/home/schreiber/szb389/.conda/envs/mlgid_physics/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic
cd "$REPO"
srun "$PY" physics_sim/merge_bank_shards.py \
    --parts "$LIB/shards" \
    --out   "$LIB/bank/bank_organic_hkl.npz" \
    --expect 6
