#!/bin/bash
#SBATCH --job-name=bankshard
#SBATCH --partition=v100-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/shards/shard_%x_%j.out
#
# One shard on FOUR cores. Submit with: sbatch --export=ALL,SHARD=2 ... (0-5)
#
# WHY v100-galvani AND ONLY FOUR CORES. cpu-galvani is two nodes, 64 cores total, against 236
# cores of pending demand, and our fairshare is spent (EffectvUsage 0.57, factor 0.18). sbatch
# --test-only put a 4-core job there at 2026-10-17 -- at the SAME timestamp it gave a 16-core
# one, which is how we know the gate is queue position, not job shape. Two earlier guesses
# (shorter walltime, then fewer cores) were both wrong for that reason; --test-only across
# partitions is what actually settled it. v100-galvani starts the same job in minutes.
#
# We take 4 of a node's 64 cores and request NO GPU, so the GPUs stay available to others: this
# is CPU work parked on spare CPU capacity, not a GPU allocation. Four cores is about 6 h per
# shard, against ~1.5 h on sixteen -- worth it when the wide job cannot start at all.
#
# Shard numbering is unchanged (i/6), so shards already built on 16 cores stay valid -- the split
# depends on the CIF stride, not on how many workers chew through it.
set -euo pipefail
: "${SHARD:?set SHARD=0..5}"

PY=/home/schreiber/szb389/.conda/envs/mlgid_physics/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic
mkdir -p "$LIB/shards"

cd "$REPO"
srun "$PY" physics_sim/generate_bank.py \
    --cif-dir "$LIB/cif" \
    --out     "$LIB/shards/shard_${SHARD}.npz" \
    --shard   "${SHARD}/6" \
    --orient-mode hkl \
    --k-orient 12 \
    --jobs 4 \
    --no-exclusions
