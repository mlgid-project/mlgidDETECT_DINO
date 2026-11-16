#!/bin/bash
#SBATCH --job-name=bankshard
#SBATCH --partition=cpu-galvani
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/shards/shard_%a-%A.out
#
# The hkl bank, built in six shards instead of one job.
#
# WHY SHARD. The single job could not get scheduled: it asked for one node for 25-30 h, and
# cpu-galvani kept its free cores reserved for a 32-core 3-day job, so backfill never admitted it
# -- 28 cores sat idle while our job waited behind a three-day reservation. Six short jobs fit the
# gaps. It also makes the build restartable: the previous attempt died after 8 h 56 m on a full
# disk quota and lost everything, where a shard failure now costs ~1.6 h.
#
# TIMING, from that failed run's own log: 46,500 CIFs in 6.96 h on 16 cores = 6,680 CIFs/h, so
# 60,474 CIFs is ~9 h and one sixth is ~1.6 h. The 3 h limit is slack, and small enough to
# backfill.
#
# Shards stride (i::6), not contiguous blocks: the CIFs are sorted by COD id and per-CIF cost
# grows with cell size, so contiguous blocks would finish at very different times.
#
# Merge with run_bank_hkl_merge.sbatch once all six are done.
set -euo pipefail

PY=/home/schreiber/szb389/.conda/envs/mlgid_physics/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic
mkdir -p "$LIB/shards"

cd "$REPO"
srun "$PY" physics_sim/generate_bank.py \
    --cif-dir "$LIB/cif" \
    --out     "$LIB/shards/shard_${SLURM_ARRAY_TASK_ID}.npz" \
    --shard   "${SLURM_ARRAY_TASK_ID}/6" \
    --orient-mode hkl \
    --k-orient 12 \
    --jobs 16 \
    --no-exclusions
