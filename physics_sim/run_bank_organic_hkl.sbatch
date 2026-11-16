#!/bin/bash
#SBATCH --job-name=bank_hkl
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=30:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank_hkl-%j.out
#
# The organic peak bank again, but with FIXED LOW-INDEX hkl fibre axes instead of random ones.
#
# WHY. The existing bank draws 8 random unit vectors per CIF, uniform on the sphere. Those are
# generic directions: a real thin film grows on a low-index contact plane, so real texture
# concentrates on a handful of hkl per material and the random bank hits a realistic texture only
# by accident. Fixing the axes to low-index plane normals makes the training distribution look
# like what the films actually do, and -- the reason it was asked for -- it gives every entry a
# NAMED orientation, which is what a future structure-identification task would need. The random
# bank's [0.18882, -0.19839, 0.96176] names nothing.
#
# WHAT: --orient-mode hkl --k-orient 12, i.e. the first 12 of
#   (001) (010) (100) | (110) (1-10) (101) (10-1) (011) (01-1) | (111) (1-11) (11-1)
# The natural low-index list is 13 long, because the {111} family has four members distinct up to
# sign; k_orient=12 drops the last, (-111). An integer triple is read as hkl -- a PLANE NORMAL --
# because the vector is mapped through the reciprocal basis, which is also why the 15 deg
# separation test is kept: the same hkl list gives different physical angles in different cells,
# and in some of them two low-index normals nearly coincide. Measured on 39 CIFs: 37 keep all 12,
# one keeps 9, one keeps 7, mean 11.79. So entries per CIF vary, unlike the random path's flat 8.
#
# ALSO: TOP_PEAKS is now 2000, i.e. effectively no cap -- the old 200 truncated 98% of entries and
# was the binding limit on how much intensity range a frame could show (2.14 decades at the median
# against 5.89 for the full list, and 2.7-3.6 for real labelled peaks). Which of those reflections
# a frame actually uses is decided at draw time, not here.
#
# Expect roughly 60k * (1 powder + ~11.8 oriented) ~ 775k entries against the random bank's
# 526,131, each holding ~700 reflections against ~187, so ~540M peaks and ~3.5 GB. Runtime: the
# random build took 6-8 h at 9 giwaxs_sim calls per CIF, this one makes 13, so 9-12 h of
# simulation; the wider store adds concatenate and compression time at the end, call it 10-14 h
# overall AT 32 CORES. This runs on 16, to backfill into a half-empty node rather than wait for a
# whole one: cpu-galvani had 17 cores idle against a 3-day queue for 32, so ~20-26 h starting now
# beats 10-14 h starting in 63 h. The 30 h limit is slack, not an estimate.
#
# This does NOT replace bank_organic.npz. Both stay on disk so the two can be A/B'd with
# everything else held fixed.
#
# NOTE: --no-exclusions, matching the random bank exactly so the comparison is apples to apples.
# Neither bank is eval-clean; that pass is still unported.
set -euo pipefail

PY=/home/schreiber/szb389/.conda/envs/mlgid_physics/bin/python
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
LIB=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic

cd "$REPO"
srun "$PY" physics_sim/generate_bank.py \
    --cif-dir "$LIB/cif" \
    --out     "$LIB/bank/bank_organic_hkl.npz" \
    --orient-mode hkl \
    --k-orient 12 \
    --jobs 16 \
    --no-exclusions
