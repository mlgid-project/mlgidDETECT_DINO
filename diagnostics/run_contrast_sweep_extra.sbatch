#!/bin/bash
#SBATCH --job-name=contrast_extra
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/contrast_extra-%j.out
#
# Runs the EXTRA_CLIPS block (25/90, 5/50, 75/99.9 x log x HE) into its OWN csv -- the main
# sweep (job 2853250) has contrast_ssl1.csv open and two nodes appending to one Lustre file
# can interleave a line. Concatenate the two csvs afterwards. The deployed default is re-run
# as row 1 here too, so the two jobs can be cross-checked before the tables are merged.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/sweeps
export PYTHONPATH=$REPO
cd "$REPO"

srun $PY -u diagnostics/sweep_contrast.py --grid clips_extra --out "$OUT/contrast_ssl1_extraclips.csv"
