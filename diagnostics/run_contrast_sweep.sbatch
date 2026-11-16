#!/bin/bash
#SBATCH --job-name=contrast_sweep
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/contrast_sweep-%j.out
#
# Contrast sweep on the labeled 41 + organic sets with a FIXED checkpoint (ssl1 by default):
# each contrast pipeline (clipping percentiles / log / HE / gamma / CLAHE) is scored with the
# same eval + postprocessing main.evaluate_giwaxs_ap uses, and ap_total / recall / precision
# are appended to a CSV. Purpose: pick the input contrast for single- and multi-channel
# training from measured numbers instead of the inherited default.
#
# The FIRST row of every grid is the deployed default (clip 5/99.5 + log + HE); it must
# reproduce ssl1's logged 41 = 0.745 / organic = 0.561, which is the sanity check on the run.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/sweeps
export PYTHONPATH=$REPO
mkdir -p "$OUT"
cd "$REPO"

# smoke: 1 setting x 3 frames per dataset, catches a broken model/postprocessing path in ~2 min
echo "=== SMOKE ==="
srun $PY -u diagnostics/sweep_contrast.py --grid base --limit 3 --out "$OUT/contrast_smoke.csv"

echo "=== FULL SWEEP ==="
srun $PY -u diagnostics/sweep_contrast.py --grid all --out "$OUT/contrast_ssl1.csv"
