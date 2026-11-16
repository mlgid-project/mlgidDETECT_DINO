#!/bin/bash
#SBATCH --job-name=mcc_smoke
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/mcc_smoke-%j.out
#
# Pre-flight for the multi-contrast run: [1] simulator emits 4 genuinely different channels,
# [2] mc_smoke's four checks on the mcc config (SSL load, sample shape, fwd+bwd, and step-0
# equivalence: with ch1-3 weights zero the net must start EXACTLY as the 1-channel model).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd "$REPO"

echo "=== [1] simulator side ==="
srun $PY -u diagnostics/mcc_smoke.py config/DINO/DINO_4scale_swin_mcc.py
echo "=== [2] model side ==="
srun $PY -u diagnostics/mc_smoke.py config/DINO/DINO_4scale_swin_mcc.py
