#!/bin/bash
#SBATCH --job-name=edge_rate
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/edge_rate2080-%j.out
#
# Pre-flight for the edge-peak run: does SimulationConfig.edge_peaks actually produce cut-off
# peaks at the rate the real labels show (41 34.8% / organic 20.1% of boxes, 100% of images)?
# Prints the simulated rate with the flag off and on, so the change is measured, not assumed.
#
# Runs on 2080-galvani: the simulator is plain torch on CUDA (no custom kernels -- MSDeformAttn
# is model-side and no model is built here), so a 2080 Ti is enough and that partition is idle.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd "$REPO"
srun $PY -u diagnostics/edge_peak_rate.py 40
