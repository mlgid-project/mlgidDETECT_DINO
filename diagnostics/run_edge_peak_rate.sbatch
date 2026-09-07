#!/bin/bash
#SBATCH --job-name=edge_rate
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/edge_rate-%j.out
#
# Pre-flight for the edge-peak run: does SimulationConfig.edge_peaks actually produce cut-off
# peaks at the rate the real labels show (41 34.8% / organic 20.1% of boxes, 100% of images)?
# Prints the simulated rate with the flag off and on, so the change is measured, not assumed.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd "$REPO"
srun $PY -u diagnostics/edge_peak_rate.py 40
