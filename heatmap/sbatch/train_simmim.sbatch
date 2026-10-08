#!/bin/bash
#SBATCH --job-name=hm_simmim
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_simmim-%j.out
# Experiment 1: frozen SimMIM backbone + heatmap head, same sim as ssl1. Resumable (resubmit the same command).
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd $REPO
$PY -u heatmap/train.py --out /mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/heatmap_runs/hm_simmim_frozen --bb simmim1 --epochs 60
