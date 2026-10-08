#!/bin/bash
#SBATCH --job-name=hm_smoke
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/smoke-%j.out
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd $REPO
$PY -u heatmap/train.py --out /mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/heatmap_runs/smoke --max_steps 20
