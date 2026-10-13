#!/bin/bash
#SBATCH --job-name=hm_ring_kind
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/ring_miss_kind-%j.out
# Missed real 41 rings: detected-but-unmatched (box shape) or not detected at all? E1b / E5 / E6 / E7.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
$PY -u heatmap/ring_miss_kind.py $D/hm_simmim_frozen_ridge_realsimE1b_tf32_seed42_2.80_1.30/dets_final.pkl $D/hm_simmim_frozen_ridge_realsimE5_tf32_seed42_2.80_1.30/dets_final.pkl $D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30/dets_final.pkl $D/hm_simmim_frozen_ridge_realsimE7_tf32_seed42_2.80_1.30/dets_final.pkl 2>&1 | grep -v Warn
