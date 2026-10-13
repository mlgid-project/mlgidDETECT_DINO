#!/bin/bash
#SBATCH --job-name=hm_bkg_prof
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/bkg_profile-%j.out
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
R=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
cd $REPO
$PY -u heatmap/bkg_profile_x.py real41=$R/dets_final.pkl:41 legacysim=$R/dets_legacysim.pkl:sim realsimE6=$R/simE6.pkl:realsimE6
