#!/bin/bash
#SBATCH --job-name=hm_highq_img
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/highq_panels-%j.out
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
R=$D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
$PY -u heatmap/highq_ring_panels.py --real $R/dets_final.pkl:41 --out $D/highq_panels --n 6 \
  --sim realsimE6=$R/simE6.pkl:realsimE6 --sim legacysim=$R/dets_legacysim.pkl:sim 2>&1 | grep -v Warn
ls $D/highq_panels
