#!/bin/bash
#SBATCH --job-name=hm_gtscale_test
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/test_gt_scale-%j.out
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PYTHONPATH=. /home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u heatmap/test_gt_scale.py 2>&1 | grep -v Warning
