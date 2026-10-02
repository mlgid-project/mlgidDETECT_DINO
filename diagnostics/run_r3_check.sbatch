#!/bin/bash
#SBATCH --job-name=r3check
#SBATCH --partition=a100-galvani
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/r3check-%j.out
set -e
cd /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
export PYTHONPATH=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
echo "######## run 1 config (spots 2-200, p_ring 0.30)"
$PY -u diagnostics/realbkg_box_sizes.py --frames 150 --config config/DINO/DINO_4scale_swin_realbkg_r3.py
echo; echo "######## run 2 config (spots 6-32, p_ring 1.0)"
$PY -u diagnostics/realbkg_box_sizes.py --frames 150 --config config/DINO/DINO_4scale_swin_realbkg_r3.py
