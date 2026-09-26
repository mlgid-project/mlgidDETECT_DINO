#!/bin/bash
#SBATCH --job-name=dynrange
#SBATCH --partition=v100-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/dynrange_%j.out
set -euo pipefail
/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python -u \
  /mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/realbkg_dynrange.py
