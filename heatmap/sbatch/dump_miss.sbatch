#!/bin/bash
#SBATCH --job-name=hm_dump_miss
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/dump_miss-%j.out
# Detections of a finished galvani run (final checkpoint, +nms floor 0.1 top-225, organic + 41) and the miss analysis on them. Inference only.
# usage: sbatch --export=ALL,RUN=hm_simmim_frozen_ridge_realsimE1b_tf32_seed42_2.80_1.30 heatmap/sbatch/dump_miss.sbatch
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)  run $RUN"
$PY -u heatmap/dump_dets.py --ckpt $D/$RUN/final_checkpoint.pth --out $D/$RUN/dets_final.pkl --sets organic 41 | tail -3
for thr in 0.3; do $PY -u heatmap/miss_analysis.py $D/$RUN/dets_final.pkl --sets 41 organic --thr $thr > $D/$RUN/miss_analysis_final.txt 2>&1; echo "miss rc=$?"; done
cat $D/$RUN/miss_analysis_final.txt
