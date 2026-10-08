#!/bin/bash
#SBATCH --job-name=hm_score_boxconv1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/score_boxconv1-%j.out
# Score the two reference DINO models with the SAME code as the heatmap runs (heatmap/evaluate.py):
# Headline convention: score floor 0.1 + top-225 + class-aware NMS (ssl1 and dino_lr4e5_1 were scored by job 2949416). AP, recall/precision at score>0.1 and >0.3, segment vs ring recall, recall by nearest-neighbour distance.
# Needs the compiled DINO ops (so: cluster, not colorbox1). Short: two checkpoints x two sets, ~5 min.
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
RUNS=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
OUT=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/heatmap_runs/dino_scoring_boxconv1.txt
export PYTHONPATH=$REPO
cd $REPO
mkdir -p "$(dirname $OUT)"
$PY -u heatmap/evaluate.py boxconv1=dino:$RUNS/dino_boxconv1/checkpoint.pth 2>&1 | grep -v "Warning\|warn" | tee "$OUT"
