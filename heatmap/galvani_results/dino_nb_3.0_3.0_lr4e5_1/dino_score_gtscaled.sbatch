#!/bin/bash
#SBATCH --job-name=dino_score
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/dino_score-%j.out
# Scaled-GT scoring of DINO checkpoints on the new-box visfix files (the end-of-job step of dino_nb_3.0_3.0_lr4e5.sbatch was skipped: its guard looked for epoch 379, evals run on even epochs).
#   1) dino_nb_3.0_3.0_lr4e5_1  at HM_GT_BOX_COEF=3.0,3.0  (its training convention)  -> evaluate_final_gtscaled.txt
#   2) dino_lr4e5_1 (old boxes, box_coef_override a=2.8, w=1.3) at 2.8,1.3 -> evaluate_final_gtscaled_oldbox_2.8_1.3.txt  (bundled, so no second job)
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
HM=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
RUNS=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
export PYTHONDONTWRITEBYTECODE=1
score() {   # score CKPT_DIR COEF OUTFILE
  ( cd "$HM" && PYTHONPATH=$HM:$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF="$2" \
    $PY -u heatmap/evaluate.py ssl1=dino:"$1/checkpoint.pth" > "$3" 2>&1 ); echo "score $1 coef $2 rc=$? -> $3"
}
score $RUNS/dino_nb_3.0_3.0_lr4e5_1 3.0,3.0 $RUNS/dino_nb_3.0_3.0_lr4e5_1/evaluate_final_gtscaled.txt
score $RUNS/dino_lr4e5_1 2.8,1.3 $RUNS/dino_lr4e5_1/evaluate_final_gtscaled_oldbox_2.8_1.3.txt
