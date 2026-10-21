#!/bin/bash
#SBATCH --job-name=oldckpt_newbox
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205,galvani-cn213
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/oldckpt_newbox-%j.out
#
# EVAL ONLY, no training: the OLD baseline dino_lr4e5_1 (trained on 2.8/1.3 boxes, last checkpoint.pth) scored on the NEW
# re-fitted validation files (visfix). Answers: is the new-convention AP drop (heatmap 41 ~0.725 -> 0.684) the benchmark
# or the training boxes? Old numbers of this checkpoint on the OLD files: organic 0.620, 41 0.754 (epoch 366 eval).
# Same deployed postprocessing as in-training eval (top-225, NMS, score>0.1). Model output is scored as is.
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OLD=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_lr4e5_1
NB=/mnt/lustre/work/schreiber/szb389/datasets/newbox
OUT=/mnt/lustre/work/schreiber/szb389/tmp_diag/oldckpt_newbox
export PYTHONPATH=$REPO
mkdir -p "$OUT/organic" "$OUT/41"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: no CUDA on $(hostname)"; exit 1; }
cd "$REPO"
for pair in "organic:$NB/organics_visfix.h5" "41:$NB/perovskites_nan_wedge_visfix.h5"; do
  name=${pair%%:*}; f=${pair#*:}
  echo "=================== OLD dino_lr4e5_1 on NEW $name ($f)"
  $PY -u main.py --eval -c config/DINO/DINO_4scale_swin_ssl_nb.py --resume "$OLD" \
      --eval_file "$f" --output_dir "$OUT/$name" --options lr=4e-5 lr_backbone=4e-5
done
