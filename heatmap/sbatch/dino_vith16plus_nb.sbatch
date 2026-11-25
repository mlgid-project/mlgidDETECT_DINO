#!/bin/bash
#SBATCH --job-name=dino_vith
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/dino_vith-%j.out
# FLAGSHIP DINO (2026-11-22): the DINO detector (4 scale, nb recipe as dino_nb_3.0_3.0_lr4e5_1: legacy sim, batch 2, lr 4e-5, lr drop 280, dn_number 100, box 3.0/3.0, 380 epochs) with a DINOv3 ViT-H+/16 BACKBONE
# (config/DINO/DINO_4scale_vith16plus_nb.py, adapter models/dinov3_backbone.py DinoV3DetBackbone, unfrozen, lr_backbone 1e-5, gradient checkpointing) instead of the SimMIM swin-L.
# Env: W = backbone weights (default the public DINOv3 ViT-H+; for an own pretraining pass W=<pretrain_runs/.../backbone_teacher.pth>), TAG = output folder suffix (default pub), SEED (default 42).
# Code: ~/mlgidDETECT_DINO_HEATMAP_flag (git archive of colorbox dev), NOT the old mlgidDETECT_DINO clone. Resumable: on a timeout resubmit the same command (main.py resumes from checkpoint.pth).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP_flag
HM=$REPO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
RUNS=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
SEED=${SEED:-42}; TAG=${TAG:-pub}
W=${W:-/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}
OUT_DIR=$RUNS/dino_vith16plus_${TAG}_s$SEED
export HM_DINOV3_REPO=/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_repo
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
NB=/mnt/lustre/work/schreiber/szb389/datasets/newbox
export PYTHONDONTWRITEBYTECODE=1
for f in "$W" "$NB/organics_visfix.h5" "$NB/perovskites_nan_wedge_visfix.h5"; do [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: no CUDA on $(hostname)"; exit 1; }
mkdir -p "$OUT_DIR"

score() {   # score CKPT_DIR COEF OUTFILE: scaled-GT scoring of a DINO checkpoint with the heatmap evaluator (not fatal)
  ( cd "$HM" && PYTHONPATH=$HM:$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF="$2" \
    $PY -u heatmap/evaluate.py ssl1=dino:"$1/checkpoint.pth" > "$3" 2>&1 ); echo "score $1 coef $2 rc=$? -> $3"
}
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start: box (1.5,1.5) sigma half-width = coef 3.0/3.0"
cd "$REPO"
PYTHONPATH=$REPO srun $PY -u main.py -c config/DINO/DINO_4scale_vith16plus_nb.py --output_dir "$OUT_DIR" --seed $SEED \
    --options lr=4e-5 lr_backbone=1e-5 box_sigma_halfwidth=1.5,1.5 epochs=380 backbone_weights=$W
echo "train rc=$?"
# 2) final scaled-GT scoring, run to the end only
grep -q "^378" "$OUT_DIR/exp_ap_organic.txt" 2>/dev/null && score "$OUT_DIR" 3.0,3.0 "$OUT_DIR/evaluate_final_gtscaled.txt" || echo "training not at epoch 378 (the last even eval epoch of 380), no final scoring"
