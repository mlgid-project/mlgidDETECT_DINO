#!/bin/bash
#SBATCH --job-name=dino_nb33
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/dino_nb33-%j.out
# DINO at the heatmap's best box: the nb_legacy baseline A recipe (backbone_curation/ssl/run_detector_nb_legacy.sbatch, config DINO_4scale_swin_ssl_nb.py,
# legacy sim, SimMIM backbone, batch 2, lr = lr_backbone = 4e-5, lr drop 280, dn_number 100, seed 42, eval every 2 epochs) with ONLY
#   box_sigma_halfwidth = (1.5, 1.5)  ->  a_coef = w_coef = 3.0 (box = centre +- 1.5 sigma, same as heatmap --box_coef 3.0,3.0)
#   epochs = 380   (baseline A was stopped at epoch ~370; the lr drop at 280 is unchanged)
# Fresh run in its own folder. The in-training eval scores against the STORED GT (coef 2,2) and is NOT reported; the final numbers come from the
# scaled-GT scoring at the end (HM_GT_BOX_COEF=3.0,3.0). Resumable: on a timeout resubmit the same script (main.py resumes from checkpoint.pth).
# The DINO repo is only read here (never written; no .pyc).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
HM=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
RUNS=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs
OUT_DIR=$RUNS/dino_nb_3.0_3.0_lr4e5_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
NB=/mnt/lustre/work/schreiber/szb389/datasets/newbox
export PYTHONDONTWRITEBYTECODE=1
for f in "$BB" "$NB/organics_visfix.h5" "$NB/perovskites_nan_wedge_visfix.h5"; do [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: no CUDA on $(hostname)"; exit 1; }
mkdir -p "$OUT_DIR"

score() {   # score CKPT_DIR COEF OUTFILE: scaled-GT scoring of a DINO checkpoint with the heatmap evaluator (not fatal)
  ( cd "$HM" && PYTHONPATH=$HM:$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF="$2" \
    $PY -u heatmap/evaluate.py ssl1=dino:"$1/checkpoint.pth" > "$3" 2>&1 ); echo "score $1 coef $2 rc=$? -> $3"
}
# 1) sanity check of the scoring path on baseline A (its stored GT = coef 2,2, so this must reproduce its epoch-370 numbers, organic ~0.60 / 41 ~0.72)
[ -f "$OUT_DIR/sanity_baselineA_coef2.txt" ] || score $RUNS/dino_nb_legacy_lr4e5_1 2.0,2.0 "$OUT_DIR/sanity_baselineA_coef2.txt"

[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start: box (1.5,1.5) sigma half-width = coef 3.0/3.0"
cd "$REPO"
PYTHONPATH=$REPO srun $PY -u main.py -c config/DINO/DINO_4scale_swin_ssl_nb.py --output_dir "$OUT_DIR" \
    --options lr=4e-5 lr_backbone=4e-5 box_sigma_halfwidth=1.5,1.5 epochs=380
echo "train rc=$?"
# 2) final scaled-GT scoring, run to the end only
grep -q "^379" "$OUT_DIR/exp_ap_organic.txt" 2>/dev/null && score "$OUT_DIR" 3.0,3.0 "$OUT_DIR/evaluate_final_gtscaled.txt" || echo "training not at epoch 379, no final scoring"
