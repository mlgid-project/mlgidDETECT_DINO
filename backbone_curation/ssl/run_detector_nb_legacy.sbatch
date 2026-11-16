#!/bin/bash
#SBATCH --job-name=dino_nb_legacy
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205,galvani-cn213   # allocate a GPU torch cannot init
#SBATCH --time=1-00:00:00              # 24h chunks, chain follow-ups with --dependency=afterany (auto-resume)
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_nb_legacy-%j.out
#
# NEW-BOX BASELINE A: the dino_lr4e5_1 recipe, byte for byte except the label convention and the validation files.
#   legacy simulator, 1-channel, SimMIM backbone init, batch 2, lr = lr_backbone = 4e-5, 500 epochs, lr drop 280, dn_number 100,
#   seed 42, eval every 2 epochs on BOTH sets.
#   CHANGED (config DINO_4scale_swin_ssl_nb.py): training boxes = centre +- 1 sigma (box_sigma_halfwidth (1,1) -> a_coef = w_coef = 2.0)
#   instead of 2.80/1.30, and eval_files = the re-fitted, visibility-fixed files in datasets/newbox/ (loaded through PyGIDDataset).
# The point: a NEW AP baseline for the new validation boxes. Nothing recorded before this run is comparable with its APs.
# Old reference (old files, old boxes): organic 0.608, 41 0.761 -- NOT a target any more.
#
# WIRING CHECK at epoch 0 -- kill the run if wrong:
#     [sim] box convention overridden: a_coef=2.0 w_coef=2.0
#     args dump: lr=4e-05 lr_backbone=4e-05 batch_size=2 epochs=500 lr_drop=280 dn_number=100, box_coef_override None,
#                eval_files = newbox/perovskites_nan_wedge_visfix.h5 and newbox/organics_visfix.h5
#     the in-training eval logs two ap_total (41, organic); GT counts 41 1678, organic 817 (diagnostics/newbox_check.py)
# ~9 min/epoch + eval (old lr4e5 log: 8:56 per epoch) -> ~150 epochs per 24h chunk; 500 epochs = ~4 chunks.
# JUDGE on epochs > 280 (after the lr drop; pre-drop ranking only correlates rho 0.49). Cancel the chain when it has the post-drop plateau.
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_nb_legacy_lr4e5_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
NB=/mnt/lustre/work/schreiber/szb389/datasets/newbox
export PYTHONPATH=$REPO

for f in "$BB" "$NB/organics_visfix.h5" "$NB/perovskites_nan_wedge_visfix.h5"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: GPU allocated but torch sees no CUDA device on $(hostname)"; exit 1; }

mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
  || echo "[run] fresh start (SimMIM backbone init, 1-channel, legacy sim, batch 2, lr 4e-5, +-1 sigma boxes, new validation files)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_ssl_nb.py \
    --output_dir "$OUT_DIR" \
    --options lr=4e-5 lr_backbone=4e-5
