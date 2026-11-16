#!/bin/bash
#SBATCH --job-name=dino_phys5
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_phys5-%j.out
#
# 100% physics-CIF training images, ring rate (0, 3), and the EXACT real preprocessing with
# nothing after it. Config: DINO_4scale_swin_physics5.py -- read its header for the grey-level
# measurement this run is built on and for what it deliberately does NOT change.
#
# ONE VARIABLE vs dino_physics4_1, which is running concurrently as the control: same bank, same
# fraction 1.0, same unify_contrast, same physics_n_powder (0, 3), same lr 4e-5, same SSL backbone,
# same box convention. Only `real_tail_only` differs, so any AP difference is the preprocessing
# tail alone.
#
# WHAT real_tail_only DOES. util.exp_preprocess.contrast_correction -- the real pipeline -- ends at
# HE. Three ops used to run after contrast_like_real that have no counterpart there:
#   apply_kernel (unnormalised 3x3 blur), digitalize_img (p=0.4, quantise to 16-64 grey levels),
#   normalize (a second min-max). It also sets he_bins 1000 -> 256, the real chain being
#   cv2.equalizeHist on uint8. Measured distinct grey levels in the valid region: real organic
#   94/116/158 and real 41 61/109/165 (min/median/max), against sim physics 8/363/367,139 -- the
#   simulator was bimodal and matched neither mode.
#
# WIRING CHECK, epoch 0: the banner must END with `n_powder=(0, 3), real_tail_only=True`:
#   [sim] physics sim ON: 100% of images from .../bank_organic.npz (58459 powder / 467672
#   oriented entries), unify_contrast=True, n_powder=(0, 3), real_tail_only=True
# If it reads real_tail_only=False the config key did not reach args and this run is a duplicate of
# dino_physics4_1 -- kill it rather than let it burn 72h.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20. Healthy
# runs reach loss_giou 0.39-0.56 by epoch 40; dino_lrsweep_1, dino_accum24_lr28_1 and
# dino_accum24_lr48_1, none of which ever localized, sat at 1.60 / 1.75 / 1.79.
#
# JUDGE POST-280 ONLY for the verdict -- pre-drop ranking correlates with the plateau at rho = 0.49.
# Compare at MATCHED epochs against dino_physics4_1 for spotting a dead run early, not for calling
# the result.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. Per-epoch eval logs ap_total for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics5_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
if [ ! -f "$BANK" ]; then
    echo "ERROR: physics peak bank not found at $BANK"
    echo "       build it first: sbatch physics_sim/run_bank_organic.sbatch"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 100% physics-CIF images, n_powder=(0,3), real_tail_only, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics5.py \
    --output_dir "$OUT_DIR"
