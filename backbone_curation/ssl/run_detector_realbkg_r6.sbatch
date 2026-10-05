#!/bin/bash
#SBATCH --job-name=dino_realbkg6
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg6-%j.out
#
# RUN 6: new background source. ONE VARIABLE against run 3.
# Config: DINO_4scale_swin_realbkg_r6.py -- read its header for the evidence.
#
# Every realbkg run decayed with training while every synthetic-background run improved. conv3
# went organic 0.594 (ep 60-100) -> 0.496 (plateau) with its TRAIN loss falling 37%. Not the bank
# (physics4_1 improves on the same CIF bank), not randomisation (160/160 distinct frames over four
# epochs, zero repeats). It was the backgrounds: a 48-slot pool, each reused ~21x per epoch.
#
# AND AN ENVELOPE BUG, found while building this: _envelope() drew the whole large-scale shape
# from bkg.npy, the REJECTED 189-donor bank whose frames all carry unremoved diffraction. After
# the contrast chain that reads as a bright ARC across the frame with no box on it. Fixed --
# MosaicBackground.USE_ENV_BANK is False, so every path now takes its envelope from the 90 clean
# donors.
#
# WIRING CHECK at epoch 0 -- kill the run if any of these is wrong:
#     [realbkg] background v2: fresh crop + PCA envelope PER FRAME from 90 clean donors
#     [sim] background: v2 -- fresh crop + PCA envelope per frame, 50% surrogate texture
#     [sim] geometry: ring box VALID CHI SPAN | organic-like segment frames 0.50
#     the args dump must show dn_number=100
# If the background line says "48-slot cached pool (v1)" this is run 3 again -- kill it.
#
# PRE-FLIGHT (diagnostics/v2_smoke.py, run before submitting): 329 ms/frame against v1's ~300,
# so about +10% epoch time; 24/24 distinct backgrounds in consecutive frames; boxes/frame p50 42,
# rings/frame 3.38, ring-free 46%.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or EITHER gate's ap_total < 0.10 at epoch 20.
#
# HOW TO JUDGE IT. The question is whether the DECAY STOPS. Report the early window (60-100) AND
# the post-280 plateau and compare the delta against run 3's -0.098 on organic. Best checkpoints
# are now written per gate and for their sum (checkpoint_best_*.pth), because the realbkg runs
# peak early -- conv4's best organic was 0.613 at EPOCH 32 and those weights were never saved.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv6
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic_hkl.npz
DONORS=/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5
STATS=/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz
export PYTHONPATH=$REPO

for f in "$BB" "$BANK" "$DONORS" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
# Job 2920327 was allocated gres/gpu:a100=1 on galvani-cn205 and torch still raised
# "No CUDA GPUs are available" 19 s in. Fail here instead, with the node named.
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: GPU allocated but torch sees no CUDA device on $(hostname)"; exit 1; }

mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
  || echo "[run] fresh start (SSL backbone, 1-channel, real-background sim, batch 2, lr 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_realbkg_r6.py \
    --output_dir "$OUT_DIR"
