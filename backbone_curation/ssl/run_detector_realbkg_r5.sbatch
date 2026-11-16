#!/bin/bash
#SBATCH --job-name=dino_realbkg5
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg5-%j.out
#
# RUN 5: half the images from the LEGACY simulator. ONE VARIABLE against run 3.
# Config: DINO_4scale_swin_realbkg_r5.py -- read its header for the evidence.
#
# Every realbkg run decays with training; every synthetic-background run improves. conv3 went
# organic 0.594 (ep 60-100) -> 0.496 (plateau) while its TRAIN loss fell 37%. physics4_1 uses the
# same CIF bank on synthetic backgrounds and improves, and randomisation was checked and is fine
# (160/160 distinct frames over four epochs, zero repeats between epochs). The cause is the
# backgrounds: a 48-slot pool from 90 donor frames, each background reused ~21x per epoch.
#
# realbkg_fraction=0.5 adds unlimited perlin backgrounds for half the stream AND brings in the
# legacy sim, which scores 0.74-0.77 on 41 where every realbkg run sits at 0.38-0.43.
#
# WIRING CHECK at epoch 0 -- kill the run if this is wrong:
#     [sim] realbkg share of images: 50% (remainder from the LEGACY sim)
# If it says 100%, the guard did not take and this is a duplicate of run 3.
# Also expect, unchanged from run 3:
#     [sim] geometry: ring box VALID CHI SPAN | organic-like segment frames 0.50
#     the args dump must show dn_number=100, NOT 400 (that is run 4's variable)
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or EITHER gate's ap_total < 0.10 at epoch 20.
#
# HOW TO JUDGE IT. Do not read the plateau alone. The question this run asks is whether the DECAY
# stops, so report the early window (60-100) AND the post-280 plateau and compare the DELTA
# against run 3's -0.098 on organic. A plateau that merely matches run 3 while the early window
# drops would mean the legacy half diluted the organic gain without fixing the memorisation.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv5
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
    -c config/DINO/DINO_4scale_swin_realbkg_r5.py \
    --output_dir "$OUT_DIR"
