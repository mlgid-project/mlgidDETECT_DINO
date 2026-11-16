#!/bin/bash
#SBATCH --job-name=dino_realbkg4
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg4-%j.out
#
# RUN 4: ONE VARIABLE against run 3 -- dn_number 100 -> 400.
# Config: DINO_4scale_swin_realbkg_r4.py -- read its header for the group arithmetic.
#
# Run 3 fixed two measured box-geometry defects and is, at epochs 60-100, ahead of every baseline
# on organic (0.595 vs lr4e5_1's 0.525 in the same window) and still far behind on 41 (0.423 vs
# 0.734). The ring-span fix bought ~+0.045 on 41 against a 0.31 gap, so geometry was real but not
# the 41 mechanism.
#
# The 41 mechanism was measured on 2026-09-15: a CONFIDENCE CALIBRATION failure, not blindness.
# On 41 the physics sim loses no recall against the baseline (0.856 vs 0.888) while its precision
# halves (0.256 vs 0.445); score separation TP-FP is 0.399 against 0.650; and 74% of its false
# positives are NEAR-MISSES on its own simulated distribution, so it is not domain transfer and
# not a data bug. That analysis named dn400 as the right lever. It was never run -- dn_number is
# 100 in every run in this project's history.
#
# dn_components.py:42-43 makes the lever larger than it looks:
#     groups = (2 * dn_number) // (2 * max_boxes_in_batch)
# COCO's ~7 boxes/image give ~14 groups, which is where DINO was tuned. Our frames carry p10 13 /
# p50 44 / p90 93 boxes and at batch 2 the max-of-two is typically 50-95, so this project has
# trained at ONE OR TWO denoising groups throughout. dn=400 restores 4-9.
#
# WIRING CHECK at epoch 0 -- kill the run if any of these is wrong:
#     the args dump must show dn_number=400
#     [sim] realbkg share of images: 100%
#     [sim] geometry: ring box VALID CHI SPAN | organic-like segment frames 0.50
#     [sim] counts: spots/entry (2, 70) rings/entry (1, 13) | oriented/frame (1, 3) | p_ring 0.6
# If the share line is missing, main.py predates the realbkg_fraction guard; if it is not 100%,
# this is a MIXTURE run and not the one-variable dn test -- kill it either way.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or EITHER gate's ap_total < 0.10 at epoch 20.
#
# WALLTIME WARNING: dn=400 adds ~700 decoder queries on top of 900, and decoder self-attention is
# quadratic, so expect ~1.3-1.5x run 3's epoch time. 72 h reaches roughly epoch 290-330 against
# run 3's ~430, and lr_drop is at 280 -- the post-drop window will be SHORT. Resume from
# checkpoint.pth for a second allocation rather than judging this on a handful of post-drop evals.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv4
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
    -c config/DINO/DINO_4scale_swin_realbkg_r4.py \
    --output_dir "$OUT_DIR"
