#!/bin/bash
#SBATCH --job-name=dino_realbkg3
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg3-%j.out
#
# RUN 3: two MEASURED box-geometry defects fixed, composition spanning both gates.
# Config: DINO_4scale_swin_realbkg_r3.py -- read its header for the evidence behind each change.
#
# Runs 1 and 2 scored organic 0.518/0.517 and 41 0.394/0.428 against lr4e5_1's 0.608/0.761. What
# was ruled OUT by measurement, so it is not being changed here:
#   * the labelling convention -- paired on identical frames it moves 4.3% of boxes (run 1) / 0.5%
#     (run 2), and the brightness gate runs in both label paths
#   * frame composition -- run 2 matched 41 within 10% on every axis and gained 0.034 of 0.333
#   * amplitude_mode='pygid' -- raw contrast is 222x local noise vs real 3.1x, but after CHAIN's
#     histogram equalisation the sim sits at p50 1.10 vs real organic 1.12 / 41 1.11
#
# WHAT CHANGED, and the numbers behind it:
#   1. Ring boxes now span the VALID CHI SPAN at their q, not the full 512 px frame. A full-height
#      box cannot be matched at IoU 0.5 against 24.7% of 41's real rings = 10.3% of every box on
#      that gate. The legacy sim behind 0.748 never did this (27.3% full height, not 100%).
#   2. Half the frames draw organic-like segment shapes. Emitted aspect was 5.1 against organic's
#      0.36 and 41's 3.45, so >90% of simulated segments were taller than organic's MEDIAN peak.
#
# VERIFIED BEFORE SUBMIT (diagnostics/realbkg_box_sizes.py, 150 frames). Every axis now sits
# BETWEEN the two gates instead of outside both:
#      ring h / valid span 1.05        (organic 1.03-1.14, 41 0.98)
#      rings full height    4.3%       (organic 80%, 41 0.6%)
#      segment aspect       1.22       (organic 0.36, 41 3.45)
#      segment box w/h   7.3 / 18.9    (organic 10.5/8.1, 41 4.6/34.2)
#      boxes/frame   p10 13 p50 44 p90 93   (organic p50 66, 41 p50 20)
#      rings/frame 4.97, ring-free 38%      (organic 2.12/62%, 41 8.85/0%)
#
# WIRING CHECK at epoch 0 -- kill the run if any of these is wrong:
#     [sim] geometry: ring box VALID CHI SPAN | organic-like segment frames 0.50
#     [sim] counts: spots/entry (2, 70) rings/entry (1, 13) | oriented/frame (1, 3) | p_ring 0.6
# If geometry says "full frame height" the fix did not reach the simulator and this is run 1 with
# a different composition -- kill it.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or EITHER gate's ap_total < 0.10 at epoch 20.
# Judge only on epochs > 280 (lr_drop); pre-drop ranking correlates only rho=0.49.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv3
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
    -c config/DINO/DINO_4scale_swin_realbkg_r3.py \
    --output_dir "$OUT_DIR"
