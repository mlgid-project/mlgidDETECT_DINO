#!/bin/bash
#SBATCH --job-name=dino_realbkg2
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg2-%j.out
#
# RUN 2: same bank, same labelling convention, frame composition aimed at 41.
# Config: DINO_4scale_swin_realbkg_r2.py -- read its header for the measured composition.
#
# THE ONE AXIS. Run 1 (dino_realbkg_conv1) makes 1.57 rings and 94.6 segments per frame,
# ring:segment 0.017, with 73% of frames ring-free. Real 41 is 8.85 / 16.5, ratio 0.536, and has
# NO ring-free frame. Run 1 scored 0.558 organic / 0.395 on 41 over epochs 140-166, against
# lr4e5_1's 0.553 / 0.743 -- the best organic number we have and the worst 41 one.
#
# This run measures 9.28 rings / 18.3 segments, ratio 0.507, 0% ring-free: within ~10% of 41 on
# every axis. Everything else is inherited from the run-1 config, so run 1 vs run 2 is a clean
# one-axis comparison of frame composition. A middle value was deliberately NOT chosen -- it
# matches neither gate; run 1 holds the organic end and this holds the 41 end.
#
# WIRING CHECK, epoch 0: the log must show
#     [sim] labels: render-iff-labelled | contrast>=2.0 snr>=6.0 | IoU seg 0.3 ring 0.1 | max_peaks 200
#     [sim] counts: spots/entry (6, 32) rings/entry (3, 13) | oriented/frame (1, 2) | p_ring 1.0 powder/frame (1, 2)
# If counts shows (2, 200) or p_ring 0.3 the r2 config did not take and this is a duplicate of
# run 1 -- kill it.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or 41 ap_total < 0.10 at epoch 20. Note the gate
# is on 41 here, not organic: this composition deliberately starves the organic end (63.6
# segments/frame there against this run's 18.3), so a weak organic number is the expected cost,
# not a fault.
#
# JUDGE POST-280 ONLY -- pre-drop ranking correlates with the plateau at rho = 0.49.
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv2
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic_hkl.npz
DONORS=/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5
STATS=/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz
export PYTHONPATH=$REPO

for f in "$BB" "$BANK" "$DONORS" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
  || echo "[run] fresh start (SSL backbone, 1-channel, real-background sim, batch 2, lr 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_realbkg_r2.py \
    --output_dir "$OUT_DIR"
