#!/bin/bash
#SBATCH --job-name=dino_realbkg7
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg7-%j.out
#
# RUN 7: what the RINGS look like. Against run 6 only the ring draw changes (config r7); background
# v2, labels, lr 4e-5, batch 2, dn_number 100, SSL backbone all as run 6.
# Config: DINO_4scale_swin_realbkg_r7.py -- read its header for the evidence.
#
# WHY. Every realbkg run sits at 41 ap 0.38-0.44 vs lr4e5_1's 0.761. The realbkg models chop 41's
# rings into short segment boxes (conv3: 4.7 boxes per ring, ring score 0.30, 61 of 99 FP/frame on
# rings). Cause, found by read-only probes (commits 973958e, 023193d): realbkg rings are perfectly
# SMOOTH along chi (patch CV 0.003 vs real 41 0.106). Gentle texture on the sim's own rings makes
# conv3 fragment them exactly as on 41 (AP 0.93 -> 0.46, 2.7 -> 9.0 boxes/ring) while the
# legacy-trained lr4e5 is unaffected (1.64 boxes/ring, its 41 value is 1.7). Ruled out: blur,
# contrast, ring count, overlap, box width, spacing, preprocessing.
#
# WIRING CHECK at epoch 0 -- kill the run if any of these is wrong:
#     [sim] counts: spots/entry (2, 100) rings/entry (2, 60) | oriented/frame (1, 4) | p_ring 0.75 powder/frame (1, 3)
#     [sim] background: v2 -- fresh crop + PCA envelope per frame, 50% surrogate texture
#     [sim] rings: segments kept on rings 0.12 | ring box w_coef 3.0 | ring sigma_q x0.7 |
#           ring min gap 8.0 px | ring chi texture sd 0.45 corr (1.0, 64.0) px
#     the args dump must show dn_number=100
# PRE-FLIGHT (diagnostics/r7_preflight.py, job 2945806): all of the above reach the simulator via
# main.SimulationDataset; 344 ms/frame vs r6 355 -> ~15 min/epoch as run 6.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or EITHER gate's ap_total < 0.10 at epoch 20.
#
# HOW TO JUDGE IT. The question is whether 41 moves off the realbkg plateau (0.38-0.44; run 6 best
# 0.417) toward lr4e5_1's 0.761 while organic keeps realbkg's early strength (run 6 0.60-0.63).
# realbkg runs reach their 41 level by epoch 20-30, so the first answer is early: compare r7 vs
# run 6 at matched epochs 20-40. Rerun diagnostics/ring_fp_probe.py on checkpoint_best_41.pth to
# see whether boxes per ring falls toward lr4e5's 1.7.
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv7
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
    -c config/DINO/DINO_4scale_swin_realbkg_r7.py \
    --output_dir "$OUT_DIR"
