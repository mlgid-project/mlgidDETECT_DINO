#!/bin/bash
#SBATCH --job-name=dino_phys3
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_phys3-%j.out
#
# 100% physics-CIF training images. Config: DINO_4scale_swin_physics3.py -- read its header
# first: it records the three risks that the 25% variant contained and this run does not
# (ring fraction 10-12% vs the standard sim's 52.8%, peak positions a WORSE q match than random,
# and a bank that is not eval-clean).
#
# vs the ssl1 recipe this changes exactly two things, deliberately bundled:
#   use_physics_sim / physics_sim_fraction 1.0   -- EVERY image in the 1000-image epoch takes its
#       peak positions and STRUCTURE-FACTOR INTENSITIES from 60,474 COD organic CIFs via pygidsim
#       instead of the uniform (2,50)/(10,50) draws in gen_intensities. The standard simulator's
#       peak sampler is not used at all; its appearance chain (widths, Perlin, Poisson, detector
#       gaps, missing wedge, box convention) still is.
#   unify_contrast                               -- the REAL contrast order (percentile clip ->
#       log10(|x|+1e-7) -> HE, mask-aware, simulation.contrast_like_real) rather than the sim's
#       log -> HE -> clip with a synthetic decade range. At fraction 1.0 this now holds for every
#       training image, so sim and real frames share one preprocessing path.
# Everything else inherited: 1-channel, SSL backbone init, 2.80/1.30 boxes, lr = lr_backbone
# = 4e-5, batch 2, 500 ep, lr_drop 280, 1000 images/epoch (main.py:163).
#
# GATES.
#   Wiring check, epoch 0: the banner must print "[sim] physics sim ON: 100% of images from
#   .../bank_organic.npz (N powder / M oriented entries), unify_contrast=True". If the bank path
#   is wrong main.py raises rather than silently training on standard images.
#   Sanity, epoch ~5: loss_giou should track a normal run (batch8_1 was ~1.05 -> 0.74 over epochs
#   1-3). PhysicsSimulation raises after 20 empty draws rather than hanging, but note
#   SimulationDataset.__getitem__ swallows exceptions in a bare except and retries forever, so a
#   stalled epoch 0 with NO error means the physics generator is failing every draw -- check the
#   bank before assuming a GPU problem.
#   VERDICT: organic/41 AP judged ONLY on the post-280 plateau (pre-drop ranking correlates only
#   rho=0.49 with the settled one) against dino_lr4e5_1's 0.6081 / 0.7613.
#   A fixed-score recall comparison is NOT admissible: phase P looked good at score 0.3 only
#   because dilution made it fire 5.5x more boxes (56.9 vs 10.4 FP/img) while losing at every
#   matched precision. Compare at matched precision (~0.83) and matched FP/img (~11).
#   If 41 drops but organic holds, check the ring fraction before concluding anything about
#   intensities -- see risk (a) in the config header.
#
# RESTARTED FROM EPOCH 0 as dino_physics3_2. dino_physics3_1 crashed twice on a NaN training
# image (AssertionError at util/box_ops.py:52 -- boxes1 there is the PREDICTIONS, so NaN model
# output in an amp=False run, so a NaN input). One image in ~20,000 arrives constant and
# normalize()'s 0/0 turns it into NaN everywhere; the old guards missed it because
# `img.min() == img.max()` is False when both are NaN. Fixed by _usable() in physics_simulation.py
# plus a belt in main.SimulationDataset.__getitem__ for both simulators.
# The second fix is why this starts over rather than resuming: the RNG was seeded once per
# PROCESS, so _1's resume regenerated epochs 0-40 image for image as its epochs 41-81 -- 82
# epochs over ~41,000 distinct images. main.py now reseeds per epoch, so the stream depends on
# the epoch number, not on when the job was restarted. _1's weights carry both defects in their
# history, so they are not a valid warm start for the fixed recipe.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. Per-epoch eval logs ap_total for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics3_2
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
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 100% physics-CIF images, unified contrast, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics3.py \
    --output_dir "$OUT_DIR"
