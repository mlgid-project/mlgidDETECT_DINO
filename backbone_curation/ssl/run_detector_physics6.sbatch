#!/bin/bash
#SBATCH --job-name=dino_phys6
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_phys6-%j.out
#
# 100% physics-CIF training images with the per-frame COMPOSITION drawn from a three-branch
# mixture. Config: DINO_4scale_swin_physics6.py -- read its header for the measured per-branch
# composition table and the pre-registered outcomes.
#
# ONE VARIABLE vs dino_physics5_1, the control: same bank, same fraction 1.0, same unify_contrast,
# same real_tail_only, same lr 4e-5, same SSL backbone, same box convention. Only
# `physics_frame_types` differs (it also supersedes physics_n_powder, which the branches set
# per-frame), so any AP difference is the composition alone.
#
# WHY. 41 wants 16.90 rings / 24.0 segments per frame (ring:seg 0.704), organic wants 3.50 / 98.6
# (0.035). Up to physics5 the simulator drew ONE composition for every frame, and widening the
# ranges cannot reach both: n_powder and n_oriented are drawn independently, so n_powder=(1,4) x
# n_oriented=(1,7) puts 68% of frames in a composition neither eval set contains, and NO setting
# reaches organic's 0.035 because a powder floor of 1 forces ~8.5 rings against organic's 3.5.
# The LEGACY sim -- the one that gets 41 AP 0.78 -- already picks a frame type first
# (simulation.py:514, rings-only / segments-only / both at 1/3 each). This ports that idea over.
#
# MEASURED composition, geometric ring criterion, 150-200 frames per branch:
#     branch            n_powder  n_oriented   rings/frame  segs/frame  ring:seg
#     A  41-shaped        (1, 3)     (1, 1)       16.72        24.26      0.689   (real 41: 16.90 / 24.0 / 0.704)
#     B  organic-shaped   (0, 1)     (3, 5)        3.68        98.34      0.037   (real org: 3.50 / 98.6 / 0.035)
#     C  broad            (0, 4)     (1, 6)       14.87        83.71      0.178
#     POOLED (this run)                           13.16        62.05      0.212
#     physics5 (control)                          13.52        33.65      0.402
#
# BIT-IDENTITY: frame_types=None leaves the old path untouched -- HEAD vs patched, 5 frames, both
# streams, identical img.sum() and box counts. dino_physics4_1 and dino_physics5_1 stay valid
# controls even if they requeue onto the patched code.
#
# WIRING CHECK, epoch 0: the banner must END with a three-entry frame_types list:
#   ... n_powder=(0, 1), real_tail_only=True, frame_types=[(0.333..., (1, 3), (1, 1)),
#       (0.333..., (0, 1), (3, 5)), (0.333..., (0, 4), (1, 6))]
# If it reads `frame_types=None` the config key did not reach args and this run is a duplicate of
# dino_physics5_1 -- kill it rather than let it burn 72h. (n_powder printing as its (0, 1) default
# is EXPECTED here: the branches override it per frame.)
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20. Healthy
# runs reach loss_giou 0.39-0.56 by epoch 40; dino_lrsweep_1, dino_accum24_lr28_1 and
# dino_accum24_lr48_1, none of which ever localized, sat at 1.60 / 1.75 / 1.79.
#
# JUDGE POST-280 ONLY for the verdict -- pre-drop ranking correlates with the plateau at rho = 0.49.
# Compare at MATCHED epochs against dino_physics5_1 for spotting a dead run early, not for calling
# the result.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. Per-epoch eval logs ap_total for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics6_1
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
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 100% physics-CIF images, 3-branch frame_types mixture, real_tail_only, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics6.py \
    --output_dir "$OUT_DIR"
