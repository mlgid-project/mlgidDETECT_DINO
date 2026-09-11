#!/bin/bash
#SBATCH --job-name=dino_phys4
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_phys4-%j.out
#
# 100% physics-CIF training images with the RING RATE widened to physics_n_powder = (0, 3).
# Config: DINO_4scale_swin_physics4.py -- read its header, it carries the measurement this run is
# built on and the three outcomes that would falsify the idea.
#
# ONE VARIABLE vs dino_physics3_2. Same bank, same fraction 1.0, same unify_contrast, same lr
# 4e-5, same SSL backbone, same box convention. Only the per-image powder-entry count differs,
# (0, 1) -> (0, 3), so any AP difference between the two runs is the ring rate alone.
#
# WIRING CHECK, epoch 0: the banner must end with `n_powder=(0, 3)`. The full line is
#   [sim] physics sim ON: 100% of images from .../bank_organic.npz (58459 powder / 467672
#   oriented entries), unify_contrast=True, n_powder=(0, 3)
# If it says n_powder=(0, 1) the config key did not reach args and the run is a duplicate of
# physics3_2 -- kill it rather than let it burn 72h.
#
# WHAT TO WATCH. dino_physics3_2 is the control and it is running concurrently, so compare at
# MATCHED epochs, not against the plateau numbers of finished runs. By epoch ~50 physics3_2 sits
# at organic 0.553 / 41 0.528 (mean over ep>=50); dino_lr4e5_1, the legacy-sim reference, sits at
# organic 0.522 / 41 0.717. The question this run answers is whether 41 climbs off 0.53 toward
# that 0.72-0.76 band WITHOUT giving back physics3_2's organic advantage.
#
# JUDGE POST-280 ONLY for the final verdict: pre-drop ranking correlates with the plateau at only
# rho = 0.49. The matched-epoch comparison above is for spotting a dead run early, not for calling
# the result.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. Per-epoch eval logs ap_total for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics4_1
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
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 100% physics-CIF images, n_powder=(0,3), unified contrast, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics4.py \
    --output_dir "$OUT_DIR"
