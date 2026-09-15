#!/bin/bash
#SBATCH --job-name=dino_dn400
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_dn400-%j.out
#
# physics4's data with DINO's contrastive denoising run at full strength: dn_number 100 -> 400.
# Config: DINO_4scale_swin_physics4_dn400.py -- read its header for the measurement chain, the
# pre-registered outcomes and the honest caveat about the missing legacy control.
#
# ONE VARIABLE vs dino_physics4_1 (finished at epoch 434, organic 0.5804 / 41 0.6412). Verified by
# diffing the resolved configs: the ONLY key that differs is dn_number.
#
# WHY. The physics model is recall-strong and precision-poor: on organic it recovers 76.8 of 102.1
# GT peaks per frame against lr4e5's 69.5, and uncapped it reaches recall 0.886 where lr4e5
# saturates at 0.687 -- but precision 0.434 vs 0.643. The excess is a spray of DUPLICATES around
# real peaks (near-miss FPs 53.2/frame vs lr4e5's 10.2 on organic), sitting at IoU 0.1-0.4 with the
# true box, just under the 0.4 segment NMS threshold. Contrastive denoising is the mechanism that
# teaches one box per object, and `dn_number // max_gt_in_batch` (dn_components.py:43) leaves our
# ~50-70-box frames training at 1-2 groups where COCO gets ~14.
#
# NOT A FIX FOR THE LEGACY SIM'S GAP: the legacy sim is starved identically (~46 boxes/frame), so a
# gain here may be a general DINO gain. The legacy-sim control arm is NOT queued (no free GPU) and
# a positive result is ambiguous until it runs.
#
# WIRING CHECK, epoch 0: config_args_all.json in the output dir must read "dn_number": 400 and
# "physics_n_powder": [0, 3]. dn_number 100 means the key did not reach args -- kill, it is a
# duplicate of dino_physics4_1.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20. An OOM in
# the first epochs points at the enlarged dn query block (decoder sequence ~1030 -> ~1680).
#
# JUDGE POST-280 ONLY -- pre-drop ranking correlates with the plateau at rho = 0.49.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics4_dn400_1
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
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 100% physics-CIF images, n_powder (0,3), dn_number 400, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics4_dn400.py \
    --output_dir "$OUT_DIR"
