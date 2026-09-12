#!/bin/bash
#SBATCH --job-name=dino_a24lr48
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_a24lr48-%j.out
#
# Effective batch 24 at lr 4.8e-4. Config: DINO_4scale_swin_accum24_lr48.py -- read its header for
# the measurement this is built on and the early kill gate.
#
# ONE VARIABLE vs dino_accum24_1 (lr 1.4e-4, post-280 41 mean 0.7784 / max 0.7918, a new best).
# grad_accum_steps, warmup_steps, lr_drop, epochs, backbone and simulator are all identical; only
# lr and lr_backbone differ. The two arms 2.8e-4 and 4.8e-4 bracket linear scaling (12 x 4e-5).
#
# WIRING CHECK, epoch 0: the config dump in $OUT_DIR/config_args_all.json must read
#   "lr": 4.8e-4, "lr_backbone": 4.8e-4, "grad_accum_steps": 12, "warmup_steps": 300
# and the first Epoch lines must show lr ramping from ~0 over the first 300 OPTIMIZER steps
# (= 7.2 epochs at 41.7 steps/epoch), not starting at full value.
#
# EARLY KILL GATE. The a100-galvani queue is deep -- do not let a dead run hold a slot.
#   KILL IF loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20.
# Reference: healthy runs reach loss_giou 0.39-0.56 by epoch 40; dino_lrsweep_1, which never
# localized, sat at 1.597. Do NOT judge the first few epochs -- dino_accum24_1 was at 1.941 at
# epoch 5 (lr4e5: 0.744) because warmup plus 42 steps/epoch makes the start slow, and it still
# finished as the best run on 41.
#
# JUDGE POST-280 ONLY. Pre-drop ranking correlates with the plateau at rho = 0.49.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. dino_accum24_1 reached epoch 462 of 500 in one 72h session.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_accum24_lr48_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, accum 12 x batch 2 = effective 24, lr = lr_backbone = 4.8e-4, warmup 300 opt steps)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_accum24_lr48.py \
    --output_dir "$OUT_DIR"
