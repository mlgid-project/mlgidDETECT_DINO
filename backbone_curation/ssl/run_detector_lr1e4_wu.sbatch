#!/bin/bash
#SBATCH --job-name=dino_lr1e4_wu
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_lr1e4_wu-%j.out
#
# Warmup experiment: lr = lr_backbone = 1e-4 with a 1000-step linear warmup.
# The ONLY differences vs the ssl1 recipe are those three numbers -- see
# DINO_4scale_swin_lr1e4_wu.py (verified by SLConfig diff: lr 1e-5 -> 1e-4,
# lr_backbone 1e-5 -> 1e-4, plus the new key warmup_steps; nothing else moves).
# 1-channel, SSL backbone init, legacy simulation, 2.80/1.30 boxes, same 500ep/lr_drop 280.
#
# WHY: two runs above 4e-5 have now failed identically -- dino_lr1e4_1 (1e-4/1e-5) and
# dino_lrsweep_1 (1.6e-4/1.6e-4) both classify fine (class_error 37.7% -> 1.5%) but never
# localize (loss_giou stuck at 1.59 / 1.71 at epoch 85, vs 0.346 for the dino_lr4e5_1
# winner). Both lost it in epochs 0-2: batch8_1 went giou 2.30 -> 1.60 -> 1.05 -> 0.74 by
# epoch 5, while lr1e4_1 went 2.51 -> 2.03 -> 2.05 -> 1.88 and never recovered in 88 epochs.
# There is no warmup anywhere in the codebase, so step 1 already runs at full lr; combined
# with clip_max_norm=0.1 every early step is a fixed-length move in a near-random direction.
# 1000 steps = 2.0 epochs at batch 2 (500 steps/epoch), which covers that window.
# lr_backbone is 1e-4, NOT lr/10: all 31 runs in detector_runs/ use lr == lr_backbone,
# including the winner, and the 10:1 split in dino_lr1e4_1 confounded that result.
# Warmup is implemented in engine.py:train_one_epoch and scales each param group by its own
# 'initial_lr', so the backbone and the 0.1x lr_linear_proj groups keep their ratios.
#
# Gate: organic/41 ap_total vs ssl1's 0.5683 / 0.7441; the bar to beat is dino_lr4e5_1's
# organic 0.6102 plateau (lr 4e-5). WIRING CHECK in epoch 0: the per-step `lr:` field must
# climb from ~0 toward 0.000100 over the first 1000 steps, not sit at 0.000100 from step 0.
# EARLY CHECK at epoch 5 on loss_giou.txt (the ramp pushes this later than the usual ep6-10
# organic check): batch8_1 was at 0.744 by epoch 5. Under ~1.3 means it took hold; still
# near 1.9 means warmup did not help and the ceiling just above 4e-5 is real -> kill it.
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same
# command across 72h sessions. Per-epoch eval logs `ap_total` for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_lr1e4wu_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, legacy sim, batch 2, lr = lr_backbone = 1e-4, 1000-step linear warmup)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_lr1e4_wu.py \
    --output_dir "$OUT_DIR"
