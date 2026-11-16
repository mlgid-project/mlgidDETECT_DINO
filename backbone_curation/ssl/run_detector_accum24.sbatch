#!/bin/bash
#SBATCH --job-name=dino_accum24
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max; the run needs ~76h -> one resume
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_accum24-%j.out
#
# Effective batch 24 via gradient accumulation, single GPU. Config: DINO_4scale_swin_accum24.py
# -- read its header: it records why lr moves to 1.4e-4, why the step deficit cannot be fixed by
# changing images/epoch, and the num_boxes normalization caveat.
#
# vs the ssl1 recipe this changes exactly three things, and they are NOT separable by design:
#   grad_accum_steps 12   -- effective batch 24 at the memory cost of 2 (micro-batch stays 2, so
#                            swin-L at 512x1024 still fits on one a100).
#   lr = lr_backbone 1.4e-4 -- sqrt(12) x 4e-5. Raising lr WITH the batch is the whole point; the
#                            one prior large-batch run (dino_truebatch8_1, batch 8 at unchanged
#                            4e-5) lost organic 0.5808 vs 0.6081 and is confounded by having had
#                            4x fewer optimizer steps.
#   warmup_steps 300      -- optimizer steps (~7.2 epochs), standard for large batch.
# Everything else inherited: 1-channel, SSL backbone init, 2.80/1.30 boxes, batch_size 2,
# 500 ep, lr_drop 280, clip_max_norm 0.1, 1000 images/epoch (main.py:163).
#
# SINGLE GPU ON PURPOSE. DDP is not wired in this repo (main.py:323 has init_distributed_mode
# commented out, model_without_ddp = model with no DDP wrap, DistributedSampler imported but
# unused). A 2-GPU allocation would idle the second card. Accumulation first so that a negative
# result is attributable to batch size rather than to an untested distribution path.
#
# GATES.
#   Wiring, epoch 0: lr in the log must RAMP from ~4.7e-7 toward 1.4e-4 over the first 300
#   optimizer steps (~7 epochs), not start at 1.4e-4. If it starts flat at 1.4e-4 the warmup is
#   being counted in iterations or not read at all.
#   THE DECISIVE EARLY GATE, epoch ~85: loss_giou. The base config records 1e-4 and 1.6e-4 at
#   batch 2 classifying fine but never localizing -- giou stuck at 1.59 / 1.71 against 0.35 for
#   the working recipe. If giou is still above ~1.5 here, the batch-24 noise reduction did NOT
#   lift the ceiling, 4e-5 is an lr ceiling independent of batch size, and this run can be killed
#   early rather than burning 76h. If giou tracks toward ~0.35, the hypothesis is alive.
#   VERDICT: organic/41 AP judged ONLY on the post-280 plateau (pre-drop ranking correlates only
#   rho=0.49 with the settled one) against dino_lr4e5_1's 0.6081 / 0.7613 -- compute-matched and
#   axis-matched to this run, so the curves overlay directly.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir. The run needs ~76h against a
# 72h limit, so expect exactly one resubmit of this same command.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_accum24_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, accum 12 x batch 2 = effective 24, lr = lr_backbone = 1.4e-4, warmup 300 opt steps)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_accum24.py \
    --output_dir "$OUT_DIR"
