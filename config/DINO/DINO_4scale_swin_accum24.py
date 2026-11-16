# Effective batch 24 by gradient accumulation (MODIFICATIONS.md: batch/lr axis).
#
# WHAT THIS TESTS. Every run in detector_runs/ is batch 2. The one real large-batch attempt,
# dino_truebatch8_1 (batch 8, lr unchanged at 4e-5, otherwise identical), LOST: organic 0.5808 vs
# dino_lr4e5_1's 0.6081, 41 0.7622 vs 0.7613. But that run is confounded by optimizer steps --
# __len__ is a fixed 1000 images/epoch, so batch 8 got 4x fewer updates for the same epoch count
# and may simply have been undertrained rather than harmed by the batch size. This run separates
# the two by raising lr with the batch instead of holding it fixed.
#
# grad_accum_steps 12 x batch_size 2 = effective batch 24 at the MEMORY COST OF 2, so the swin-L
# backbone at 512x1024 still fits on one a100. engine.py divides the loss by accum (mean, not sum,
# over the effective batch), clips the ACCUMULATED gradient, and counts warmup in OPTIMIZER steps.
#
# LEARNING RATE 1.4e-4 = sqrt(12) x 4e-5, the standard Adam-family scaling rule.
# This deliberately enters a band that FAILED at batch 2: the base config records 1e-4 and 1.6e-4
# both classifying fine (class_error 37.7% -> 1.5%) but never localizing (loss_giou stuck at
# 1.59 / 1.71 at epoch 85 against 0.35), and a 1000-step warmup reaching 1e-4 only to walk back
# out mid-epoch-3. The hypothesis is that the ceiling was a GRADIENT-NOISE ceiling, not an lr
# ceiling: 12x the batch cuts gradient noise ~3.5x, which is exactly what would lift it. If
# loss_giou is still stuck above ~1.5 by epoch 85, that hypothesis is dead and the answer is that
# 4e-5 is an lr ceiling independent of batch size.
# warmup_steps 300 = ~7.2 epochs at 42 optimizer steps/epoch (~1.4% of the run's 20,833 steps).
#
# SIZING -- compute-matched AND axis-matched to dino_lr4e5_1 (0.6081 / 0.7613), on purpose:
#   1000 images/epoch (main.py:163, unchanged) x 500 epochs = 500,000 images, identical to it.
#   500 dataloader iterations/epoch -> 42 optimizer steps/epoch -> 20,833 total (vs 250,000 at
#   batch 2); lr_drop at epoch 280 = 11,667 steps. ~76 h.
# NOTE raising images/epoch does NOT buy optimizer steps: total steps = total images / effective
# batch regardless of how epochs are sliced. Only more compute, a higher lr, or a smaller
# effective batch can close that gap. The 12x step deficit is intrinsic and is what lr must cover.
#
# CAVEAT -- NOT BIT-EXACT TO A TRUE BATCH 24. DINO normalizes its losses by num_boxes over the
# batch (models/dino/dino.py:408,413). Accumulating 12 micro-batches each normalized by its OWN
# box count yields the mean of per-micro-batch means, not the true batch-24 mean, so images in
# box-sparse micro-batches are weighted up. This is the standard accumulation approximation and
# is accepted here, but it is a real difference from dino_truebatch8_1's genuine batch 8.
_base_ = ['DINO_4scale_swin_ssl.py']

grad_accum_steps = 12
lr = 1.4e-04
lr_backbone = 1.4e-04
warmup_steps = 300
