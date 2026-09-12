# Effective batch 24, lr 4.8e-04 -- arm of the two-point lr bracket above dino_accum24_1.
#
# WHAT THIS TESTS. dino_accum24_1 (same recipe at lr 1.4e-4) set a NEW BEST on 41: post-280 mean
# 0.7784 / max 0.7918, against 0.7622 for dino_truebatch8_1 and 0.7613 for dino_lr4e5_1. That is
# +0.016 on the mean against a post-drop scatter of ~0.006. It paid organic 0.5766 vs 0.6081.
#
# AND IT WAS STILL CLIMBING WHEN THE SCHEDULE STOPPED IT. Slope in ap_total per 100 epochs over
# epochs 180-278 (the window just before the lr drop), on 41:
#     dino_accum24_1   +0.026   still improving
#     dino_lr4e5_1     -0.009   already converged
# Post-drop accum24 is flat (+0.0004 per 100 epochs over epochs 300-462), so it is not short of
# EPOCHS -- it is short of OPTIMIZER STEPS. __len__ is a fixed 1000 images/epoch (main.py:175), so
# batch 2 gives 500 iterations/epoch and grad_accum 12 gives 41.7 optimizer steps/epoch:
#     dino_lr4e5_1     500   steps/epoch   140,000 by the lr drop   183,000 total
#     dino_accum24_1    41.7 steps/epoch    11,667 by the lr drop    19,250 total
# It reached a new best on 41 with 12x fewer weight updates than any baseline.
#
# WHY lr AND NOT A LONGER RUN. Closing the step deficit by running longer costs 12x the wall clock
# (matching 140,000 steps means the drop at epoch 3,360, about 22 days). Raising lr takes BIGGER
# steps instead of more of them at 1x compute, which is the original reason to scale lr with batch
# size. dino_accum24_1 scaled only by sqrt(12). This arm is FULL LINEAR scaling, 12 x 4e-5 -- what the linear-scaling rule prescribes for a 12x batch.
#
# lr = 4.8e-04. Everything else -- grad_accum_steps, warmup_steps, lr_drop, epochs, the SSL
# backbone, the simulator -- is IDENTICAL to DINO_4scale_swin_accum24.py, so lr is the only
# variable across the three-point sweep 1.4e-4 / 2.8e-4 / 4.8e-4. Note warmup_steps stays at 300
# deliberately: changing it with lr would confound the sweep. If this arm fails, warmup LENGTH is
# a separate hypothesis worth its own run, not a reason to distrust the lr reading.
#
# RISK: the aggressive half of the bracket, and a 12x jump over the batch-2 ceiling; this is the one that may flat-fail. The precedent is dino_lrsweep_1 (1.6e-4 at batch 2, NO warmup, NO accumulation),
# which never localized -- loss_giou 1.71 / 1.60 / 1.65 at epochs 20 / 40 / 85 and organic AP stuck
# at 0.011-0.018. That failure does not transfer directly (this run has 12x the effective batch and
# a 300-step warmup), but it is the shape to watch for.
#
# EARLY KILL GATE -- the queue is deep, do not let a dead run hold a slot. Healthy history:
#     loss_giou at ep 20 / 40 / 85:  accum24 0.678 / 0.556 / 0.379   lr4e5 0.502 / 0.391 / 0.346
#     diverged lrsweep_1:            1.706 / 1.597 / 1.647
# KILL IF loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20. Do NOT judge on the
# first few epochs: accum24 was at loss_giou 1.941 at epoch 5 (against lr4e5's 0.744) purely
# because warmup plus 42 steps/epoch makes the start slow, and it still ended up the best run on 41.
#
# JUDGE POST-280 for the verdict; pre-drop ranking correlates with the plateau at only rho = 0.49.
_base_ = ['DINO_4scale_swin_ssl.py']

grad_accum_steps = 12
lr = 4.8e-04
lr_backbone = 4.8e-04
warmup_steps = 300
