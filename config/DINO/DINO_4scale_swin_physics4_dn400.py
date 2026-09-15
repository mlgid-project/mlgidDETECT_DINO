# dino_physics4_dn400_1 -- physics4's data, DINO's denoising run at full strength.
#
# ONE VARIABLE vs dino_physics4_1 (finished, epoch 434, organic 0.5804 / 41 0.6412): dn_number
# 100 -> 400. Same bank, same physics_n_powder=(0, 3), same unify_contrast, same SSL backbone,
# same lr 4e-5, same box convention, same preprocessing (real_tail_only is NOT set here, exactly
# as in physics4). Inherits DINO_4scale_swin_physics4.py so nothing else can drift.
#
# WHY. Measured 2026-09-15 on the finished checkpoints (`diagnostics/numselect_sweep.py`,
# `$WORK/tmp_diag/{capsplit,fploc,tpiou,nmssweep}.py`), at the deployed operating point
# (top-225, class-aware NMS ring 0.1 / seg 0.4, score > 0.1):
#
#   gate     model     recall  precision  ap_total   med TP score  med FP score
#   41       lr4e5      0.888    0.445     0.7631       0.848         0.198
#   41       physics4   0.856    0.256     0.6412       0.657         0.258
#   organic  lr4e5      0.681    0.643     0.6222       0.879         0.212
#   organic  physics4   0.752    0.434     0.5804       0.612         0.294
#
# The physics model is NOT peak-blind -- on organic it recovers 76.8 of 102.1 GT peaks per frame
# against lr4e5's 69.5, and uncapped (num_select 900) it reaches recall 0.886 where lr4e5
# SATURATES at 0.687. It loses the AP back on precision. Its matched boxes are fine too (median
# TP IoU 0.394 vs lr4e5's 0.343 on organic -- it localises BETTER).
#
# The excess is a SPRAY OF DUPLICATES around real peaks, not noise firings. Splitting every false
# positive by IoU with the nearest GT:
#   model      gate      dup(IoU>0.3)   near(0<IoU<=0.3)   bg(IoU==0)
#   physics4   41          3.9/fr          55.3/fr          42.9/fr
#   lr4e5      41          1.5/fr          13.0/fr          30.9/fr
#   physics4   organic     7.5/fr          53.2/fr          39.2/fr
#   lr4e5      organic     0.8/fr          10.2/fr          27.6/fr
# Background firings are comparable (1.4x); the NEAR bucket is 4.3x. Those boxes sit at IoU
# 0.1-0.4 with the true box -- just under the 0.4 segment NMS threshold, so they survive.
#
# Confirmed by the NMS sweep: dropping segment NMS IoU 0.4 -> 0.10 is worth physics4 +0.042 on 41
# and +0.032 on organic, while moving lr4e5 by <=0.002 and ssl1 by <=0.007. The duplicates are
# real and suppressible. That is a postprocessing patch on a model-side problem, and is NOT
# shipped -- this run tries to make the model stop emitting them.
#
# THE LEVER. `prepare_for_cdn` (models/dino/dn_components.py:42-43) turns dn_number into GROUPS as
# `dn_number // max_gt_in_batch`. Contrastive denoising -- reconstruct the box from a noised copy,
# reject the more-noised negative -- is precisely the mechanism that teaches one box per object.
# Our frames carry ~50-70 GT boxes, so at the default dn_number=100 we train at ~1-2 groups where
# COCO's ~7-object images give ~14. dn_number=400 restores ~6 groups.
#
# HONEST CAVEAT: the legacy sim is starved the same way (~46 boxes/frame), so this may be a
# GENERAL DINO gain rather than a physics-specific fix. The legacy-sim control arm (lr4e5 recipe,
# dn_number=400) is deliberately NOT queued yet -- no free a100-galvani. Until it runs, a positive
# result here does NOT establish that the hedging is what got fixed.
#
# PRE-REGISTERED OUTCOMES, judged post-280 against dino_physics4_1 at matched epochs:
#   1. organic UP and 41 UP, with near-FP/frame down toward lr4e5's 10-13 -> denoising was the
#      binding constraint; queue the legacy control to see if it is physics-specific.
#   2. AP flat but near-FP/frame down -> duplicates suppressed without buying accuracy; the AP
#      cost of the spray was smaller than the NMS sweep implied.
#   3. AP flat and near-FP/frame flat -> dn groups are not what teaches suppression here; drop
#      this lever and go to cls_loss_coef / focal_alpha.
#   4. AP DOWN -> the extra dn queries are crowding out the matching queries; kill.
#
# WIRING CHECK, epoch 0: `config_args_all.json` in the output dir must read "dn_number": 400 and
# "physics_n_powder": [0, 3]. If dn_number is 100 the config key did not reach args and this run
# is a duplicate of dino_physics4_1 -- kill it rather than burn 72h.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20. Healthy
# runs reach loss_giou 0.39-0.56 by epoch 40.
#
# MEMORY: dn adds `max_gt * 2 * groups` queries to the 900 matching queries, so the decoder
# sequence goes from ~1030 to ~1680. swin-L batch-2 training measures ~8.7 GB on a 40 GB a100, so
# there is ample headroom, but an OOM in the first epochs would point here first.
_base_ = ['DINO_4scale_swin_physics4.py']

dn_number = 400
