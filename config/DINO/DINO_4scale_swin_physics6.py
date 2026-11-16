# Physics-CIF at 100%, with the per-frame COMPOSITION drawn from a three-branch mixture.
#
# WHAT THIS TESTS. One variable against dino_physics5_1: `physics_frame_types`. Bank, fraction 1.0,
# unify_contrast, real_tail_only, lr 4e-5, SSL backbone and box convention are identical.
#
# WHY. The two eval sets want OPPOSITE compositions -- 41 is 16.90 rings / 24.0 segments per frame
# (ring:seg 0.704), organic is 3.50 / 98.6 (0.035) -- and up to physics5 the simulator drew ONE
# composition for every frame. Widening the ranges cannot fix that, because n_powder and
# n_oriented are drawn independently: at n_powder=(1,4) x n_oriented=(1,7) the 28 equally likely
# cells put 18% at 41's ratio (>= 0.5), 14% at organic's (<= 0.10) and 68% in a composition
# NEITHER eval set contains. Worse, no cell reaches organic's 0.035 at all -- a powder floor of 1
# guarantees ~8.5 rings while real organic has 3.5. 41 needs a ring FLOOR, organic needs a ring
# CEILING; one distribution cannot hold both.
#
# So pick the frame TYPE first, which is what the LEGACY simulator already does at
# simulation.py:514 (`rings_or_seg_or_both` -> rings-only / segments-only / both, 1/3 each) -- and
# the legacy sim is the one that reaches 41 AP 0.78. The physics sim had no equivalent.
#
# THE MIXTURE, (weight, n_powder, n_oriented), equal thirds like the legacy branch. Per-entry
# yields measured over the 0..9 powder sweep: 8.5 rings per powder entry, 24.5 segments per
# oriented entry. MEASURED over 150-200 frames with the geometric ring criterion
# (diagnostics/ring_rate_sims.py: box spans >= 70% of the valid chi rows at its radius):
#
#     branch            n_powder  n_oriented    rings/frame   segs/frame   ring:seg
#     A  41-shaped        (1, 3)     (1, 1)        16.72         24.26       0.689
#       -> real 41                                 16.90         24.00       0.704
#     B  organic-shaped   (0, 1)     (3, 5)         3.68         98.34       0.037
#       -> real organic                             3.50         98.60       0.035
#     C  broad            (0, 4)     (1, 6)        14.87         83.71       0.178
#     POOLED (this run)                            13.16         62.05       0.212
#     physics5 (control)                           13.52         33.65       0.402
#
# Branches A and B land on their targets to within 1-2% on every number. Branch C is deliberately
# NOT a target: it spans the gap and beyond so the model sees compositions outside the two eval
# sets rather than learning exactly two modes. Branch A uses n_oriented=(1,1) rather than (0,2):
# (0,2) gives the same mean but leaves 33% of 41-shaped frames with NO segments, and real 41
# frames always have them.
#
# PRE-REGISTERED OUTCOMES.
#   (i)  41 up, organic up   -- composition was the binding constraint; next lever is the weights.
#   (ii) 41 up, organic down -- branch B is not reproducing organic despite matching its counts;
#        look at what else differs (positions: KS sum 0.422 vs 0.280 for uniform draws).
#   (iii) both flat          -- composition is NOT the limiter and the physics sim's ceiling is
#        elsewhere; stop widening the sim and test the mixture route (physics_sim_fraction < 1).
#
# NOTE ON THE POOLED NUMBERS. Mean ring:seg DROPS 0.402 -> 0.212, the opposite direction to the
# physics3_2 -> physics4 step that bought +0.070 on 41 and +0.045 on organic. That is intentional
# but it IS a hypothesis swap: that step was evidence about the MEAN, and this run bets that the
# per-frame corners matter more than the mean. If (iii) comes out, the mean reading was right.
#
# WIRING CHECK, epoch 0: the banner must end with
#   frame_types=[(0.333..., (1, 3), (1, 1)), (0.333..., (0, 1), (3, 5)), (0.333..., (0, 4), (1, 6))]
# If it reads `frame_types=None` the config key did not reach args and this is a rerun of physics5.
#
# KILL GATES, same as physics5: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20.
#
# INHERITED RISKS, unchanged: peak POSITIONS are a worse match to the eval sets than uniform draws
# (KS sum 0.422 vs 0.280, noise floor 0.048), and the bank is NOT eval-clean (--no-exclusions) --
# deliberately deferred while the model predicts boxes rather than structure.
_base_ = ['DINO_4scale_swin_ssl.py']

use_physics_sim = True
physics_sim_fraction = 1.0
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
unify_contrast = True
real_tail_only = True
# (weight, n_powder, n_oriented); weights are normalised in PhysicsSimulation._parse_frame_types.
physics_frame_types = [
    (1.0, (1, 3), (1, 1)),      # A  41-shaped:      16.7 rings / 24.3 segs / ring:seg 0.689
    (1.0, (0, 1), (3, 5)),      # B  organic-shaped:  3.7        / 98.3      / 0.037
    (1.0, (0, 4), (1, 6)),      # C  broad:          14.9        / 83.7      / 0.178
]
