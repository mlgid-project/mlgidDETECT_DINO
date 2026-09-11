# Physics-CIF at 100%, with the RING RATE widened (MODIFICATIONS.md section L).
#
# WHAT THIS TESTS. dino_physics3_2 (DINO_4scale_swin_physics3.py) is this run's only difference:
# physics_n_powder = (0, 3) instead of the (0, 1) default in physics_simulation.py. Everything
# else -- bank, fraction 1.0, unify_contrast, lr 4e-5, SSL backbone, box convention -- is
# identical, so any AP difference is attributable to the ring rate alone.
#
# WHY. Risk (a) in the physics3 header -- "RING FRACTION ... if 41 AP collapses while organic
# holds, this is the first thing to check" -- is exactly what happened, and it is now measured.
# 120 frames per simulator, geometric ring criterion (box spans >= 70% of the valid chi rows at
# its radius, is_ring_geom in diagnostics/postproc_diag.py), current box convention:
#
#     source            rings/frame  segs/frame  ring:seg  frames with 0 rings
#     real 41              16.90        24.0       0.704        --
#     real organic          3.50        98.6       0.035        --
#     sim legacy           17.05        30.7       0.555       34/120
#     sim physics-CIF       4.38        37.2       0.118       62/120
#
# The legacy sim sits on 41's composition; the physics sim sits on organic's. That is the whole
# shape of the physics3_2 result: +0.030 organic over dino_lr4e5_1 (33 of 35 shared epochs, mean
# over ep>=50 0.5528 vs 0.5224) and -0.189 on 41 (34 of 35, 0.5275 vs 0.7167). Rings are the EASY
# class (41 ring recall 0.856 vs 0.713 for segments) and 41% of 41's objects, so a quarter of the
# rings costs 41 far more than it costs organic.
#
# WHY (0, 3) AND NOT A HIGHER FIXED VALUE. The two gates want OPPOSITE ring rates (0.704 vs
# 0.035), so no single rate matches both -- a compromise matches neither. A WIDE per-image range
# makes the regime a per-image random variable: randint(0, 3) still leaves ~25% of frames
# ring-free (organic-like) while the top of the range reaches ~13 rings/frame (41-like). The
# legacy sim does the same thing coarsely, via rings_or_seg_or_both in simulation.py:486 splitting
# images into thirds (rings only / segments only / both).
#
# WHAT WOULD FALSIFY THE IDEA. If 41 recovers toward the 0.72-0.76 band while organic keeps
# physics3_2's advantage, the ring rate was the whole story. If 41 recovers but organic falls back
# to the legacy 0.55-0.62 band, then ring rate simply TRADES the two gates and the physics bank
# buys nothing beyond a different point on the same curve. If 41 does not recover at all, the
# cause is elsewhere -- peak POSITIONS (risk b: the bank's q distribution is a WORSE match to the
# eval sets than uniform, KS sum 0.422 vs 0.280, noise floor 0.048) become the prime suspect.
#
# INHERITED RISKS, unchanged from physics3: peak positions (b above) and NOT EVAL-CLEAN (the bank
# is built with --no-exclusions, so a COD structure matching an eval material can contribute
# peaks). Any AP from this run is PROVISIONAL until the bank is rebuilt with exclusions.
_base_ = ['DINO_4scale_swin_ssl.py']

use_physics_sim = True
physics_sim_fraction = 1.0
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
unify_contrast = True
physics_n_powder = (0, 3)
