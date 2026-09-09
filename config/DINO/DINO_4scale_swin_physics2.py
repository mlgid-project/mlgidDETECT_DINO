# Physics-CIF peak configuration, second attempt (MODIFICATIONS.md section I).
#
# WHAT THIS TESTS. The standard simulator draws peak intensities UNIFORMLY in a bounded range
# (gen_intensities, simulation.py:1161: rand()*(hi-lo)+lo over ring (2,50) / segment (10,50)),
# then rescales linearly -- no skew and essentially no correlation with q. Real diffraction has a
# few strong reflections and a long weak tail spanning orders of magnitude, set by structure and
# form factors. pygidsim produces exactly that from a CIF, and physics_simulation.py splices those
# peak lists into the otherwise unchanged renderer. Peak POSITIONS come along for the ride; the
# intensities are the point.
#
# RELATION TO THE DECLINED PHASE P (docs/PHYSICS_SIM_INVESTIGATION.md on branch `development`).
# That run was DECLINED 2026-08-03: from-scratch organic 0.5395 vs ssl1's 0.5634, 41 0.6255 vs
# 0.7454, and at a matched operating point it lost on every stratum. Four things differ here:
#   1. Library. Phase P's bank was 98.5% perovskite (26,341 of 26,734 entries from a COD
#      perovskite selection, 393 from 51 organic CIFs). This one is 60,474 COD organics selected
#      by physics_sim/fetch_cod_organics.py, median cell 2374 A^3.
#   2. q coverage. The old bank stopped at |q| = 4.24 (q_xy_max = q_z_max = 3.0), leaving the
#      outer 14% of an organic frame (q_max 4.95) with no physics peaks. Now 3.5 -> 4.95.
#   3. Box convention. Phase P predates box_coef_override and sampled its own half-widths, so its
#      box-size statistics differed from the current recipe. physics_simulation.py now builds
#      boxes as centre +/- sigma*coef with the run's own (a_coef, w_coef) = (2.80, 1.30).
#   4. Dilution 25%, not 50%. Phase P's failure mode was a trigger-happy detector firing 5.5x
#      more boxes; heavy dilution with a mismatched distribution is what produces that.
# Plus lr 4e-5 from the base config, where phase P ran at 1e-5.
#
# BUNDLED SECOND CHANGE: unify_contrast. The sim and the real preprocessing are two separate
# implementations that differ in ORDER (sim log -> HE -> clip; real percentile clip -> log -> HE)
# and in the log ARGUMENT (sim maps onto a synthetic decade range normalize(x)*U(50,5000)+1
# because its intensity units are arbitrary; real takes log10(|x|+1e-7)). Physics intensities are
# what make the real form applicable, so the two levers are tested together -- deliberately, at
# the cost of attribution if the run moves.
#
# CAVEAT -- NOT EVAL-CLEAN YET. The bank is built with --no-exclusions: the mlgidMATCH-based
# eval-exclusion pass (physics_sim/build_exclusion_list.py on `development`) is not ported, so a
# COD structure matching an eval material can still contribute peaks. Treat any AP from this run
# as provisional until the bank is rebuilt with exclusions.
_base_ = ['DINO_4scale_swin_ssl.py']

use_physics_sim = True
physics_sim_fraction = 0.25
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
unify_contrast = True
