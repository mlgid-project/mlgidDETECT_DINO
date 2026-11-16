# Physics-CIF peak configuration at 100% (MODIFICATIONS.md section I).
#
# WHAT THIS TESTS. Every training image's peak configuration comes from real crystallography:
# positions AND structure-factor intensities simulated from COD organic CIFs with pygidsim
# (physics_sim/generate_bank.py -> bank npz), spliced into the otherwise unchanged renderer by
# physics_simulation.py. The standard simulator is not used at all. This is the deliberate,
# user-chosen successor to the 25% variant (DINO_4scale_swin_physics2.py): the future model
# iteration is meant to use physical peak POSITIONS as well as intensities, so the training
# distribution should be physical end to end rather than a dilution of a random one.
#
# vs the standard sim, which draws intensities UNIFORMLY (gen_intensities, simulation.py:1161:
# rand()*(hi-lo)+lo over ring (2,50) / segment (10,50)) with no skew and no correlation with q.
# Measured I/Imax shape, pooled per pattern:
#     real organic 0.007 median / 0.911 below 0.1     real 41   0.009 / 0.859
#     bank organic 0.043            / 0.790           uniform sim 0.363 / 0.057
#
# RELATION TO THE DECLINED PHASE P (docs/PHYSICS_SIM_INVESTIGATION.md on branch `development`,
# DECLINED 2026-08-03 at 50% dilution: from-scratch organic 0.5395 vs ssl1 0.5634, 41 0.6255 vs
# 0.7454, and a loss on every stratum at a matched operating point). Differences here:
#   1. Library. Phase P's bank was 98.5% perovskite (26,341 of 26,734 entries). This one is
#      60,474 COD organics from physics_sim/fetch_cod_organics.py, median cell 2374 A^3.
#   2. q coverage. The old bank stopped at |q| = 4.24 (q_xy_max = q_z_max = 3.0), leaving the
#      outer 14% of an organic frame (q_max 4.95) with no physics peaks. Now 3.5 -> 4.95.
#   3. Box convention. Phase P predates box_coef_override and sampled its own half-widths.
#      physics_simulation.py now builds boxes as centre +/- sigma*coef with (a_coef, w_coef) =
#      (2.80, 1.30).
#   4. lr 4e-5 from the base config, where phase P ran at 1e-5.
#   5. Fraction 1.0, where phase P ran 0.5. NOTE this is FURTHER from phase P's setting, not
#      closer -- see the risks below. It is an intentional design choice, not a tuning step.
#
# BUNDLED SECOND CHANGE: unify_contrast. The sim and the real preprocessing are two separate
# implementations that differ in ORDER (sim log -> HE -> clip; real percentile clip -> log -> HE)
# and in the log ARGUMENT (sim maps onto a synthetic decade range normalize(x)*U(50,5000)+1
# because its intensity units are arbitrary; real takes log10(|x|+1e-7)). Physics intensities are
# what make the real form applicable. At fraction 1.0 this now applies to EVERY training image,
# so sim and real frames finally go through the same contrast pipeline -- which is the stated
# point of the track.
#
# THREE RISKS THAT 25% CONTAINED AND 100% DOES NOT. Recorded here so the run is read honestly:
#   a. RING FRACTION. Physics images are 10-12% rings; the standard sim is 52.8%. At 100% the
#      detector sees almost no rings, and 41.h5 is ring-heavy. If 41 AP collapses while organic
#      holds, this is the first thing to check -- it is a composition artefact, not evidence
#      about intensities.
#   b. PEAK POSITIONS are currently a WORSE match to the eval sets than random ones: KS sum vs
#      the real q distribution is 0.422 for the organic bank against 0.280 for the uniform sim
#      (noise floor 0.048). At 100% these are the only positions the model ever sees.
#   c. NOT EVAL-CLEAN. The bank is built with --no-exclusions; the mlgidMATCH eval-exclusion pass
#      (physics_sim/build_exclusion_list.py on `development`) is not ported, so a COD structure
#      matching an eval material can contribute peaks. Any AP from this run is PROVISIONAL until
#      the bank is rebuilt with exclusions.
_base_ = ['DINO_4scale_swin_ssl.py']

use_physics_sim = True
physics_sim_fraction = 1.0
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
unify_contrast = True
