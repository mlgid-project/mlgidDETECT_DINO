# Physics-CIF at 100%, ring rate (0,3), and the EXACT real preprocessing with nothing after it.
#
# WHAT THIS TESTS. One variable against dino_physics4_1: `real_tail_only`. Bank, fraction 1.0,
# unify_contrast, physics_n_powder (0,3), lr 4e-5, SSL backbone and box convention are identical.
#
# WHY. `unify_contrast` already made the CONTRAST CHAIN match the real pipeline (percentile clip
# -> log10 -> normalise -> HE, mask-aware). But three ops still ran AFTER it that
# util.exp_preprocess.contrast_correction does not have at all -- it ends at HE:
#     apply_kernel    an unnormalised 3x3 blur (_SMOOTH_KERNEL sums to 8.3, so it scales too)
#     digitalize_img  @with_probability(0.4), quantises to randint(16, 64) grey levels
#     normalize       a second min-max; also the step that produced the NaN crash (section K1)
#
# MEASURED, distinct grey levels in the valid region (20 frames per source):
#     real organic     94 /    116 /     158      (min / median / max)
#     real 41          61 /    109 /     165
#     sim physics       8 /    363 / 367,139
# Real frames cluster tightly at ~110 levels. The simulator is BIMODAL -- 40% of images crushed
# to <=65 levels by digitalize_img, 60% left at full float precision -- and NEITHER mode matches.
#
# THREE SUB-CHANGES, bundled deliberately because they are one hypothesis ("make it exactly the
# real preprocessing"), NOT three independent levers:
#   1. stop after contrast_like_real -- drop the three ops above;
#   2. he_bins 1000 -> 256, the real chain being cv2.equalizeHist on a uint8 image;
#   3. snap onto the k/255 grid (util/exp_preprocess.py:148-151 does img*255 -> uint8 ->
#      equalizeHist -> /255, so a real frame takes at most 256 values on that exact grid).
#      Necessary, not cosmetic: contrast_like_real interpolates the CDF at bin CENTRES, so it
#      yields continuous values AND extrapolates past the last centre -- with the trailing
#      normalize() removed the max came out at 1.00256, i.e. out of range. This is the real
#      pipeline's own quantisation, not a reintroduction of digitalize_img's 16-64 levels.
# If this run wins, it does not say which of the three did it. That split is a later question and
# is only worth asking if the answer is positive.
#
# RESULT OF THE CHANGE, same measurement as above:
#     real organic     94 /    116 /     158      real 41   61 / 109 / 165
#     physics4 (tail)  20 /    248 / 345,584      <- bimodal, matches neither
#     physics5 (this)  24 /    113 /     232      <- median lands on real
# Verified bit-identical on the default path: frame 0 of physics4's stream still gives
# img.sum() = 205665.921875 and 50 boxes, so dino_physics3_2 and dino_physics4_1 are untouched.
#
# WHAT THIS DOES NOT DO. The synthetic PRE-contrast effects stay: mul_perlin, add_glass,
# add_linear_background, apply_poisson_noise, apply_salt_pepper_noise, and the detector geometry
# (add_dark_area, apply_detector_gaps). Real GIWAXS frames genuinely have amorphous background,
# counting noise, a missing wedge and detector gaps, so stripping those would make the simulator
# CLEANER than reality rather than closer to it, and the eval mask depends on the last two.
#
# WIRING CHECK, epoch 0: the banner must end `n_powder=(0, 3), real_tail_only=True`.
#
# INHERITED RISKS, unchanged from section I: peak POSITIONS are a worse match to the eval sets than
# uniform draws (KS sum 0.422 vs 0.280, noise floor 0.048), and the bank is NOT eval-clean
# (--no-exclusions) -- deliberately deferred while the model predicts boxes rather than structure.
_base_ = ['DINO_4scale_swin_ssl.py']

use_physics_sim = True
physics_sim_fraction = 1.0
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
unify_contrast = True
physics_n_powder = (0, 3)
real_tail_only = True
