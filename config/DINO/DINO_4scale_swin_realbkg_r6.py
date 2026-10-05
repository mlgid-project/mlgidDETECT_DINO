_base_ = ['DINO_4scale_swin_realbkg_r3.py']

# RUN 6. ONE VARIABLE against run 3: where the background comes from.
# dn_number stays 100 (run 4's dn400 was ~neutral: +0.014 organic, -0.016 on 41 at matched epochs,
# and confounded by the envelope bug below), so run 3 vs run 6 reads the background cleanly.
#
# WHY. Every realbkg run DECAYS with training while every synthetic-background run IMPROVES:
#     organic, best early window -> post-drop plateau
#     conv1 0.578 -> 0.518   conv2 0.552 -> 0.517   conv3 0.594 -> 0.496   conv4 0.606 -> 0.561
#     lr4e5_1 0.525 -> 0.608  boxconv1 0.480 -> 0.585  ssl1 0.510 -> 0.562  physics4_1 -> 0.556
# Overfitting, not instability: conv3's TRAIN loss fell 37% (giou 0.548 -> 0.345) while organic AP
# fell 0.594 -> 0.496. NOT the bank -- physics4_1 runs the same CIF bank on synthetic backgrounds
# and improves. NOT broken randomisation -- 160/160 distinct frames over four epochs, zero repeats
# between any pair (diagnostics/image_freshness.py).
#
# THE BUG FOUND WHILE BUILDING THIS. `_envelope()` drew the whole large-scale q/chi shape from
# bkg.npy, which is the REJECTED 189-donor bank -- every one of those frames carries unremoved
# real diffraction, which is why the donor set was cut to the 90 clean bare-Si frames the TILES
# come from. So every background carried one of 189 sigma-64 smeared ring systems, and after the
# contrast chain it reads as a bright ARC across the frame with no box on it. Measured, that
# profile DIPS to 0.45 at low q where a real background is brightest, bumps at q~400, and falls to
# 0.04 at high q against the clean donors' 0.6. MosaicBackground.USE_ENV_BANK is now False, so the
# envelope comes from the clean donors on every path. See tmp_diag/sim2/images/09_background_v2.
#
# WHAT v2 CHANGES. The flat fluctuation canvas is cached; everything else is per frame.
#   A   fresh crop per frame. Free -- the crop was being CACHED while the 183 ms envelope was
#       recomputed every entry, both backwards.
#   B2  the envelope is SAMPLED from a PCA over the 90 clean donors' own smooth shapes (6 PCs hold
#       99.3% of their log-variance), clipped to the per-pixel range the donors span so a sample
#       cannot leave the real family. Replaces picking one of a fixed set. This is the axis
#       re-cropping does not touch, and the envelope is the only part of the frame carrying
#       structure -- the tiles are divided by their own blur and carry none.
#   B1  surrogate_frac of frames take their texture from a random-phase surrogate: the measured
#       power spectrum with new phases, never repeating. MIXED, not substituted, because it
#       discards the higher-order structure -- hot pixels, line defects, panel seams -- that only
#       real tiles carry and that the detector should learn to ignore.
realbkg_bkg_v2        = True
realbkg_surrogate_frac = 0.5
realbkg_v2_canvas     = (1536, 3072)
realbkg_v2_refresh    = 200
realbkg_v2_n_pc       = 6

# Verified against the real donors before submitting: A+B2 tracks the real radial profile across
# the whole q range and matches real dynamic range (max/median 2.2 vs real 2.3; the old path 2.9).
