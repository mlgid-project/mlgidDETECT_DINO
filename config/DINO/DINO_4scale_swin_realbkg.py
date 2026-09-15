# pygidSIM peaks on REAL, feature-stripped GIWAXS backgrounds.
#
# WHAT THIS CHANGES vs every previous run: the image source. Instead of a modelled background
# (glass + linear + Perlin + Poisson, or the parametric field of sim_image_improvement), each
# frame is a REAL GIWAXS frame from ekaterina_aftermlgidFIT with all of its own diffraction
# features removed, with pygidSIM's peaks drawn onto it. Seams, the low-q halo, radial streaks,
# beam-stop geometry, gain structure and the q-falloff are therefore real, not modelled.
#
# WHAT IS MEASURED, not assumed. Peak widths, shapes, brightness and counts all come from
# mlgidFIT's own 2-D Gaussian fits of 1,926 real peaks in 57 real frames:
#     labelled peaks/frame   sim 28    real 35        sigma_q  px   sim 4.2   real 3.8
#     sigma_chi px           sim 24.5  real 22.5      sigma_chi/q   sim 6.2   real 5.8
#     amp / local noise p50  sim 4.7   real 3.1       HE texture    sim 0.21  real 0.19
# The single largest correction to the old simulator: real peaks are ARCS (median sigma_chi 20 px
# against sigma_q 4 px) and are FAINT (median 3.1x the local noise), where the old sim made round,
# bright blobs.
#
# LEAKAGE: donors are leak-checked against the evaluation sets -- max azimuthal-I(q) cosine to any
# organic_labeled.h5 frame is 0.888, under the 0.949 same-material baseline and far under the
# 0.999 same-frame value.
#
# KNOWN GAPS, both open and both measured (giwaxs_sim.ipynb section 6):
#   * background lag-1 autocorrelation 0.11 against real organic's 0.31;
#   * selected donors score leftover-z ~15 against a detection threshold of 3, so a few frames
#     carry a faint unremoved real feature with NO box on it -- an unlabelled positive that the
#     detector is penalised for finding. This is the main reason to treat the first run as
#     exploratory rather than as a clean A/B.
#
# ONE VARIABLE is not achievable here: the image source changes wholesale, so this is not
# comparable to dino_physics* as a single-lever test. Judge it on the absolute AP curve.
#
# JUDGE POST-280 ONLY -- pre-drop ranking correlates with the plateau at rho = 0.49.

_base_ = ['DINO_4scale_swin_ssl.py']

use_realbkg_sim    = True
physics_bank_path  = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
realbkg_donor_path = '/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5'
realbkg_stats_path = '/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz'
realbkg_n_oriented = (1, 3)
realbkg_p_ring     = 0.15
