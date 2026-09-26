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
# hkl bank, built 2026-09-20: fixed low-index fibre axes, k-orient 12, TOP_PEAKS 2000 --
# 733,933 entries (58,459 powder / 675,474 oriented), 584,960,422 peaks, 5.38 GB. The previous
# bank_organic.npz (random fibre axes, TOP_PEAKS 200) is still on disk beside it. NOTE this makes
# a run incomparable to the dino_physics* arms, which is already true for other reasons.
physics_bank_path  = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic_hkl.npz'
realbkg_donor_path = '/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5'
realbkg_stats_path = '/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz'
realbkg_n_oriented = (1, 3)

# LABELLING CONVENTION, agreed with the user 2026-09-21 after reviewing 25 frames
# (tmp_diag/sim2/images/08_box_convention). A peak is RENDERED IF AND ONLY IF IT GETS A BOX:
# one gate governs both, so the frame has no visible-but-unlabelled tier. Real labelled frames
# have no such tier either (organic_labeled.h5 carries no unlabelled peaks), and the old
# behaviour taught the model to suppress structure that looks exactly like a real peak.
#
# Thresholds are measured, not chosen:
#   * real background sits at 4.1x the local noise, and real labelled peaks at median 3.1x /
#     p10 1.5x (mlgidFIT fits of 1,926 peaks in 57 frames). So "2x the background" can only mean
#     2x the NOISE; against the LEVEL it would be 8.2x and would reject most real labels.
#     2.0 is still stricter than the calibrated p10 of 1.5 -- it labels conservatively on purpose.
#   * real boxes barely overlap: organic 11 overlapping pairs in 27,734, worst IoU 0.154;
#     41 segments 19 in 10,705, worst 0.400; 41 rings ONE pair in 2,220.
#   * real frames carry at most 168 boxes (organic, p50 66) and 65 (41, p50 20).
realbkg_unified_labels = True
realbkg_contrast_min   = 2.0     # peak height / LOCAL BACKGROUND NOISE
realbkg_snr_min        = 6.0     # matched-filter SNR; peaks are arcs, so area counts too
realbkg_seg_iou_max    = 0.30    # segment-segment; None = the historical no-suppression
realbkg_ring_iou_max   = 0.10    # rings, tighter: real rings essentially never overlap
realbkg_max_peaks      = 200     # reflections DRAWN per frame, before any gate

# PEAK COUNTS. Until 2026-09-26 these were module constants in physics_simulation.py with no way
# to reach them from a config, so the label review ran at (2, 200) while a training run would
# have been pinned at (8, 60) -- and the dynamic-range result depends on which is used.
realbkg_spots_cap = (2, 200)     # reflections per ORIENTED entry   (constant was (8, 60))
realbkg_rings_cap = (1, 15)      # rings per POWDER entry           (constant was (3, 15))
#   Lower bound 1, not 3: at 3 a powder entry could never produce a one- or two-ring frame, and
#   real data has 12% of organic frames and 7% of 41 frames exactly there. Raising the ceiling is
#   not wanted -- 15 already exceeds organic's observed max of 11.

# RING RATE. Set by the user 2026-09-26: rings in roughly every third frame, about 9 of them
# when they occur (rings_cap (3,15) has mean 9), so 2.70 rings per frame averaged over all.
#
# Measured from the LABELS, not from model predictions -- note that sim-ring-rate-lever's 3.5 and
# 16.9 were ssl1 predictions at score>0.3, which is the wrong quantity for setting ground truth:
#     organic   2.12 rings/frame, 62% of frames ring-free, max 11, 63.6 segments/frame
#     41        8.85 rings/frame,  0% ring-free,           max 31, 16.5 segments/frame
#     this      2.40 rings/frame, 70% ring-free,           max 15  (rings_cap (1,15), mean 8)
#
# So this sits on ORGANIC's ring composition and does not reach 41, which never has a ring-free
# frame. That is the same position the physics-CIF sim took, and it is why dino_physics3_2 beat
# the baseline on organic (+0.030) and lost 0.189 on 41. Expect that split again.
#
# ONE KNOWN GAP left open on purpose:
#   * rings and segments are ANTI-correlated in real frames (organic -0.48, 41 -0.25, pooled
#     -0.40): a frame is either ring-rich and segment-poor or the reverse. Here they are drawn
#     independently, so the sim makes frames with ~180 segments AND ~20 rings, a combination
#     neither eval set contains. Fixing it means coupling the powder draw to the oriented draw,
#     which is a new mechanism rather than a parameter.
realbkg_p_ring     = 0.30
realbkg_n_powder   = (1, 1)

# MOSAIC BACKGROUNDS. With this on, `realbkg_donor_path` is not read at all. Every background is
# assembled from tiles of the 90 reviewed peak-free bare-silicon Lambda frames
# (tmp_diag/sim2/donors_final.json), built fresh at run start and continuously refreshed, so no
# two runs share background pixels in the same arrangement and nothing is cached between them.
# The old bank is kept only as the source of the smooth radial envelope and the detector masks,
# where its unremoved peaks cannot survive (sigma 64 blur / pure geometry).
realbkg_mosaic         = True
realbkg_mosaic_pool    = 48    # backgrounds held in memory at once
realbkg_mosaic_refresh = 64    # rebuild one pool slot every N simulated frames (~15 ms/frame)

# ARBITRARY INTENSITY RANGE. pygidSIM returns NORMALISED structure-factor intensities; a real
# pyGID frame carries whatever units its own processing left, and the labelled set's maxima span
# 3.6e4 to 2.2e6. Each frame's finished raw image is rescaled so its maximum lands on a
# log-uniform draw over these decades. It is a pure gain -- every ratio, the visibility gate and
# the boxes are unchanged, and the contrast chain is gain-invariant up to HE quantisation
# (measured: mean |diff| 3e-7, max 0.0078 = two of 255 levels). It only makes the RAW frames read
# in realistic units. Set to None to keep raw frames in donor counts.
realbkg_intensity_decades = (3.0, 7.3010)   # 1e3 to 2e7, the range the two
#   validation sets actually occupy: highest fitted peak height per frame runs
#   3.0e3 to 5.1e7 on 41.h5 and 2.3e3 to 1.4e6 on organic_labeled.h5.
# 'pygid': peak amplitude = pygidSIM's normalised intensity x one per-frame scale,
#          so the physics' relative intensities are preserved exactly.
# 'fitted': the old path -- keeps only the ORDERING and redraws values from the
#          lognormal fitted to real labelled peaks (amp/local-noise).
realbkg_amplitude_mode = 'pygid'

# DETECTOR MASKS. Real converted masks (10 geometries: Eiger2 CdTe 4M, Eiger2 4M, Eiger 4M at
# ID10/P08/ID13, 0.099-0.629 m, 15-25 keV) combined per frame with a freshly generated GIWAXS
# missing wedge at a random incidence angle. The mask carries the frame's q_max, because its gaps
# sit at the q they do because of that geometry. Three of the thirteen delivered geometries are
# left out -- see KEEP_DEFAULT in realbkg_sim/detector_masks.py for why.
realbkg_mask_bank = True
realbkg_mask_keep = 'default'
