_base_ = ['DINO_4scale_swin_realbkg_r6.py']

# RUN 7. Against run 6: what the RINGS look like. Background (v2), labels, lr, dn stay as run 6.
#
# WHY (realbkg-41-gap-split, 2026-10-06, all read-only probes in diagnostics/):
#   * the realbkg 41 gap is CONFIDENCE, not precision: physics4 is as imprecise (0.255 vs
#     0.25-0.29) yet scores 0.642 because its TP-FP score separation is 0.395 vs realbkg 0.11-0.19
#   * and it is RINGS: conv3 puts 61 of 99 FP/frame on GT rings (lr4e5 9), as short fragments
#     (median 14% of the chi span), 4.7 predictions per ring, ring TP score 0.30 vs 0.72 segments
#   * out-of-distribution ring APPEARANCE: conv3 scores its OWN sim's rings at 0.96 but LEGACY-sim
#     rings at 0.36 with 3.7 boxes/ring -- the same failure as on 41 -- while legacy segments are
#     fine (0.94). lr4e5 handles realbkg rings at 0.91-0.94. realbkg rings are a narrow family.
#   * NOT blur: blurring real 41 makes the realbkg models much worse (conv3 0.380 -> 0.114 at 3 px)
#   * NOT contrast: dimming r7 rings ~250x to 41's processed contrast leaves conv3 at 0.93
#   * THE CAUSE IS RING TEXTURE ALONG CHI. realbkg rings render perfectly smooth (patch-scale CV
#     0.003 vs real 41 0.106, legacy 0.031 via mul_perlin + Poisson). Gentle broadband texture
#     on r7's OWN rings makes conv3 fragment them exactly as on 41 (texture_scan.py, max sd 0.45:
#     AP 0.93 -> 0.46, 2.7 -> 9.0 boxes per ring, segments untouched at 0.986) while the
#     legacy-trained lr4e5 is unaffected (AP 0.637 -> 0.614, 1.64 boxes/ring vs its 1.7 on 41)
#     -- i.e. the texture is the kind a model that handles 41 already tolerates.
#
# WHAT CHANGES, each matched to a measured target (ring_fix_sweep.py, 150 frames, processed image;
# real 41 / legacy / run 6):
#   rings per frame        17.9 broad to 63, 43% >= 21  | 17.0 to 55 | 17.3, 44% >= 21 | 6.1
#   rings with a segment   17%                          | 15%        | 2%              | 44%
#   ring box width p50     7.2 px                       | 7.6        | 6.8             | 5.0
#   visible FWHM / box     1.77                         | 1.48       | 1.79            | 2.42
#   segments per frame     42.7 (68.7 in ring-free)     | 24         | 32              | 42.5
#   bands unlabelled:rings 7.6 : 17.9                   | 10.4 : 17  | 5.4 : 16.5      | 9.8 : 6.1
#   nearest-ring gap p10/p50  4.0 / 14.2 px              | 4.2 / 17.9 | 7.2 / 14.6      | 2.5 / 31.5
# Known residuals: ring contrast reads higher than real (narrower, better separated rings take
# rarer, higher ranks under histogram equalisation) -- shown NOT to matter by ring_contrast_scan;
# and fine-scale (1-2 px) ring grain stays below real's, because realbkg peaks sit ~70x higher
# above noise than real ones -- legacy shows this is not needed to handle 41.
#
# The ring-count draw stays a WIDE per-frame distribution (13% ring-free, 14% 1-5 rings) rather
# than a compromise value, so organic's few-ring frames (38% ring-free, median 1) stay covered.
realbkg_p_ring           = 0.75
realbkg_n_powder         = (1, 3)
realbkg_rings_cap        = (2, 60)
realbkg_ring_sq_scale    = 0.7     # ring sigma_q x0.7: legacy-like visible width per box
realbkg_ring_w_coef      = 3.0     # ring box radial extent 3.0*sigma_q (segments keep 1.3)
realbkg_seg_on_ring_keep = 0.12    # keep 12% of segments whose box overlaps a ring radially
realbkg_ring_min_gap     = 8.0     # px between ring boxes; organic CIFs otherwise pack rings edge to edge
realbkg_ring_chi_texture = 0.45    # per-ring sd ~ U(0, 0.45) of a mean-one log-normal factor along chi
realbkg_ring_chi_corr    = (1.0, 64.0)   # correlation length log-uniform in px -> broadband texture
# segment supply restored after seg_on_ring_keep removes the ones on rings (24 -> 42.7/frame)
realbkg_n_oriented       = (1, 4)
realbkg_spots_cap        = (2, 100)
