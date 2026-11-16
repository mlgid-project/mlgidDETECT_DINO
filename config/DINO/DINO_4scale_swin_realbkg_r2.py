# Run 2 of the real-background sim: SAME convention, SAME bank, composition aimed at 41.
#
# WHY. dino_realbkg_conv1 (run 1) is the best organic number we have and the worst 41 number:
# 0.558 / 0.395 over epochs 140-166, against lr4e5_1's 0.553 / 0.743. Measured over 120 frames it
# makes 1.57 rings and 94.6 segments per frame, ring:segment 0.017 -- against real organic's
# 0.033, physics3_2's 0.118 and real 41's 0.536. It did not land on organic's composition, it
# went 2x PAST it. 41 is 41% rings by object count and rings are the EASY class there (recall
# 0.856 vs 0.713 for segments), so starving them costs that gate disproportionately.
#
# Two causes, and the larger was not the ring rate:
#   * segments: spots_cap (2, 200) averages 101 reflections per oriented entry against the old
#     constant's 34, so ~200 are drawn per frame and 94.6 survive the gate.
#   * rings: p_ring 0.30 draws 2.4/frame; the IoU 0.10 suppression, which under the unified
#     convention removes the PEAK rather than just its box, takes that to 1.57.
#
# THIS RUN AIMS AT 41, not at a compromise. A middle value matches neither gate -- that is the
# standing lesson from the ring-rate work -- so run 2 goes to the far end and run 1 holds the
# organic end. Between them the two bracket the axis, which a single blended run never would.
#
# TARGET, from the LABELS: 41 has 8.85 rings and 16.5 segments per frame, ratio 0.536, and NO
# ring-free frames at all. Hence p_ring 1.0: every frame gets rings, which run 1 never did for
# 73% of its frames.
#
# MEASURED over 120 frames (diagnostics/realbkg_dynrange.py), after one refinement pass:
#
#                rings/fr   segs/fr   ratio   ring-free   boxes p50 / max
#     run 2         9.28      18.3    0.507       0%          26 / 66
#     real 41       8.85      16.5    0.536       0%          20 / 65
#     run 1         1.57      94.6    0.017      73%          99 / 190
#
# Every axis within about 10% of 41. The first pass, at spots (5,25) / rings (3,15), gave 9.76
# rings and 13.5 segments -- ratio 0.72, slightly ring-heavy -- so spots widened to (6,32) and
# rings trimmed to (3,13).
#
# NOT matched: corr(rings, segments) is +0.10 here against 41's -0.25. Rings and segments are
# drawn independently, so the anti-correlation real frames show is absent. Fixing it needs the
# powder draw coupled to the oriented draw, which is a mechanism rather than a parameter.
#
# ONE AXIS ONLY. Bank, render-iff-labelled, 2.0 x local noise, SNR 6.0, IoU 0.30/0.10, the
# 200-peak cap, 1 channel, lr 4e-5, batch 2 are all inherited unchanged, so run 1 vs run 2 is a
# clean comparison of frame composition and nothing else.

_base_ = ['DINO_4scale_swin_realbkg.py']

# SEGMENTS down to 41's level. 1-2 entries x mean 15 reflections = ~22 drawn, against run 1's
# ~200. 41 carries 16.5 segments/frame; organic carries 63.6, so this deliberately starves the
# organic end.
realbkg_n_oriented = (1, 2)
realbkg_spots_cap  = (6, 32)

# RINGS up to 41's level. Every frame gets them (41 has no ring-free frame), 1-2 powder entries
# of 3-15 rings each = ~13.5 drawn, which after the IoU 0.10 suppression should land near 41's
# 8.85. Lower bound back to 3, not run 1's 1: the 1-2 ring bucket is an ORGANIC feature (12% of
# its frames) and 41 sits at 7%.
realbkg_p_ring    = 1.0
realbkg_n_powder  = (1, 2)
realbkg_rings_cap = (3, 13)
