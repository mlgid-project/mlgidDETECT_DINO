_base_ = ['DINO_4scale_swin_realbkg.py']

# RUN 3. Same bank, same labelling convention, two measured GEOMETRY defects fixed and a frame
# composition that spans BOTH gates instead of targeting one.
#
# WHY runs 1 and 2 lost (organic 0.518/0.517, 41 0.394/0.428 against lr4e5_1's 0.608/0.761):
# measured, not guessed.
#   * NOT the labelling convention. Paired on identical frames (diagnostics/convention_ab.py),
#     unified vs historical labels differ by 4.3% of boxes on run 1 and 0.5% on run 2, and the
#     brightness gate runs in BOTH paths.
#   * NOT frame composition. Run 2 matched 41 within 10% on rings/frame, segments/frame, ratio and
#     ring-free fraction, and gained 0.034 of a 0.333 deficit.
#   * NOT amplitude_mode='pygid'. Raw peak contrast is 222x local noise against real's 3.1x, but
#     CHAIN ends in histogram equalisation and AFTER preprocessing the sim sits at p50 1.10
#     against real organic 1.12 and real 41 1.11. The raw gap does not reach the network.
#   * IT WAS THE BOX GEOMETRY, on both gates, in two different ways:

# FIX 1, the 41 gate. Every ring box used to be the full 512 px frame height. Real 41 ring boxes
# track the valid chi span at their q instead -- height/span p10 0.90, p50 0.98, p90 1.03 -- and
# only 0.6% are full height, because the detector wedge leaves most columns short of 512 rows. A
# full-height prediction cannot be matched at IoU 0.5 against 24.7% of 41's rings, which is 10.3%
# of every box on that gate (14.8% at IoU 0.75). The legacy simulator behind 0.748 on 41 did NOT
# do this: p10 80 / p50 334 / p90 511, 27.3% full height. Real organic is 80% full height, and
# since its annotators overshoot the span (ratio 1.03-1.14) the span rule suits it too.
realbkg_ring_box_from_mask = True

# FIX 2, the organic gate. Segment shape is bimodal ACROSS the two gates and the fitted width
# lognormal only reproduces one mode:
#     organic segments  sigma_q 8.1  sigma_chi  2.9   aspect 0.36   (wide and short)
#     41      segments  sigma_q 3.5  sigma_chi 12.2   aspect 3.45   (narrow and tall)
#     emitted           sigma_q 3.8  sigma_chi 19.5   aspect 5.1
# So the simulator is close to 41 and 14x too elongated for organic: its segment box height p10 of
# 11.95 px sits ABOVE organic's median of 8.1, i.e. over 90% of simulated segments are taller than
# a typical real organic peak and that gate never sees its own peak shape. Half the frames now use
# the organic-like mode. Chosen per FRAME, not per peak: no real frame mixes the two.
realbkg_seg_wide_frac  = 0.50
realbkg_seg_wide_sigma = ((8.1, 0.45), (2.9, 0.45))

# COMPOSITION spanning both gates rather than either one. Run 1 sat at organic's end (93.7
# boxes/frame, 1.6 rings, 73% ring-free), run 2 at 41's (25.8 boxes, 9.2 rings, 0% ring-free), and
# both lost -- with the geometry broken, neither composition could be read cleanly. Real: organic
# p50 66 boxes / 2.12 rings / 62% ring-free, 41 p50 20 boxes / 8.85 rings / 0% ring-free.
realbkg_spots_cap = (2, 70)
realbkg_rings_cap = (1, 13)
realbkg_p_ring    = 0.60
realbkg_n_powder  = (1, 2)
realbkg_n_oriented = (1, 3)
