_base_ = ['DINO_4scale_swin_realbkg_r7.py']

# RUN 8. Against run 7 ONE knob: realbkg_seg_on_ring_keep 0.12 -> 0.5. Everything else is r7.
#
# WHY (seg_context_probe.py, 2026-10-08; run 7 ep73 on real 41, median TP score of matched segments):
#   * segments that sit on a ring score 0.36 vs 0.58 isolated; run 6 0.78 / 0.77, legacy 0.80 / 0.89
#   * the same weakness shows on run 7's OWN sim (on ring 0.45 vs isolated 0.89, run 6 on the same
#     keep-1.0 frames 0.85) -- so the model has learned segments on rings are unlikely, not that
#     41 looks foreign
#   * real 41: 23.3% of GT segments overlap a ring box radially. r7 sim at keep 0.12: 10.7%, at 0.3
#     16.4%, at 0.5 21.7%, at 1.0 37.1%
#   * NOT the ring texture: switching it off leaves on-ring scores unchanged (0.45 vs 0.45)
#   * the same run 7 checkpoint scores on-ring segments higher on frames with more of them:
#     0.45 (keep 0.12) -> 0.52 (0.3) -> 0.57 (0.5)
# Known cost: r7 had 4.4 on-ring segments/frame vs 41's 5.6; keep 0.5 gives ~10.9 (isolated
# segments are 37/frame vs 41's 18.4, so the SHARE was low, not the count). keep 0.3 (7.5/frame,
# share 16%) is the milder alternative.
realbkg_seg_on_ring_keep = 0.5
