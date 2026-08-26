# Segment aspect ratio — FROM-SCRATCH run (MODIFICATIONS.md phases AB / AC / AC.1 / AC.3).
#
# The single variable vs ssl1 is the SHAPE of the simulated segments. `simulation.py`
# simulate_labels forced, for every segment,
#     a_widths = maximum(a_widths, widths * U(1,2))       i.e.  sigma_chi >= sigma_q * U(1,2)
# so a simulated segment could NEVER be wider in q than tall in chi. Measured consequence
# (AC.1, jobs 2782668 / 2782687), segment box aspect box_h/box_w:
#     organic labels   p50 0.68,  64.5% wider than tall,  90.5% below the simulator's floor of 3.5
#     41 labels        p50 7.84,   1.1% wider than tall,  19.8% below it
#     simulator        p50 9.23,   0.4% wider than tall,   2.7% below it
# and the fitted physical peaks (AC.2, job 2782724) say the same about the IMAGE, not just the box:
#     sigma_chi / sigma_q   organic 0.67   41 3.13   simulator 3.18
# The simulator reproduces 41's peak shape to within 2% and misses organic's by 4.7x. It is a
# perovskite simulator, which is why 41 has always been the stronger gate.
#
# Downstream effect it is meant to fix (AB, job 2781962): the model draws boxes 2.55x too tall on
# organic segments FOR THE DETECTIONS THAT COUNT AS CORRECT (41: 1.08, rings on both gates: ~1.0).
# The deployed matcher's IoU floor of 0.1 hides that in recall and precision. AC.4 (job 2782735)
# then showed the <5 px chi-gap recall bucket swings 0.218 -> 0.394 with predicted box height while
# the >=33 px bucket moves 0.023 -- an 8x difference in sensitivity, i.e. tall boxes merge close
# pairs and duplicate suppression deletes one of them.
#
# NOT changed here: a_coef / w_coef. AC.4 priced a global chi rescale of the predicted boxes and the
# gate is indifferent across 0.70-1.20x (organic moves 0.006, 41 moves 0.006), while matching
# organic's own measured convention exactly (k_chi 0.92, a_coef 1.85) COSTS -0.0118 organic and
# -0.0100 on 41. So the boxing convention is a separate, priced-at-zero knob and is left alone; this
# run changes peak SHAPE only, one variable.
#
# From-scratch = SSL-pretrained backbone + random detector head + ssl1's exact recipe, per the
# user's stated preference for from-scratch verdicts. Matched control = ssl1 (job 2755004):
# organic ap_total 0.5683 / 41 0.7441; <5 px recall 0.352 / 0.449; organic pred_h/gt_h 2.55.
#
# GATE (pre-registered): PRIMARY = organic and 41 ap_total must not regress (0.5683 / 0.7441).
# SECONDARY = organic pred_h/gt_h p50 on segments moves 2.55 toward 1.0 while 41 stays near 1.08
# (diagnostics/box_size_probe.py block 3), and <5 px chi-gap recall rises from 0.352 / 0.449.
_base_ = ['DINO_4scale_swin_ssl.py']     # SSL backbone init + ssl1 schedule/LR (from-scratch)

seg_q_elongated_frac = 0.35              # the single variable under test
seg_q_aspect_range = (0.15, 1.2)         # log-uniform sigma_chi/sigma_q for those segments
