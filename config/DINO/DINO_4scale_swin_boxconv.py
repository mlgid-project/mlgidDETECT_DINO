# Box CONVENTION — FROM-SCRATCH run (MODIFICATIONS.md phases AC.4 / AC.5 / AE).
#
# The single variable vs ssl1 is what a ground-truth box MEANS: how many sigma out its edge sits.
#     simulation.py:510-513   box = pos +- widths*w_coef ,  a_pos +- a_widths*a_coef
#     simulation.py:842-843   sigma_q = box_w / w_coef  ,   sigma_chi = box_h / a_coef
# At the defaults a_coef 3.5 / w_coef 1.0 a simulated box is +-1.75 sigma in chi and +-0.5 sigma in
# q. Because the coefficients cancel between construction and rendering, changing them relabels the
# same image rather than changing the physics (pre-flight below bounds the residual).
#
# WHY THESE VALUES. AE (job 2796061) swept a 7x8 grid of chi x q rescales applied to ssl1's
# predicted boxes about their own centres, deployed evaluation at every node. The sum of `ap_total`
# over both gates ridges at chi 0.80-0.85 x q 1.20-1.50; the raw argmax is chi 0.80 / q 1.50
# (+0.0195) but chi 0.80 / q 1.30 (+0.0190) is statistically tied and is at or beside the maximum on
# BOTH gates read separately -- organic 0.5812 (grid max 0.5831) and 41 0.7502 (grid max 0.7503):
#     a_coef = 3.5 * 0.80 = 2.80      w_coef = 1.0 * 1.30 = 1.30
# The q direction is the robust half: q 0.85 is negative in 6 of 7 chi rows while every row improves
# from 0.85 toward ~1.3, and this agrees with an INDEPENDENT measurement -- box_w / FWHM_q is 0.65 on
# organic and 0.63 on 41 (the two real gates AGREE) against 0.39 in the simulator, a 1.6x deficit.
# In chi the two real gates DISAGREE (box_h/FWHM_chi 0.73 organic vs 1.16 on 41, the simulator on top
# of 41 at 1.10), so no coefficient can satisfy both and 0.80 is the compromise the grid prefers.
#
# HONEST CAVEATS, both pre-registered:
#  * The grid prices a global rescale of a model TRAINED at k_chi 1.75 / k_q 0.50. A model trained at
#    2.80 / 1.30 learns the size per peak instead of taking a uniform squeeze, so +0.0190 is a guide
#    to WHERE to put the coefficients, not a prediction of the retrain's score.
#  * The plateau within 1 jackknife SE (0.0129) covers 30 of 56 nodes and the argmax is only +1.51
#    sigma (organic +1.27, 41 +0.86). No single node is significant on its own. What is not noise is
#    the SHAPE: both gates independently prefer chi < 1 and q > 1.
#  * AC.4 separately showed that matching organic's measured convention EXACTLY (a_coef 1.85) COSTS
#    -0.0118 organic / -0.0100 on 41 -- the evaluation's IoU floor of 0.1 does not reward tightness.
#    So this run targets the AP optimum, not convention-matching, and lands only partway toward the
#    real labels on both axes.
#
# PRE-FLIGHT (job 2796084 at 2.98/1.30, re-run 2796088 at 2.80/1.30): the change is a relabelling to
# within a small residual. Three filters read the BOX rather than the widths -- detector-gap
# rejection, the 1.6 px minimum extent in filter_dark_area, and clamp_boxes feeding sigma back into
# img_from_labels -- so the surviving population shifts slightly: 84% of frames keep an IDENTICAL
# object count at 2.80/1.30, segments/frame moves 29.72 -> 29.53 (-0.65%), ring:segment 0.5232 ->
# 0.5253 (+0.4%), and the rendered image differs by mean |dI|/std(I) = 0.00031 (median 0.00002).
# Measured box ratios land on 1.3000 in q at every percentile and 0.802-0.813 in chi against the
# exact 0.800, the ~1% excess being the smallest boxes filtered out by the 1.6 px floor. That
# residual is reported WITH the run, not hidden: this is a relabelling plus a 0.65% segment loss.
#
# NOT changed here: seg_q_elongated_frac stays at its 0.0 default. `dino_qaspect1` (job 2782747) put
# it at 0.35 and was negative on both gates -- last 5 evals organic -0.0045, 41 -0.0094 -- while
# moving organic pred_h/gt_h only 2.55 -> 2.18. Peak SHAPE and box CONVENTION are separate variables
# and this run changes the convention only.
#
# From-scratch = SSL-pretrained backbone + random detector head + ssl1's exact recipe, per the user's
# stated preference for from-scratch verdicts. Matched control = ssl1 (job 2755004):
# organic ap_total 0.5683 / 41 0.7441; best-epoch organic 0.5860 @238 / 41 0.7620 @258;
# <5 px chi-gap recall 0.352 / 0.449; organic pred_h/gt_h p50 2.55, 41 1.08.
#
# GATE (pre-registered): PRIMARY = organic and 41 ap_total must not regress vs ssl1 at matched
# epochs, both reported. SECONDARY = organic pred_h/gt_h p50 on segments moves below 2.55 while 41
# stays near 1.08 (diagnostics/box_size_probe.py block 3), and <5 px chi-gap recall rises from
# 0.352 / 0.449.
_base_ = ['DINO_4scale_swin_ssl.py']     # SSL backbone init + ssl1 schedule/LR (from-scratch)

box_coef_override = (2.80, 1.30)         # (a_coef, w_coef) — the single variable under test
