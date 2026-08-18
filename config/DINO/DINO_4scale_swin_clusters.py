# Azimuthal peak clusters — FROM-SCRATCH run (MODIFICATIONS.md phase U).
#
# The single variable vs ssl1 is the SYNTHETIC LABEL GEOMETRY: several peaks sharing one q (one
# ring) at different chi, with the chi-gap distribution the real data actually has. Everything else
# — model, recipe, polar grid 512x1024, preprocessing, postprocessing, ONNX — is untouched, so this
# is a drop-in for the deployed 512x1024 inference path (unlike phase R).
#
# WHY (probes S/T, jobs 2754438 / 2754964):
#   - the recall gap is NOT contrast-limited (prominence separation AUC 0.489 on organic);
#   - 84.5% of missed peaks sit within 8 q-px of a peak the model DID detect, median chi-gap 3.9 px
#     against ~8.5 px-tall boxes -- a peak-SEPARATION failure along chi;
#   - the stock simulator cannot produce that configuration: add_peaks_on_rings spaces peaks >=33 px
#     apart on a fixed grid, fires on 10% of images, caps at 4 per ring, and needs a ring >=200 px
#     tall. Real data: 85.9% of organic peaks have a same-q sibling, 12.5% of gaps are under 5 px.
#   - NMS is NOT the bottleneck (job 2755004): disabling it entirely moves tight-pair recall by
#     +0.006 while costing 0.027-0.054 ap_total, so postprocessing stays at the deployed settings.
#
# From-scratch = SSL-pretrained backbone + random detector head + ssl1's exact recipe, per the
# user's stated preference for from-scratch verdicts. Matched control = ssl1 at these same settings
# (measured 2026-08-18, job 2755004: organic ap_total 0.5683 / 41 0.7441; tight-pair (<5 px) recall
# 0.352 / 0.449).
#
# GATE (pre-registered, MODIFICATIONS.md U): PRIMARY = recall on same-q sibling peaks by chi-gap
# bucket at a MATCHED operating point; SECONDARY = organic/41 ap_total must not regress.
# 41 is the uncontaminated gate: the chi-gap prior below is parametric with constants rounded from
# ORGANIC only, so no 41 label statistics enter the training data.
_base_ = ['DINO_4scale_swin_ssl.py']     # SSL backbone init + ssl1 schedule/LR (from-scratch)

use_peak_clusters = True                 # the single variable under test
cluster_extra_ratio = 0.6                # extra sibling peaks as a fraction of the segment count
cluster_tight_frac = 0.32                # share of chi-gaps from the tight component
cluster_tight_gap_px = (1.0, 6.0)
cluster_broad_lognorm = (4.007, 0.95)    # ln(55) median, sigma
cluster_size_p = (0.358, 0.265, 0.178, 0.069, 0.050, 0.034, 0.016, 0.031)
