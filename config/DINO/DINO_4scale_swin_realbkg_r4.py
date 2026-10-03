_base_ = ['DINO_4scale_swin_realbkg_r3.py']

# RUN 4. ONE VARIABLE against run 3: dn_number 100 -> 400. Everything else -- bank, labelling
# convention, both geometry fixes, composition, lr, batch, backbone -- is inherited unchanged, so
# conv3 vs conv4 is a clean read on denoising supervision.
#
# WHY. The 2026-09-15 precision analysis found the physics-sim 41 gap is a CONFIDENCE CALIBRATION
# failure, not blindness: on 41 physics4 loses no recall against the baseline (0.856 vs 0.888) and
# its precision halves (0.256 vs 0.445), with score separation TP-FP at 0.399 against 0.650. 74%
# of its false positives are NEAR-MISSES on its own simulated distribution, so this is not domain
# transfer and not a data bug. That analysis named dn400 as the right lever and it was never run:
# dn_number = 100 in every run in this project's history, without exception.
#
# AND THE LEVER IS BIGGER THAN IT LOOKS. dn_components.py:42-43 turns the config value into GROUPS:
#     groups = (2 * dn_number) // (2 * max_boxes_in_batch)
# so the denoising supervision collapses as frames get denser:
#     max boxes    7    13    20    44    60    93   120   200
#     dn=100      14     7     5     2     1     1     1     1
#     dn=400      57    30    20     9     6     4     3     2
# COCO images carry ~7 boxes and get ~14 groups, which is what DINO was tuned at. Our frames carry
# p10 13 / p50 44 / p90 93, and at batch 2 the max-of-two is typically 50-95 -- so this project has
# been training at ONE OR TWO denoising groups throughout. Denoising has been nearly inert here,
# which is exactly the mechanism (box refinement and duplicate rejection) that the measured
# near-miss hedging needs.
dn_number = 400

# COST, so the run is read correctly. dn adds max_boxes*2*groups decoder queries, i.e. ~720-800 at
# dn=400 against ~100-200 at dn=100, on top of 900 object queries. Decoder self-attention is
# quadratic in sequence length, so expect roughly 1.3-1.5x the epoch time of run 3 (which runs
# ~10 min/epoch). A 72 h wall then reaches only ~290-330 epochs against run 3's ~430. lr_drop is
# at 280, so the post-drop window this arm must be judged in will be SHORT -- plan on resuming
# from checkpoint.pth for a second allocation rather than reading a 10-eval plateau.
