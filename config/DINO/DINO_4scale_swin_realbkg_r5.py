_base_ = ['DINO_4scale_swin_realbkg_r3.py']

# RUN 5. ONE VARIABLE against run 3: half the images come from the LEGACY simulator.
# dn_number stays at 100 (run 4 tests dn400 separately), so run 3 vs run 5 is a clean read.
#
# WHY. Measured 2026-10-05 across four realbkg runs and four controls: every realbkg run DECAYS
# with training and every synthetic-background run IMPROVES.
#     organic, best early window -> post-drop plateau
#     conv1 0.578 -> 0.518   conv2 0.552 -> 0.517   conv3 0.594 -> 0.496   conv4 0.606 -> (0.561)
#     lr4e5_1 0.525 -> 0.608  boxconv1 0.480 -> 0.585  ssl1 0.510 -> 0.562  physics4_1 -> 0.556
# It is overfitting, not instability: conv3's TRAIN loss falls the whole way (giou 0.548 -> 0.345,
# bbox 0.0837 -> 0.0541) while its organic AP falls 0.594 -> 0.496. lr4e5_1's loss falls AND its AP
# rises.
#
# NOT THE BANK: physics4_1 uses the same physics CIF bank on SYNTHETIC backgrounds and improves.
# NOT BROKEN RANDOMISATION, checked explicitly (diagnostics/image_freshness.py): replicating
# main.py's per-epoch reseed gives 160/160 distinct frames over four epochs with ZERO identical
# frames between any pair, so the dino_physics3_1 resume bug is genuinely fixed.
#
# IT IS THE BACKGROUNDS. The realbkg sim holds a 48-slot pool of pre-built mosaics drawn from only
# 90 donor frames, and one slot is replaced per 64 images -- so at 1000 images/epoch every
# background is reused about 21 times per epoch, and every mosaic the run ever sees comes from
# those same 90 frames. The legacy sim draws fresh perlin noise per frame and reuses nothing.
#
# WHAT THIS BUYS, on two axes at once:
#   1. unlimited background diversity for half the stream, to break the memorisation
#   2. the legacy sim's 41 strength. It scores 0.74-0.77 on 41 where every realbkg run sits at
#      0.38-0.43, and its composition is 41-like (42.6 boxes/frame, ring:segment 0.685, 17
#      rings/frame) against realbkg's organic-like frames.
# The two simulators fail in OPPOSITE directions, which is why the 2026-09-15 precision analysis
# queued the mixture as the lever; it has been unrunnable until the realbkg_fraction guard.
#
# Box convention is shared: main.py applies box_coef_override to _sim_config BEFORE building
# FastSimulation, so both halves label at a_coef=2.8 / w_coef=1.3 and the labels do not disagree.
#
# NOTE this is the first run to exercise the guard, so the epoch-0 share line is the thing to check.
realbkg_fraction = 0.5
