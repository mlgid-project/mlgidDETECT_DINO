# Stage-freezing experiment (RECIPE_v3 Idea 5b) on the ROUND-1 SSL backbone.
#
# Identical to DINO_4scale_swin_ssl.py (same simmim1 backbone_dir, same everything) EXCEPT it
# FREEZES the early backbone stages so the SSL real-data features there can't be overwritten by
# the simulated detector training. The A/B is therefore dino_ssl1 (no freeze) vs dino_ssl1_frozen
# (early stages frozen) on the SAME backbone -> isolates exactly the freezing variable.
#
# What gets frozen (backbone.py:185 substring-matches param names of the swin):
#   'patch_embed'  -> the 4x4 patch projection
#   'layers.0.'    -> stage 1 (stride 4, generic edges/texture/noise — most transferable)
#   'layers.1.'    -> stage 2 (stride 8)
# Stages 3-4 (layers.2 = 18 blocks, layers.3) + the entire DINO transformer/heads stay trainable.
# Frozen params are dropped from the optimizer (requires_grad=False) -> also faster / less memory.
_base_ = ['DINO_4scale_swin_ssl.py']

backbone_freeze_keywords = ['patch_embed', 'layers.0.', 'layers.1.']
