# DINO-5scale on the ROUND-1 SSL backbone (RECIPE_v3 Idea 6).
#
# Identical to DINO_4scale_swin_ssl.py (same simmim1 backbone_dir, classes, schedule) EXCEPT it
# feeds the backbone's SHARPEST level (stage 1, stride 4) to the detector instead of discarding it:
#   return_interm_indices = [0,1,2,3]   -> backbone returns stages 1,2,3,4 (strides 4/8/16/32)
#   num_feature_levels     = 5          -> those 4 + one extra stride-64 level = 5 scales
# backbone.py allows [0,1,2,3] (assert), and bb_num_channels becomes [192,384,768,1536].
#
# SSL tie-in: stage 1 IS SimMIM-pretrained (layers.0 weights are in the export); only the per-level
# output norm for index 0 ('norm0') is absent from the export -> it loads random via strict=False
# (harmless). So this plugs the detector into the pretrained-but-currently-unused sharp features.
#
# COST: ~4x encoder tokens (~10.9k -> ~44k) -> ~2-3x train time/memory. Run with --amp; if it OOMs
# at batch 2 on a 40GB A100, set batch_size=1 here. Multi-GPU is NOT wired in this repo
# (init_distributed_mode is commented out, model not DDP-wrapped) -> single-GPU only for now.
# For a clean read, also run a FROM-SCRATCH 5-scale control (5-scale helps non-SSL models too).
_base_ = ['DINO_4scale_swin_ssl.py']

return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 5
