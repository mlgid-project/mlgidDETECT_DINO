# FROM-SCRATCH 5-scale control for the DINO-5 experiment (RECIPE_v3 Idea 6).
#
# Same as the from-scratch baseline (DINO_4scale_swin.py — NO backbone_dir, random init) EXCEPT
# 5 feature levels instead of 4. This is the control for DINO_5scale_swin_ssl.py: comparing the two
# answers "how much of any 5-scale gain is the SSL backbone vs just having a 5th scale" (5-scale
# helps non-SSL models too). Keeps everything else identical (classes, schedule, window, batch).
_base_ = ['DINO_4scale_swin.py']

return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 5
