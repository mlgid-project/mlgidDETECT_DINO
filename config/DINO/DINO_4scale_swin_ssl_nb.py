_base_ = ['DINO_4scale_swin_ssl.py']

# THE RECIPE (best DINO setting so far; trained on the simulator only, scored on the two labelled sets):
#   python main.py -c config/DINO/DINO_4scale_swin_ssl_nb.py --output_dir runs/dino_nb
# Everything not set here comes from DINO_4scale_swin.py: lr = lr_backbone = 4e-5, batch 2, 2 classes (segment, ring), dn_number 100.
#
# Backbone: Swin-L pretrained with SimMIM on real frames. Put the file at weights/swin_large_patch4_window12_384_22k.pth
# (download link in the README); backbone_dir below points there.
# Validation sets: the re-fitted ("visfix") labels, one folder shared with the heatmap detector: data/41.h5 and data/organic_labeled.h5
# (stored radius_width / angle_width = 2 sigma, rings carry angle_width = inf, which util/pygidloader.py clamps to the full chi range).
# APs from this config are NOT comparable with runs on the older validation files.
backbone_dir = 'weights'
eval_file = 'data/41.h5'
eval_files = {
    '41': 'data/41.h5',
    'organic': 'data/organic_labeled.h5',
}

# Training boxes = centre +- 1.5 sigma (main.py turns this into the simulator's a_coef = w_coef = 3.0), the same convention as
# the heatmap detector (--box_coef 3.0,3.0). The validation boxes are +-1 sigma, so the final scoring scales them:
#   HM_GT_BOX_COEF=3.0,3.0 python heatmap/evaluate.py dino=dino:<run>/checkpoint.pth
# (the AP printed during DINO training compares against the unscaled boxes and is not the number to report).
# Scaled-GT AP (organic / 41): 0.587 / 0.722 after 380 epochs; the same recipe at +-1 sigma gave 0.605 / 0.722.
box_sigma_halfwidth = (1.5, 1.5)
box_coef_override = None
epochs = 380
lr_drop = 280
