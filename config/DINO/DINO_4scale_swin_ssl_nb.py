_base_ = ['DINO_4scale_swin_ssl.py']

# NEW-BOX BASELINE (2026-10-21): the dino_lr4e5_1 recipe (legacy sim, SSL backbone, lr set on the command line
# as lr=4e-5 lr_backbone=4e-5, batch 2, 500 epochs, lr drop 280) with the RE-FITTED validation sets and the
# matching box convention. APs from this config are NOT comparable with any earlier run.
#
# Validation files: stored radius_width / angle_width = 2 sigma, box = centre +- width/2 = +- sigma; visibility
# copied back from the old files by peak id (organic_labeled / 41_pygid); rings carry angle_width = inf, clamped
# to the full chi range by util/pygidloader.py. Both are PyGID files, so 41 now loads through PyGIDDataset
# (images/masks verified identical to the old 41.h5 path, diagnostics/newbox_check.py). Boxes that stick out of
# the image are intentional (peak centre at the border) and are NOT clipped.
eval_file = '/mnt/lustre/work/schreiber/szb389/datasets/newbox/perovskites_nan_wedge_visfix.h5'
eval_files = {
    '41': '/mnt/lustre/work/schreiber/szb389/datasets/newbox/perovskites_nan_wedge_visfix.h5',
    'organic': '/mnt/lustre/work/schreiber/szb389/datasets/newbox/organics_visfix.h5',
}

# training boxes = centre +- 1 sigma (main.py turns this into the simulators' a_coef = w_coef = 2.0)
box_sigma_halfwidth = (1.0, 1.0)
box_coef_override = None
