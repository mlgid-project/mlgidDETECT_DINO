# Detector config: identical to the 2-class ring/segment baseline (DINO_4scale_swin.py),
# except the swin-L 48x6 backbone is INITIALIZED from the SimMIM-pretrained weights
# instead of random. Everything else (classes, eval_files, schedule) is inherited.
#
# Mechanism: setting `backbone_dir` makes models/dino/backbone.py load
#   <backbone_dir>/swin_large_patch4_window12_384_22k.pth   (the PTDICT name for swin_L_384_22k)
# into the swin backbone via load_state_dict(strict=False) — it keeps all encoder keys
# (drops only 'head'; dilation=False so layers.3 is kept too). Place the export_backbone.py
# output at exactly that path/name before training.
_base_ = ['DINO_4scale_swin.py']

backbone_dir = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export'
