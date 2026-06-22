# Round-2 detector config: IDENTICAL to DINO_4scale_swin_ssl.py (round-1 SSL detector),
# except backbone_dir points at the simmim2 (RECIPE_v2) backbone export instead of simmim1.
#
# Kept deliberately identical otherwise (same _base_, classes, schedule, eval_files, NO
# stage-freezing, NO 5-scale) so the comparison dino_ssl1 vs dino_ssl2 isolates exactly ONE
# variable: the SSL pretraining recipe (v1 vs v2). Freezing / 5-scale are separate experiments
# (RECIPE_v3 ideas 5b / 6) to run AFTER this A/B, each as its own isolated change.
#
# Mechanism (same as round 1): backbone_dir makes models/dino/backbone.py load
#   <backbone_dir>/swin_large_patch4_window12_384_22k.pth  into the swin backbone
#   via load_state_dict(strict=False). Place export_backbone.py output there before training.
_base_ = ['DINO_4scale_swin.py']

backbone_dir = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim2/backbone_export'
