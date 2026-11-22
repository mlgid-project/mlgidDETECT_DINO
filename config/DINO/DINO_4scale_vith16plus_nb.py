_base_ = ['DINO_4scale_swin_ssl_nb.py']

# DINO detector (4 scale, the nb recipe of DINO_4scale_swin_ssl_nb.py) with a DINOv3 ViT-H+/16 backbone instead of the SimMIM swin-L (2026-11-22, flagship run).
# backbone_weights = public DINOv3 ViT-H+ (override on the command line with backbone_weights=<backbone_teacher.pth> for an own pretraining).
# The swin settings (window / patch size, backbone_dir) are not used by the ViT branch. Backbone lr 1e-5 like the heatmap's unfrozen ViT-H+ rows (4e-5 for the head and decoder);
# gradient checkpointing per block (use_checkpoint).
backbone = 'dinov3_vith16plus'
backbone_weights = '/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
lr = 4e-05
lr_backbone = 1e-05
use_checkpoint = True
