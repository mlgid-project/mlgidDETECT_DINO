# Co-DINO FROM-SCRATCH co-train (docs/CO_DINO_INVESTIGATION.md): the FAITHFUL test of
# the collaborative aux head. Unlike the warm-start variant (DINO_4scale_swin_codino.py),
# this reproduces ssl1's exact training recipe (SSL-pretrained backbone via backbone_dir,
# random detector head, uniform LR) but with the aux head PRESENT FROM EPOCH 0, so the
# encoder co-adapts to the dense one-to-many supervision from the start -- the regime the
# Co-DETR paper uses, and the one a warm-start under-tests (an architectural/training-scheme
# change from a converged encoder can't fully reorganize).
#
# Apples-to-apples baseline = ssl1 (organic 0.586 / 41 0.762), which is THIS recipe minus
# the aux head. GATE = organic AP AND the faint/high-q recall probe (diag_compare.py).
#
# NOTE: no lr_cohead_mult here -- the head is co-trained from init at the body LR, not
# grafted onto a converged model, so the 10x graft trick (warm-start variant) does not apply.
_base_ = ['DINO_4scale_swin_ssl.py']       # SSL backbone init (backbone_dir) + ssl1 schedule/LR

use_co_heads     = True
co_num_heads     = 1        # K=1 MVP (single FCOS center-sampling head)
co_assigner      = 'fcos'
co_loss_coef     = 2.0      # paper lambda_2 (encoder-supervision weight)
co_center_radius = 1.5
