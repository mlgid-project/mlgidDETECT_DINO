# Co-DINO Phase-0 MVP (docs/CO_DINO_INVESTIGATION.md): ONE collaborative auxiliary head
# with FCOS center-sampling one-to-many assignment, attached to the encoder feature
# pyramid during TRAINING to densify positive supervision for hard/faint peaks. The head
# is discarded at inference -> zero deploy cost, ONNX graph byte-identical to baseline
# (models/dino/co_heads.py; guarded by self.training + the export whitelist).
#
# THESIS: our ceiling is low SENSITIVITY (recall) to faint (vis=1 ~0.33) and high-q
# (q682-1024 ~0.44) peaks -- an encoder-representation problem. A one-to-many aux head
# gives every location near a GT peak a DIRECT cls+box gradient, which is the mechanism
# for faint-peak sensitivity (unlike CCTM's detail-refinement, declined).
#
# Protocol: fine-tune from ssl1 (run_detector_codino.sbatch), A/B organic + 41 AP vs
# ssl1 (organic 0.586 / 41 0.762). GATE = organic AP AND the faint/high-q recall probe
# (diag_compare.py) -- AP-up-but-recall-flat = "refining easy peaks" (CCTM-null shape) =>
# decline. FCOS assigner (not ATSS): our elongated arcs would starve ATSS anchor-IoU.
_base_ = ['DINO_4scale_swin.py']

use_co_heads     = True
co_num_heads     = 1        # K=1 MVP (paper K=1 ~ 2/3 of K=2 gain); single FCOS head
co_assigner      = 'fcos'   # center-sampling, robust to extreme-aspect boxes
co_loss_coef     = 2.0      # paper lambda_2 (encoder-supervision weight)
co_center_radius = 1.5      # FCOS center-sampling radius (x per-level stride)
lr_cohead_mult   = 10.0     # fresh heads on a converged warm-start (mirrors lr_cctm_mult)
# use_cctm / use_boost_loss stay off (their base defaults) -- orthogonal, combinable later.
