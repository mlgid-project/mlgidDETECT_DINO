# Exp B of the Cross-DINO portable subset (docs/CROSS_DINO_INVESTIGATION.md S4b/S7):
# CCTM (Cross Coding Twice Module) inserted at the encoder->decoder boundary in
# models/dino/deformable_transformer.py (after `memory`, before query selection).
# It reinjects the pre-encoder (input-projected backbone) feature into the encoder
# memory via two rounds of elementwise sigmoid-gated fusion, giving the decoder a
# finer "cross feature".
#
# ON the exported ONNX path (unlike Boost Loss, which was training-only), but SAFE:
# only Linear/mul/add/sigmoid, does not touch the MSDeformAttn custom op, preserves
# the token/feature count. An ONNX parity check on the first checkpoint closes the
# deployment-risk question (doc S7 step 2).
#
# WARM-START DESIGN: the module is an EXACT IDENTITY at init (zero-init LayerScale),
# so this fine-tune begins precisely at ssl1's operating point and learns how much
# fusion to add -- avoiding the epoch-0 shock that made Boost Loss a net tax.
#
# Protocol: fine-tune from the ssl1 checkpoint (run_detector_cctm.sbatch), A/B the
# organic + 41 AP curves against the ssl1 baseline (organic ~0.586 / 41 ~0.762;
# continued-train band organic ~0.55-0.58 / 41 ~0.73-0.75). Gate on organic.
# Boost Loss (Exp A) was DECLINED -- beta-sweep 1.0->0.5 monotonically negative
# (organic 0.42 -> 0.525, both below baseline). CCTM is the second and final
# portable Cross-DINO module. STOP RULE (doc S7): if CCTM also fails to move
# organic, Cross-DINO is investigated-and-declined -- do NOT port the Strip-MLP
# backbone.
# CCTM-LR NOTE: an uniform-LR warm-start (dino_cctm1, aborted) confirmed the identity
# init works (epoch-0 organic 0.561 / 41 0.759 ≈ ssl1, no shock) but also that gamma
# barely moves at the body's 1e-5 fine-tune rate -- an architectural module grafted onto
# a converged model needs a faster rate to gain traction, else a null is a false negative.
# So the cctm.* params train at 10x (1e-4, the usual from-scratch rate) via lr_cctm_mult
# while the warm-started backbone/encoder stay at 1e-5 (see util/get_param_dicts.py).
_base_ = ['DINO_4scale_swin.py']

use_cctm = True
lr_cctm_mult = 10.0        # cctm.* params at args.lr * 10 = 1e-4; body stays at 1e-5
