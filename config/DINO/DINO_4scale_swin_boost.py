# Exp A of the Cross-DINO portable subset (docs/CROSS_DINO_INVESTIGATION.md S4a/S7):
# Boost Loss + Category-Size soft label (arXiv:2505.21868 Eq. 4-5) on the 2-class
# ring/segment DINO. TRAINING-ONLY (loss is not on the exported ONNX path).
# Protocol: fine-tune from the ssl1 checkpoint (run_detector_boost.sbatch), A/B the
# organic + 41 AP curves against the well-characterized warm-start band
# (organic ~0.55-0.58 / 41 ~0.73-0.75, established across dino_semi2-4 burn-ins).
# Gate on organic; stop rule per the investigation doc S7.
#
# DOMAIN CAVEAT (S4a): cs = sqrt(normalized w*h). Our boxes are elongated —
# typical ring cs ~ 0.17 (w~1.0, h~0.03), segment cs ~ 0.04 — so ALL positive
# targets are far below 1 and the positive loss term shrinks accordingly, with
# rings weighted ~4x over segments. beta is the calibration knob (cs^beta;
# beta<1 lifts small cs toward 1): paper default 1.0 first, sweep {0.5, 0.25}
# if training degrades or segment recall drops.
_base_ = ['DINO_4scale_swin.py']

use_boost_loss = True
boost_alpha = 0.25         # paper default (matches focal_alpha)
boost_beta = 1.0           # paper default; the per-dataset knob (see caveat above)
boost_gamma = 2.0          # paper default (matches focal gamma)
