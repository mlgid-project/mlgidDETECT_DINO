# Style-transfer input matching Phase-0 MVP (docs/STYLE_TRANSFER_INVESTIGATION.md):
# monotone per-image CDF match of SYNTHETIC training images onto a reference intensity
# distribution pooled from REAL corpus frames (backbone_ssl_corpus.h5, disjoint from eval,
# in the model's [0,1] input space). Closes the pixel-distribution SHAPE gap the diagnostics
# show (synthetic spiky/bright vs real smooth) -- distinct from the reverted Path A (mask
# geometry + quantization levels, MODIFICATIONS.md phase H).
#
# TRAINING-ONLY: applied only inside SimulationDataset.__getitem__ (main.py); real-image
# preprocessing, the eval path, and the exported ONNX graph are byte-identical to baseline.
# Rank-preserving remap -> no-data (0) stays 0 and intensity ordering is preserved.
#
# Warm-start screen from ssl1. GATE = the faint(vis=1)/high-q(q682-1024) recall probe
# (diag_compare.py), NOT AP alone -- Path A gated on AP and saw no gain; the recall probe
# can reveal a faint-recall change AP would miss. Stop after Phase 0 if the probe is flat.
_base_ = ['DINO_4scale_swin.py']

use_style_match = True
style_ref_path = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/style_ref.pt'
