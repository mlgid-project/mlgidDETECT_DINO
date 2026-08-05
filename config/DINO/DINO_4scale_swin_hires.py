# Higher-resolution FROM-SCRATCH run (docs/HIRES_INVESTIGATION.md). The single variable vs ssl1
# is the polar grid WIDTH: 1024 -> 2048 q-pixels (HEIGHT/chi unchanged at 512), doubling the
# sampling of the q-axis where the high-q recall deficit lives. The native data carries the detail
# (organic raw q-image 1641x1641; 41 raw 1350x1350), so this exposes REAL signal, not upsampling.
#
# From-scratch = SSL-pretrained backbone (backbone_dir, inherited) + random detector head + ssl1's
# exact recipe (uniform 1e-5, no amp, 500 ep, lr-drop 280). The Swin-L 48x6 backbone is
# resolution-agnostic (windowed attention + relative-position bias, no absolute PE), so the
# 512x1024-pretrained backbone transfers to 512x2048 unchanged -- only the input tensor is wider.
#
# Matched control = ssl1 (organic 0.586 / 41 0.762) = THIS recipe at 512x1024. This is a NEW model
# line: the exported ONNX input becomes (1,1,512,2048), NOT a drop-in for the 512x1024 deployment.
# GATE = organic AP + the faint/high-q recall probe (diag_compare.py) vs ssl1 AND vs the deployed
# ssl1+baseline ensemble (organic 0.605 / 41 0.780).
#
# `polar_shape` drives both the synthetic sim (simulation.HEIGHT/WIDTH via main.py SimulationDataset)
# and the real-image eval resample (evaluate_giwaxs_ap PREPROCESSING_POLAR_SHAPE). Absent in every
# other config => [512,1024], so all prior runs / the deployed inference path are byte-identical.
_base_ = ['DINO_4scale_swin_ssl.py']     # SSL backbone init + ssl1 schedule/LR (from-scratch)

polar_shape = [512, 2048]                 # [H(chi), W(q)] -- the single variable under test
