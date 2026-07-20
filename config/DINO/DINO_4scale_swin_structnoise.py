# Structural sim2real noise injection (step 2 of the sim2real track; datasets/struct_noise.py).
#
# MOTIVATION (measured, not assumed). style-match (phase M) closed the 1-D intensity histogram
# gap and did NOT move faint recall. A direct structural measurement (tmp_diag/struct_gap.py,
# robust_grain.py, mf_snr.py) then showed the remaining gap is SPATIAL TEXTURE, invisible to the
# 1-D histogram: REAL corpus frames carry a heavy WHITE pixel-grain floor (robust/MAD high-pass
# residual ~0.121, autocorr ~1 px, isotropic); SYNTHETIC frames are ~4x smoother (MAD ~0.032).
# The detector only ever trains on smooth synthetic backgrounds -> its low-level features never
# adapt to the real noise floor -> it fails on real faint/high-q peaks. This injects real-level
# white grain so training backgrounds look real (a domain-adaptation augmentation).
#
# DESIGN: grain-only. Per-image white Gaussian grain, std ~ U[0.05, 0.13] (0.13 -> synth MAD grain
# ~0.13, matching real 0.121), added to the FINAL [0,1] image. NO peak-boosting: matched-filter SNR
# (mf_snr.py) shows peaks (median footprint ~210 px) sit at MF-SNR ~26 even in real grain, so
# spatial coherence keeps them detectable (only ~1% buried); explicit boosting was verified
# unnecessary. no-data (0) preserved.
#
# TRAINING-ONLY: applied only inside SimulationDataset.__getitem__ (main.py); real-image
# preprocessing, the eval path, and the exported ONNX graph are byte-identical to baseline.
#
# Warm-start screen from ssl1 (fair: a DATA change needs no architectural co-adaptation, same as
# style-match). GATE = the faint(vis=1)/high-q(q682-1024) recall probe (diag_compare.py), NOT AP
# alone. Stop after this Phase-0 screen if the probe is flat.
_base_ = ['DINO_4scale_swin.py']

use_struct_noise = True
struct_noise_sigma = (0.05, 0.13)
struct_noise_boost = False
