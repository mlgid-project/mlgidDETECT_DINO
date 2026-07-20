"""Structural sim2real noise injection (step-2 of the sim2real track), TRAINING-ONLY.

MOTIVATION (measured, not assumed). style-match closed the 1-D intensity histogram gap and did
NOT move faint recall. A direct structural measurement (tmp_diag/struct_gap.py + robust_grain.py)
then showed two coupled facts the 1-D histogram is blind to:
  * REAL corpus frames carry a heavy WHITE background grain floor: robust (MAD) high-pass residual
    ~0.121, autocorrelation length ~1 px (white), isotropic. SYNTHETIC frames are ~4x smoother
    (MAD ~0.032) because the sim's noise is added before a 3x3 smoothing kernel and is correlated.
  * SYNTHETIC peaks are generated with contrasts (median 0.08, faint quartile 0.024, p10 0.006)
    BELOW that real grain floor -> the model is trained to detect peaks on a smooth background at
    SNR that could never exist in real data, and NEVER sees the real faint regime: a moderate peak
    at SNR~1-3 buried in heavy grain. That is a concrete mechanism for the stuck faint recall.

CHOSEN DESIGN: grain-only (add_grain). A per-pixel WHITE Gaussian grain (real autocorr=1 px ->
white; isotropic), std=sigma drawn per image from [sigma_lo, sigma_hi], added to the FINAL [0,1]
image AFTER the sim's own smoothing so it survives; no-data (0) preserved exactly, valid pixels in
[1/255, 1]. This is a DOMAIN-ADAPTATION augmentation: the detector only ever trains on smooth
synthetic backgrounds and then fails on grainy real ones; injecting real-level grain adapts its
low-level features to the real noise floor while leaving peak supervision intact.

Why NO peak-boosting (the naive worry was wrong): per-pixel contrast vs sigma suggested grain would
bury most peaks, but a detector INTEGRATES over a peak's footprint, so the correct detectability
metric is matched-filter SNR ||signal||_2/sigma ~ amplitude*sqrt(N). Measured (tmp_diag/mf_snr.py):
peaks have median footprint ~210 px, and at real grain sigma=0.12 even the faintest quartile sits
at MF-SNR ~26 -- only ~1% of peaks are truly buried. So grain-only keeps peaks comfortably
detectable via spatial coherence; explicit boosting is unnecessary (it moved MF-SNR 26.2 -> 26.4).
The optional boost (boost_peaks / grain_with_peak_floor) is kept, off by default, for the rare high-
sigma regime; enable via struct_noise_boost=True.

DEPLOY-SAFE: applied ONLY in SimulationDataset.__getitem__ (main.py), gated by use_struct_noise
(default off). Real-image preprocessing, the eval path, and the exported ONNX graph never touch it
-> deployment byte-identical. Parallel in every structural respect to datasets/style_match.py.
"""
import torch
import torch.nn.functional as F

_MIN_VALID = 1.0 / 255.0


def add_grain(img, sigma):
    """Add per-pixel white Gaussian grain (std=sigma) to the valid pixels of a [0,1] image.
    img (H,W)/(1,H,W) float [0,1], 0=no-data. no-data stays 0, valid clamped to [1/255, 1]."""
    if sigma <= 0:
        return img
    valid = img > 0
    out = (img + torch.randn_like(img) * sigma).clamp(_MIN_VALID, 1.0)
    return torch.where(valid, out, torch.zeros_like(out))


def sample_sigma(sigma_lo, sigma_hi):
    """Per-image sigma ~ Uniform[sigma_lo, sigma_hi] (python float)."""
    if sigma_hi <= sigma_lo:
        return sigma_lo
    return float(torch.empty(1).uniform_(sigma_lo, sigma_hi).item())


def _gblur(x2d, sigma):
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x2d.device, dtype=x2d.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    x = x2d[None, None]
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x[0, 0]


def boost_peaks(img, boxes, sigma, snr_floor=1.5, max_boost=8.0, bg_sigma=8.0):
    """Lift each labeled peak's amplitude above its local background to >= snr_floor*sigma, so it
    survives grain of std=sigma. Deterministic (no noise added). img (H,W)/(1,H,W) [0,1]; boxes
    (N,4) xyxy pixel. Boost only raises the positive residual (peak hump) inside a box, capped at
    max_boost -> never shrinks bright peaks, adds no hard edges, cannot amplify pure noise."""
    if sigma <= 0 or boxes is None or len(boxes) == 0:
        return img
    t = img[0] if img.dim() == 3 else img
    H, W = t.shape
    bg = _gblur(t, bg_sigma)
    resid = (t - bg).clamp(min=0.0)                      # peak humps above local background
    target = snr_floor * sigma
    fac = torch.ones_like(t)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2) + 1)
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2) + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        a = float(resid[y1:y2, x1:x2].max())             # peak amplitude above local bg
        if a <= 1e-6:
            continue
        f = min(max_boost, max(1.0, target / a))
        if f > 1.0:
            sub = fac[y1:y2, x1:x2]
            fac[y1:y2, x1:x2] = torch.maximum(sub, torch.full_like(sub, f))
    boosted = (t + (fac - 1.0) * resid).clamp(0.0, 1.0)
    return boosted[None] if img.dim() == 3 else boosted


def grain_with_peak_floor(img, boxes, sigma, snr_floor=1.5, max_boost=8.0, bg_sigma=8.0):
    """Inject white grain (std=sigma) AND boost each labeled peak so it stays >= snr_floor*sigma
    above local background -> labeled peaks remain detectable-but-hard in real-level grain.
    img (H,W)/(1,H,W) [0,1], 0=no-data; boxes (N,4) xyxy pixel. no-data preserved, output in [0,1]."""
    if sigma <= 0:
        return add_grain(img, sigma)
    boosted = boost_peaks(img, boxes, sigma, snr_floor, max_boost, bg_sigma)
    return add_grain(boosted, sigma)
