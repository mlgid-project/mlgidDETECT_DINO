"""
Canonical fingerprinting for GIWAXS polar frames — shared by shard + eval extractors.

Goal: map ANY GIWAXS frame (shard 512x1024 polar PNG, or eval reciprocal q-map of
arbitrary shape) into a small, contrast-invariant descriptor so that the SAME physical
scattering pattern matches across different conversion pipelines / quantization, while
distinct patterns do not.

Two signals per frame:
  - descriptor: 32x64 (chi x q) per-image z-scored patch, flattened float32 (2048-d).
    Cosine of two z-scored patches ~= spatial correlation -> robust to brightness/contrast.
  - phash: 64-bit perceptual hash (imagehash.phash, hash_size=8) on a percentile-stretched
    uint8 version -> fast Hamming blocking for near-duplicate / exact-copy detection.

Eval reciprocal maps are caked to the shard polar convention (|q| horizontal, chi vertical)
before fingerprinting. Orientation/origin-corner ambiguity is handled at match time by
trying all flips and taking the max similarity (see detect_leaks.py).
"""
import numpy as np
import cv2
from PIL import Image
import imagehash

NCHI, NQ = 512, 1024          # shard polar convention (rows=chi, cols=q)
DCHI, DQ = 32, 64             # descriptor grid


def _valid_mask(a):
    return np.isfinite(a) & (a != 0.0)


def cake_reciprocal(img, nq=NQ, nchi=NCHI):
    """Reciprocal-space q-map (qxy x qz) -> polar (chi x q), origin = bottom-left corner.
    Inverse-mapping with normalized-convolution masking (no bleed from no-data).
    Mirrors datasets/polar_converted/convert_giwaxs_polar.py. Used only for ~50 eval frames."""
    import scipy.ndimage as ndi
    a = np.asarray(img, np.float64)
    H, W = a.shape
    m = _valid_mask(a)
    A = np.where(m, a, 0.0)
    M = m.astype(np.float64)
    qmax = float(np.hypot(W - 1, H - 1))
    chi = np.linspace(np.pi / 2, 0.0, nchi)          # row 0 = 90 deg
    q = np.linspace(0.0, qmax, nq)                    # col 0 = q0
    QQ, CC = np.meshgrid(q, chi)
    col = QQ * np.cos(CC)
    row = (H - 1) - QQ * np.sin(CC)
    coords = np.vstack([row.ravel(), col.ravel()])
    num = ndi.map_coordinates(A, coords, order=1, mode="constant", cval=0.0).reshape(nchi, nq)
    den = ndi.map_coordinates(M, coords, order=1, mode="constant", cval=0.0).reshape(nchi, nq)
    out = np.zeros((nchi, nq), np.float32)
    good = den > 0.5
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def to_u8(img):
    """Percentile-stretch valid pixels to 0..255 (log domain); no-data -> 0. Consistent
    across shard (already log/hist-eq baked) and eval (linear) so phash is comparable."""
    a = np.asarray(img, np.float32)
    m = _valid_mask(a)
    out = np.zeros(a.shape, np.uint8)
    if m.sum() < 16:
        return out
    v = a[m]
    if v.min() >= 0:                                  # linear eval intensity -> log
        v = np.log1p(np.clip(v, 0, None))
        a = np.where(m, np.log1p(np.clip(a, 0, None)), 0.0)
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((a - lo) / (hi - lo), 0, 1)
    out[m] = (g[m] * 255).astype(np.uint8)
    return out


def descriptor(img):
    """img: 2D polar (chi x q), any intensity scale, no-data=0/NaN.
    -> (DCHI*DQ,) float32 z-scored over valid pixels (invalid -> 0)."""
    a = np.asarray(img, np.float32)
    m = _valid_mask(a)
    a = np.where(m, a, 0.0)
    if m.sum() >= 16 and a[m].min() >= 0:             # log-compress positive intensities
        a = np.where(m, np.log1p(a), 0.0)
    small = cv2.resize(a, (DQ, DCHI), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(m.astype(np.float32), (DQ, DCHI), interpolation=cv2.INTER_AREA) > 0.5
    if msmall.sum() < 8:
        return np.zeros(DCHI * DQ, np.float32)
    vals = small[msmall]
    mu, sd = vals.mean(), vals.std()
    if sd < 1e-6:
        sd = 1.0
    z = np.zeros_like(small)
    z[msmall] = (small[msmall] - mu) / sd
    return z.reshape(-1).astype(np.float32)


def phash64(img):
    """64-bit perceptual hash as uint64, on the percentile-stretched uint8 image."""
    u8 = to_u8(img)
    h = imagehash.phash(Image.fromarray(u8), hash_size=8)
    bits = h.hash.reshape(-1)
    v = np.uint64(0)
    for b in bits:
        v = (v << np.uint64(1)) | np.uint64(1 if b else 0)
    return v


def quality_stats(u8_polar):
    """Quality flags from the native 8-bit polar shard image (no-data = 0)."""
    a = np.asarray(u8_polar)
    total = a.size
    valid = int((a != 0).sum())
    frac_valid = valid / total
    sat = float((a >= 254).sum()) / max(valid, 1)
    if valid:
        nz = a[a != 0]
        mean = float(nz.mean()); dyn = float(nz.max() - nz.min())
    else:
        mean = 0.0; dyn = 0.0
    blank = valid < 0.02 * total or dyn < 4
    return dict(frac_valid=round(frac_valid, 4), saturation=round(sat, 4),
                mean=round(mean, 2), dyn=dyn, blank=bool(blank))
