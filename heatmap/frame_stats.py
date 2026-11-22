"""Simple, interpretable image statistics of one model-input frame (polar image in [0,1], invalid pixels 0, mask) -- to ask WHAT differs between real frames of a latent-map region and simulated ones.

  from heatmap.frame_stats import frame_stats, NAMES
  s = frame_stats(img, mask)       # dict of floats (NaN if the frame has < 1000 valid pixels)

Features (all computed on valid pixels only; q = columns, chi = rows):
  valid        share of valid pixels (detector gaps / missing wedge)
  mean, std, p05, p50, p95   intensity distribution of the contrast-processed image
  sat, dark    share of pixels > 0.98 / < 0.05 (washed out / empty)
  q_slope      mean intensity of the last third of the q columns minus the first third (background fall-off along q)
  chi_slope    same along chi (rows)
  streak       std of the column means after removing their smooth trend / std (vertical rods and rings)
  rowvar       std of the row means after detrending (horizontal structure)
  noise        robust (MAD) std of the high-pass residual (image - 15 px local mean)
  texture      std of the mid-scale background (15 px local mean - 61 px local mean)
  peaks        local maxima of the smoothed residual above 4 noise, per valid megapixel (spots)
  peak_contrast  mean of the 30 strongest of them / noise
  edge         length of the mask boundary relative to the image perimeter (mask complexity)"""
import numpy as np
from scipy import ndimage as ndi

NAMES = ['valid', 'mean', 'std', 'p05', 'p50', 'p95', 'sat', 'dark', 'q_slope', 'chi_slope', 'streak', 'rowvar', 'noise', 'texture', 'peaks', 'peak_contrast', 'edge']


def _norm_filter(x, m, size):
    num = ndi.uniform_filter(x * m, size, mode='constant'); den = ndi.uniform_filter(m.astype(np.float32), size, mode='constant')
    return num / np.maximum(den, 1e-6), den


def _detrended_std(v, ok):
    v = v[ok]
    if len(v) < 20:
        return np.nan
    t = ndi.uniform_filter1d(v, 41, mode='nearest')
    return float(np.std(v - t) / (np.std(v) + 1e-6))


def frame_stats(img, mask):
    img = np.asarray(img, np.float32); m = np.asarray(mask, bool)
    out = dict.fromkeys(NAMES, np.nan)
    if m.sum() < 1000:
        return out
    v = img[m]; H, W = img.shape
    out['valid'] = float(m.mean()); out['mean'] = float(v.mean()); out['std'] = float(v.std())
    out['p05'], out['p50'], out['p95'] = (float(x) for x in np.percentile(v, [5, 50, 95]))
    out['sat'] = float((v > 0.98).mean()); out['dark'] = float((v < 0.05).mean())
    mf = m.astype(np.float32)
    cn, cd = (img * mf).sum(0), mf.sum(0); okc = cd > 0.2 * H; cm = np.where(okc, cn / np.maximum(cd, 1), np.nan)
    rn, rd = (img * mf).sum(1), mf.sum(1); okr = rd > 0.2 * W; rm = np.where(okr, rn / np.maximum(rd, 1), np.nan)
    idx = np.where(okc)[0]
    if len(idx) > 60:
        a, b = idx[:len(idx) // 3], idx[-(len(idx) // 3):]
        out['q_slope'] = float(cm[b].mean() - cm[a].mean()); out['streak'] = _detrended_std(cm[idx], np.ones(len(idx), bool))
    idr = np.where(okr)[0]
    if len(idr) > 60:
        a, b = idr[:len(idr) // 3], idr[-(len(idr) // 3):]
        out['chi_slope'] = float(rm[b].mean() - rm[a].mean()); out['rowvar'] = _detrended_std(rm[idr], np.ones(len(idr), bool))
    bg15, d15 = _norm_filter(img, mf, 15); bg61, d61 = _norm_filter(img, mf, 61)
    inner = (d15 > 0.95) & m
    if inner.sum() < 500:
        return out
    res = np.where(inner, img - bg15, 0.0)
    r = res[inner]; mad = float(np.median(np.abs(r - np.median(r))) * 1.4826); out['noise'] = mad
    out['texture'] = float(np.std((bg15 - bg61)[inner & (d61 > 0.95)])) if (inner & (d61 > 0.95)).any() else np.nan
    sm = ndi.gaussian_filter(res, 1.5); mx = (sm == ndi.maximum_filter(sm, 7)) & inner & (sm > 4 * max(mad, 1e-4) / 2.0)
    pk = sm[mx]
    out['peaks'] = float(len(pk) / (inner.sum() / 1e6))
    out['peak_contrast'] = float(np.sort(pk)[-30:].mean() / max(mad, 1e-4)) if len(pk) else 0.0
    edge = (m[1:, :] != m[:-1, :]).sum() + (m[:, 1:] != m[:, :-1]).sum()
    out['edge'] = float(edge / (2 * (H + W)))
    return out
