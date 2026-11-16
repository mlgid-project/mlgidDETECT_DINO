"""Appearance gap, measured: how do background noise, RINGS and SEGMENTS look in the processed (HE) image the network sees, per image source? CPU only.

  python heatmap/appearance_compare.py <name>=<dump.pkl>:<set> [...] [--max_frames 60]
e.g. real41=hm_runs/plots/dets_he_tf32_seed42.pkl:41  organic=...:organic  legacy=...:sim  realsim=hm_runs/plots/dets_realsim_single_ep7_on_simframes.pkl:sim_bench2_single
Input pickles: heatmap/dump_dets.py or heatmap/probe_frames.py (frames carry img, mask, gt boxes, optional 'gt_ring' flag, else the geometric ring criterion).
Per source it prints quartiles (p25 / median / p75) of
  BACKGROUND NOISE  (per frame, valid pixels away from the mask, residual = image - 8 px blur): sd, lag-1 autocorrelation along q and along chi
  RINGS   amplitude (profile peak - flank median, 0-1 image units), amplitude / flank sd, FWHM and wing ratio (profile at +-25 px / peak), integrated SNR (amplitude / (flank sd / sqrt(valid rows))),
          variation along chi at 2 / 8 / 32 row blocks (cv), gap to the nearest ring, box width, box width / FWHM
  SEGMENTS  amplitude (smoothed peak - annulus median), amplitude / annulus sd, FWHM along q and along chi, box width and height"""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import cv2
from scipy.ndimage import binary_erosion
from heatmap import evaluation as E

H, W = 512, 1024


def noise_stats(img, mask):
    core = binary_erosion(mask, iterations=12)
    if core.sum() < 5000:
        return None
    r = np.where(core, img - cv2.GaussianBlur(np.where(mask, img, img[mask].mean()).astype(np.float32), (0, 0), 8.0), 0.0)
    sd = float(r[core].std())
    def lag(axis):
        a, b = (r[:, :-1], r[:, 1:]) if axis == 1 else (r[:-1], r[1:])
        m = (core[:, :-1] & core[:, 1:]) if axis == 1 else (core[:-1] & core[1:])
        return float((a[m] * b[m]).mean() / max(sd ** 2, 1e-9))
    return dict(sd=sd, lag_q=lag(1), lag_chi=lag(0))


def fwhm(p, k, cap=60):
    half = (p[k] + np.nanmin(p)) / 2 if False else None
    return None


def width_at_half(prof, pk, base, amp, cap=80):
    half = base + amp / 2
    l = pk
    while l > 0 and prof[l] > half and pk - l < cap: l -= 1
    r = pk
    while r < len(prof) - 1 and prof[r] > half and r - pk < cap: r += 1
    return float(r - l)


def ring_row(img, mask, b, xs):
    xc = int(round((b[0] + b[2]) / 2))
    if xc < 45 or xc > W - 46:
        return None
    m = mask[:, xc - 40:xc + 41]; a = img[:, xc - 40:xc + 41]
    rows = m[:, 36:45].all(1) & m[:, 40]
    if rows.sum() < 40:
        return None
    P = np.where(m[rows], a[rows], np.nan)
    prof = np.nanmean(P, 0)
    base = float(np.nanmedian(np.r_[prof[:16], prof[-16:]])); pk = 36 + int(np.nanargmax(prof[36:45])); amp = float(prof[pk] - base)
    sd = float(np.nanstd(np.r_[P[:, :16].ravel(), P[:, -16:].ravel()]))
    if amp <= 0:
        return dict(amp=amp, amp_sd=amp / max(sd, 1e-6), fwhm=np.nan, wing=np.nan, snr=np.nan, cv2=np.nan, cv8=np.nan, cv32=np.nan, gap=np.nan, w=float(b[2] - b[0]), wf=np.nan)
    fw = width_at_half(prof, pk, base, amp)
    wing = float(np.nanmean([prof[max(pk - 25, 0)], prof[min(pk + 25, 80)]]) - base) / amp
    col = np.nanmean(P[:, 38:43], 1)
    cvs = []
    for n in (2, 8, 32):
        k = len(col) // n
        blk = col[:k * n].reshape(k, n).mean(1) if k >= 4 else np.array([np.nan])
        cvs.append(float(np.nanstd(blk) / max(abs(np.nanmean(blk)), 1e-6)) if k >= 4 else np.nan)
    ctr = (b[0] + b[2]) / 2
    others = xs[np.abs(xs - ctr) > 1e-3]
    return dict(amp=amp, amp_sd=amp / max(sd, 1e-6), fwhm=fw, wing=wing, snr=amp / max(sd / np.sqrt(rows.sum()), 1e-6), cv2=cvs[0], cv8=cvs[1], cv32=cvs[2],
                gap=float(np.min(np.abs(others - ctr))) if len(others) else np.nan, w=float(b[2] - b[0]), wf=float((b[2] - b[0]) / fw))


def seg_row(img, mask, b):
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, x1 = max(x0, 0), min(max(x1, x0 + 2), W); y0, y1 = max(y0, 0), min(max(y1, y0 + 2), H)
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None
    pad = 12
    X0, X1, Y0, Y1 = max(x0 - pad, 0), min(x1 + pad, W), max(y0 - pad, 0), min(y1 + pad, H)
    sub = img[Y0:Y1, X0:X1]; msub = mask[Y0:Y1, X0:X1]
    ann = msub.copy(); ann[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    if ann.sum() < 30 or not msub[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].any():
        return None
    sm = cv2.GaussianBlur(np.where(msub, sub, np.median(sub[msub])).astype(np.float32), (0, 0), 1.0)
    core = sm[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    iy, ix = np.unravel_index(np.argmax(core), core.shape); iy += y0 - Y0; ix += x0 - X0
    base = float(np.median(sub[ann])); sd = float(sub[ann].std()); amp = float(sm[iy, ix] - base)
    if amp <= 0:
        return dict(amp=amp, amp_sd=amp / max(sd, 1e-6), fq=np.nan, fc=np.nan, w=float(b[2] - b[0]), h=float(b[3] - b[1]))
    pq = sm[max(iy - 1, 0):iy + 2, :].mean(0); pc = sm[:, max(ix - 1, 0):ix + 2].mean(1)
    return dict(amp=amp, amp_sd=amp / max(sd, 1e-6), fq=width_at_half(pq, ix, base, amp, 40), fc=width_at_half(pc, iy, base, amp, 120), w=float(b[2] - b[0]), h=float(b[3] - b[1]))


def main():
    p = argparse.ArgumentParser(); p.add_argument('src', nargs='+'); p.add_argument('--max_frames', type=int, default=60)
    a = p.parse_args()
    cache = {}
    for s in a.src:
        name, rest = s.split('=', 1); path, ds = rest.rsplit(':', 1)
        if path not in cache:
            cache[path] = pickle.load(open(path, 'rb'))
        frames = cache[path]['sets'][ds][:a.max_frames]
        N, R, S = [], [], []
        for f in frames:
            img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool); gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
            ns = noise_stats(img, mask)
            if ns:
                N.append(ns)
            if len(gt) == 0:
                continue
            ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else E.is_ring_geom(gt, mask)
            xs = np.sort((gt[ring, 0] + gt[ring, 2]) / 2)
            for b in gt[ring]:
                r = ring_row(img, mask, b, xs)
                if r: R.append(r)
            for b in gt[~ring][:80]:
                r = seg_row(img, mask, b)
                if r: S.append(r)
        print(f'\n================ {name}: {len(frames)} frames, {len(R)} rings, {len(S)} segments', flush=True)
        def show(title, rows, keys):
            print(f'  {title}')
            for k, lab in keys:
                v = np.array([r[k] for r in rows], float); v = v[np.isfinite(v)]
                if len(v):
                    print(f'    {lab:34s} p25 {np.percentile(v, 25):8.3f}  median {np.median(v):8.3f}  p75 {np.percentile(v, 75):8.3f}')
        show('BACKGROUND (per frame)', N, [('sd', 'pixel noise sd'), ('lag_q', 'lag-1 autocorr along q'), ('lag_chi', 'lag-1 autocorr along chi')])
        show('RINGS', R, [('amp', 'amplitude'), ('amp_sd', 'amplitude / flank sd (one pixel)'), ('snr', 'integrated snr'), ('fwhm', 'FWHM px'), ('wing', 'wing ratio at +-25 px'), ('cv2', 'chi variation cv, 2-row blocks'),
                          ('cv8', 'chi variation cv, 8-row blocks'), ('cv32', 'chi variation cv, 32-row blocks'), ('gap', 'gap to nearest ring px'), ('w', 'box width px'), ('wf', 'box width / FWHM')])
        show('SEGMENTS', S, [('amp', 'amplitude'), ('amp_sd', 'amplitude / annulus sd'), ('fq', 'FWHM along q px'), ('fc', 'FWHM along chi px'), ('w', 'box width px'), ('h', 'box height px')])


if __name__ == '__main__':
    main()
