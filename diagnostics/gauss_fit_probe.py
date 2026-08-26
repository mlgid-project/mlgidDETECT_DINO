"""Phase AC.2 — the chi boxing convention, measured with a fitted sigma instead of a half-max walk.

AC set the q convention from data (both gates agree: box_w/FWHM_q = 0.65 organic, 0.63 on 41, sim
0.39, so w_coef is 1.64x too tight) but could NOT settle chi: organic read 0.73 and 41 read 1.16,
and the two disagree. Deciding `a_coef` needs that resolved, because the simulator currently sits at
1.10 -- right for 41, 1.5x too loose for organic -- and lowering it would decalibrate 41.

TWO REASONS THE AC NUMBER COULD BE WRONG IN CHI, both fixed here:

  CROWDING   `_fwhm_1d` walks outward from the maximum until the profile falls below half max. A
             same-radius neighbour props the profile up and the walk runs into it, inflating FWHM_chi
             and depressing box_h/FWHM_chi. Organic carries 98.6 segments per frame against 41's 24,
             so the confound is ~4x stronger on exactly the gate that read low. Here every pixel
             inside ANY OTHER labelled box is masked out before fitting, and the probe reports the
             masked and unmasked fits side by side so the size of the effect is visible rather than
             argued about.

  TRANSFORM  AC measured on the CLAHE'd image, where a Gaussian is no longer a Gaussian, and had to
             carry a calibration (measured/true = 0.74 in chi, 0.92 in q) derived on the simulator
             and assumed to transfer to real data. Here the real gates are fitted on
             `raw_polar_image` -- the physical peak, before log and equalisation -- so the real-side
             sigma needs no calibration at all. The simulator's sigma needs no fit either: it is
             known exactly, sigma_chi = box_h/a_coef and sigma_q = box_w/w_coef by construction
             (`simulation.py:335-339` and `668-669`), so its convention is 1.75 sigma in chi and
             0.50 sigma in q by definition.

So the comparison is physical sigma against physical sigma, with nothing calibrated on either side.
The simulator is ALSO fitted on its processed image, which recovers a known answer and therefore says
how much the AC estimator was biased -- a check on the earlier phase, not an input to this one.

Model per peak: A * exp(-(x-mx)^2/(2 sq^2) - (y-my)^2/(2 sc^2)) + B + C*(x-cx) + D*(y-cy).
The planar term matters on raw data, where the background varies steeply with q. Fits pinned at the
window edge, fits below the local noise, and windows with too few unmasked pixels are dropped and
counted, never silently kept.

Rings excluded throughout (AA.4 span criterion for the real gates, the simulator's own flag for sim).
GPU for the simulation only; the fitting is CPU. ~10 min. See tmp_diag/run_gaussfit.sbatch.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from scipy.optimize import least_squares

from simulation import FastSimulation
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from diagnostics.ring_count_fix import classify

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM_FRAMES = 60
PAD = 3.0          # window half-size, in box half-extents
MIN_PIX = 40
PCTS = [10, 25, 50, 75, 90]
A_COEF_TRUE = 3.5  # simulation.py SimulationConfig -- box_h = 2*a_widths*a_coef, sigma = box_h/a_coef
W_COEF_TRUE = 1.0


def pcts(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return {p: float(np.percentile(v, p)) for p in PCTS} if len(v) else {p: float('nan') for p in PCTS}


def _resid(p, X, Y, Z):
    A, mx, my, sq, sc, B, C, D = p
    m = A * np.exp(-((X - mx) ** 2) / (2 * sq * sq) - ((Y - my) ** 2) / (2 * sc * sc)) \
        + B + C * X + D * Y
    return m - Z


def fit_peak(img, b, others, mask_neighbours):
    """Fitted (sigma_q, sigma_chi) for one box, or None with a reason string."""
    H, W = img.shape
    cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
    bw = max(abs(b[2] - b[0]), 1.0); bh = max(abs(b[3] - b[1]), 1.0)
    hw = max(PAD * bw / 2, 6.0); hh = max(PAD * bh / 2, 6.0)
    x0 = int(np.clip(round(cx - hw), 0, W - 1)); x1 = int(np.clip(round(cx + hw), x0 + 3, W))
    y0 = int(np.clip(round(cy - hh), 0, H - 1)); y1 = int(np.clip(round(cy + hh), y0 + 3, H))
    sub = np.asarray(img[y0:y1, x0:x1], dtype=np.float64)
    if sub.size < MIN_PIX:
        return None, 'window'

    yy, xx = np.mgrid[y0:y1, x0:x1]
    ok = np.isfinite(sub) & (sub > 0)
    if mask_neighbours and len(others):
        for o in others:
            ok &= ~((xx >= o[0]) & (xx <= o[2]) & (yy >= o[1]) & (yy <= o[3]))
        ok |= ((xx >= b[0]) & (xx <= b[2]) & (yy >= b[1]) & (yy <= b[3])) & np.isfinite(sub) & (sub > 0)
    if ok.sum() < MIN_PIX:
        return None, 'pixels'

    Z = sub[ok]; X = xx[ok].astype(np.float64); Y = yy[ok].astype(np.float64)
    med = np.median(Z); mad = np.median(np.abs(Z - med)) * 1.4826
    amp0 = float(Z.max() - med)
    if not np.isfinite(amp0) or amp0 <= 0 or amp0 < 2 * max(mad, 1e-12):
        return None, 'noise'

    p0 = [amp0, cx, cy, max(bw / 2, 0.8), max(bh / 2, 0.8), med, 0.0, 0.0]
    lo = [amp0 * 1e-3, cx - bw, cy - bh, 0.4, 0.4, -np.inf, -np.inf, -np.inf]
    hi = [amp0 * 50, cx + bw, cy + bh, hw, hh, np.inf, np.inf, np.inf]
    p0 = [float(np.clip(v, l, h)) for v, l, h in zip(p0, lo, hi)]
    try:
        r = least_squares(_resid, p0, bounds=(lo, hi), args=(X, Y, Z),
                          max_nfev=400, xtol=1e-8, ftol=1e-8)
    except Exception:
        return None, 'fail'
    if not r.success:
        return None, 'fail'
    A, _mx, _my, sq, sc, *_ = r.x
    if sq > 0.9 * hw or sc > 0.9 * hh:
        return None, 'pinned'
    if A < 2 * max(mad, 1e-12):
        return None, 'noise'
    return (float(sq), float(sc)), 'ok'


def scan(name, frames, tag):
    """frames yields (image, boxes, is_ring). Returns fitted stats, masked and unmasked."""
    out = {}
    for mode in (True, False):
        S = dict(sq=[], sc=[], kq=[], kc=[], why={})
        for img, b, is_ring in frames():
            for i in np.where(~is_ring)[0]:
                # index-based, so duplicate boxes cannot delete the wrong row
                others = np.delete(b, i, axis=0) if mode else np.zeros((0, 4))
                res, why = fit_peak(img, b[i], others, mode)
                S['why'][why] = S['why'].get(why, 0) + 1
                if res is None:
                    continue
                sq, sc = res
                S['sq'].append(sq); S['sc'].append(sc)
                S['kq'].append(abs(b[i][2] - b[i][0]) / 2 / sq)
                S['kc'].append(abs(b[i][3] - b[i][1]) / 2 / sc)
        out['masked' if mode else 'raw'] = dict(
            n=len(S['sq']), why=S['why'], sq=pcts(S['sq']), sc=pcts(S['sc']),
            kq=pcts(S['kq']), kc=pcts(S['kc']))
    return dict(name=name, tag=tag, **out)


def real_frames(path, use_raw):
    def gen():
        cfg = Config()
        cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
        cfg.INPUT_DATASET = path
        ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                           load_labels=True) if detect_dataset_type(path) == 'pygid'
              else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                                   buffer_size=5))
        for gc in ds.iter_images():
            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            if not len(b):
                continue
            proc = np.asarray(gc.converted_polar_image)
            while proc.ndim > 2:
                proc = proc[0]
            _h, _sp, is_ring = classify(proc, b)
            if use_raw:
                im = np.asarray(gc.raw_polar_image, dtype=np.float64)
                while im.ndim > 2:
                    im = im[0]
                if im.shape != proc.shape:
                    continue
            else:
                im = proc.astype(np.float64)
            yield im, b, is_ring
        if hasattr(ds, 'close'):
            ds.close()
    return gen


def sim_frames(n, dev):
    def gen():
        sim = FastSimulation(device=dev)
        sim.sim_config.use_peak_clusters = False
        for k in range(n):
            sd = 90000 + k
            random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
            try:
                img, bx, _m, isr = sim.simulate_img()
            except Exception:
                continue
            b = bx.detach().cpu().numpy().astype(np.float64)
            if not len(b):
                continue
            im = img.detach().cpu().numpy()
            while im.ndim > 2:
                im = im[0]
            yield im.astype(np.float64), b, isr.detach().cpu().numpy().astype(bool)
    return gen


def prow(lab, d, key):
    print(f"  {lab:<34s}" + "".join(f"{d[key].get(p, float('nan')):10.2f}" for p in PCTS))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  PAD={PAD}  MIN_PIX={MIN_PIX}", flush=True)

    rows = []
    for nm, pth in DSETS:
        print(f"  fitting {nm} on RAW polar ...", flush=True)
        rows.append(scan(nm, real_frames(pth, True), 'raw polar (physical peak)'))
        print(f"  fitting {nm} on PROCESSED ...", flush=True)
        rows.append(scan(nm, real_frames(pth, False), 'CLAHE (what AC used)'))
    print("  fitting simulator on PROCESSED ...", flush=True)
    rows.append(scan("simulator", sim_frames(N_SIM_FRAMES, dev), 'CLAHE (what AC used)'))

    hdr = "".join(f"{('p' + str(p)):>10s}" for p in PCTS)
    print("\n" + "=" * 104)
    print("  FITTED PEAK SIGMA (px) — 2D Gaussian + planar background, neighbours masked out")
    print("=" * 104)
    print(f"  {'set / image':<34s}{'fits':>7s}   rejected")
    for r in rows:
        m = r['masked']
        rej = ", ".join(f"{k} {v}" for k, v in sorted(m['why'].items()) if k != 'ok')
        print(f"  {r['name'] + '  [' + r['tag'] + ']':<34s}{m['n']:7d}   {rej}")
    print(f"\n  sigma in chi                      {hdr}")
    for r in rows:
        prow(r['name'] + ' [' + r['tag'][:12] + ']', r['masked'], 'sc')
    print(f"\n  sigma in q                        {hdr}")
    for r in rows:
        prow(r['name'] + ' [' + r['tag'][:12] + ']', r['masked'], 'sq')

    print("\n" + "=" * 104)
    print("  THE CONVENTION: box half-extent in units of the fitted sigma")
    print(f"  simulator TRUTH by construction: chi = a_coef/2 = {A_COEF_TRUE / 2:.2f} sigma,"
          f"  q = w_coef/2 = {W_COEF_TRUE / 2:.2f} sigma")
    print("=" * 104)
    print(f"  k_chi = (box_h/2) / sigma_chi     {hdr}")
    for r in rows:
        prow(r['name'] + ' [' + r['tag'][:12] + ']', r['masked'], 'kc')
    print(f"\n  k_q   = (box_w/2) / sigma_q       {hdr}")
    for r in rows:
        prow(r['name'] + ' [' + r['tag'][:12] + ']', r['masked'], 'kq')

    print("\n" + "=" * 104)
    print("  HOW MUCH CROWDING WAS DISTORTING IT  (median k, neighbours masked vs not)")
    print("=" * 104)
    print(f"  {'set / image':<34s}{'k_chi masked':>14s}{'k_chi unmasked':>16s}"
          f"{'k_q masked':>12s}{'k_q unmasked':>14s}")
    for r in rows:
        print(f"  {r['name'] + '  [' + r['tag'][:12] + ']':<34s}"
              f"{r['masked']['kc'][50]:14.2f}{r['raw']['kc'][50]:16.2f}"
              f"{r['masked']['kq'][50]:12.2f}{r['raw']['kq'][50]:14.2f}")

    json.dump(rows, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/gauss_fit.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
