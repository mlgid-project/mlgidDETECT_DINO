"""Build the background-donor bank: real GIWAXS frames with every diffraction feature removed.

This is what makes `realbkg_simulation.py` possible. Each donor is a real linear frame converted to
polar by the EVALUATION path's own geometry rule (`pygidloader.load_worker`:
GEO_QMAX = hypot(q_z[-1], q_xy[-1]), GEO_PIXELPERANGSTROEM = shape[0]/q_z[-1], then
calc_polar_image), with its own diffraction features removed and every other pixel passed through
untouched.

FEATURE REMOVAL. In a polar GIWAXS frame every feature -- spot, arc or ring -- is NARROW IN q,
while the background (low-q halo, radial falloff, panel structure) is broad in q. So the detector
is a difference-of-Gaussians ALONG q, OR-ed with a 2-D DoG for compact spots, both on log
intensity because the dynamic range spans decades. The noise level is re-estimated by
sigma-clipping: a tile packed with arcs has a huge MAD, which would hide exactly what we are
looking for. Thresholds are one-sided, so DARK features -- panel seams, beam-stop edges -- are
never flagged and survive untouched. That is the point: those are the parts a parametric
background could not produce.

THE FILL MATTERS AS MUCH AS THE DETECTION. Flagged pixels are refilled from a masked Gaussian
estimate that excludes the flagged pixels themselves, plus a fluctuation copied from the SAME
q-COLUMN at a random chi offset. A GIWAXS background varies fast with q and slowly with chi, so a
chi-shifted copy has the right radial level and carries real, correlated texture. Measured: with
an iid resample the lag-1 autocorrelation of a donor fell from 0.88 to 0.015 while untouched tiles
of the same frame stayed at 0.71; with the chi-shift it holds at 0.65. The shifts are random
rather than on a fixed grid -- a fixed grid refills a long removed stripe from the same few offsets
and the transplanted texture repeats as visible dashes.

    python realbkg_sim/build_donor_bank.py [frames_per_entry]
    BANK_OUT=... python realbkg_sim/build_donor_bank.py 12     # what produced bank4

PRODUCED (2026-09-15): sim_background_bank4.h5, 444 donors, removed fraction p50 0.23,
leftover-z p50 35.8. `realbkg_simulation.py` then ranks and keeps 189 of them.

OPEN: leftover-z p50 for the SELECTED pool is ~15 against a detection threshold of 3, so a few
donors still carry a faint unremoved real feature -- an unlabelled positive in training. The
candidate fix is to run the trained detector over the corpus and inpaint what IT finds, instead of
this hand-built q-narrow detector.
"""
import json
import time

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK   = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
DATA   = f'{WORK}/datasets'
DONOR_SRC = os.environ.get('DONOR_SRC', f'{DATA}/ekaterina_aftermlgidFIT')
CACHE  = os.environ.get('REALBKG_CACHE', f'{WORK}/tmp_diag/sim2')
INVENTORY = f'{CACHE}/inventory.json'

from util.configuration import Config
from util.exp_preprocess import calc_polar_image

HEIGHT, WIDTH = 512, 1024
OUT = os.environ.get('BANK_OUT', f'{DATA}/sim_background_bank.h5')

def to_polar(raw, q_z, q_xy):
    c = Config(); c.PREPROCESSING_CUDA = False
    c.PREPROCESSING_POLAR_SHAPE = [HEIGHT, WIDTH]
    c.GEO_QMAX = float(np.hypot(q_z[-1], q_xy[-1]))
    c.GEO_PIXELPERANGSTROEM = raw.shape[0] / float(q_z[-1])
    c.GEO_RECIPROCAL_SHAPE = list(raw.shape)
    pol = np.asarray(calc_polar_image(c, np.nan_to_num(raw).astype(np.float32),
                                      polar_shape=[HEIGHT, WIDTH]), dtype=np.float32)
    return pol, (~np.isnan(pol) & (pol != 0)), c.GEO_QMAX

def _nc(img, w, sig):
    num = cv2.GaussianBlur((img*w).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(w.astype(np.float32), (0, 0), sig)
    return num/np.maximum(den, 1e-6)

def _tile_sig(d, good, T=64):
    sig = np.empty_like(d)
    for r in range(0, d.shape[0], T):
        for c in range(0, d.shape[1], T):
            blk = d[r:r+T, c:c+T][good[r:r+T, c:c+T]]
            sig[r:r+T, c:c+T] = (1.4826*np.median(np.abs(blk-np.median(blk)))
                                 if blk.size > 48 else np.nan)
    v = sig[np.isfinite(sig)]
    sig[~np.isfinite(sig)] = np.median(v) if v.size else 1.0
    return np.maximum(cv2.GaussianBlur(sig, (0, 0), T/2), 1e-6)

def _detect(f, det, s_peak, s_bkg, k):
    lg = np.log10(np.maximum(f, 1e-3))
    d2 = cv2.GaussianBlur(lg, (0,0), s_peak) - cv2.GaussianBlur(lg, (0,0), s_bkg)
    dq = (cv2.GaussianBlur(lg, (0,0), sigmaX=s_peak, sigmaY=0.01)
          - cv2.GaussianBlur(lg, (0,0), sigmaX=s_bkg, sigmaY=0.01))
    good = det.copy()
    for _ in range(3):
        z = np.maximum(d2/_tile_sig(d2, good), dq/_tile_sig(dq, good))
        good = det & (z <= k)
        if good.sum() < 0.25*det.sum(): break
    return z

def suppress(img, mask, s_peak=2.5, s_bkg=16.0, k=3.0, grow=4, rounds=2, edge=7):
    """-> (background, removed_mask, leftover_score). Every unflagged pixel is the real frame's.

    The fill TRANSPLANTS texture: the replacement fluctuation is taken from the same frame shifted
    by a random offset, not resampled pixel-by-pixel. Independent resampling has the right variance
    but is white, so refilled regions came out as flat blotches against a correlated background --
    visible as rectangular patches in the first version of this bank.
    """
    det = cv2.erode(mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*edge+1,)*2)).astype(bool)
    lo = float(np.percentile(img[mask], 1)) if mask.any() else 1.0
    f = np.where(mask, img, lo).astype(np.float32)
    removed = np.zeros_like(mask)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow+1,)*2)
    for _ in range(rounds):
        z = _detect(f, det & ~removed, s_peak, s_bkg, k)
        new = det & (z > k) & ~removed
        if new.sum() == 0: break
        new = cv2.dilate(new.astype(np.uint8), ker).astype(bool) & mask
        removed |= new
        keep = mask & ~removed
        if keep.sum() < 0.2*mask.sum(): break
        sm = _nc(f, keep, s_bkg)
        res = np.where(keep, f - sm, 0.0).astype(np.float32)
        # Fill from the SAME q-COLUMN at a different chi. A GIWAXS background varies fast with q
        # and slowly with chi, so a chi-shifted copy has the right radial level and carries real,
        # correlated texture. (A 2-D roll or an iid resample both destroy the pixel-to-pixel
        # correlation: measured lag-1 autocorrelation fell from 0.88 to 0.015 with iid fill, while
        # untouched tiles of the same frame stayed at 0.71.)
        fill = np.zeros_like(f); got = np.zeros_like(mask)
        need = removed.copy()
        # random, non-repeating shifts. A fixed grid of shifts refills a long removed stripe from
        # the same few offsets and the transplanted texture repeats, which showed up as faint
        # regular dashes in the composed frames.
        for d in np.unique(np.random.randint(20, HEIGHT-20, 40)):
            if not need.any():
                break
            src_ok = np.roll(keep, int(d), 0)
            take = need & src_ok
            if not take.any():
                continue
            fill[take] = np.roll(res, int(d), 0)[take]
            got |= take; need &= ~take
        pool = res[keep]
        if need.any() and pool.size > 100:
            fill[need] = np.random.choice(pool, int(need.sum()))
        f = np.where(removed, sm + fill, f).astype(np.float32)
    z = _detect(f, det, s_peak, s_bkg, k)
    leftover = float(np.percentile(z[det], 99.9)) if det.any() else 99.
    return np.where(mask, np.maximum(f, 0), 0.0).astype(np.float32), removed, leftover


if __name__ == '__main__':
    rows = json.load(open(INVENTORY))
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = []
    for r in rows:
        n = r['n']
        idx = np.unique(np.linspace(0, n-1, min(PER, n)).astype(int))
        jobs += [(r['path'], r['entry'], int(i)) for i in idx]
    print(f'{len(jobs)} donor frames from {len(rows)} entries', flush=True)
    bkgs, masks, meta, t0 = [], [], [], time.time()
    for n, (p, ent, i) in enumerate(jobs):
        try:
            with h5py.File(p, 'r') as fh:
                g = fh[ent]
                qz, qxy = np.asarray(g['data/q_z']), np.asarray(g['data/q_xy'])
                raw = np.asarray(g['data/img_gid_q'][i], dtype=np.float32)
            if raw.shape != (len(qz), len(qxy)):
                if raw.shape == (len(qxy), len(qz)):
                    raw, qz, qxy = raw.T, qxy, qz        # some entries store the axes swapped
                else:
                    continue
            pol, m, qmax = to_polar(raw, qz, qxy)
            if m.mean() < 0.25: continue
            b, rm, left = suppress(pol, m)
            bkgs.append(b.astype(np.float32)); masks.append(m)
            meta.append(dict(path=p, entry=ent, frame=i, qmax=float(qmax),
                             removed=float(rm.mean()), leftover=left,
                             med=float(np.median(pol[m])), p99=float(np.percentile(pol[m], 99))))
        except Exception as e:
            print('ERR', p, ent, i, repr(e)[:80], flush=True)
        if (n+1) % 25 == 0:
            print(f'  {n+1}/{len(jobs)}  kept {len(bkgs)}  {time.time()-t0:.0f}s', flush=True)
    B = np.stack(bkgs); M = np.stack(masks)
    with h5py.File(OUT, 'w') as o:
        o.create_dataset('background', data=B, compression='gzip', compression_opts=1)
        o.create_dataset('mask', data=M, compression='gzip', compression_opts=1)
        o.create_dataset('meta', data=np.array([json.dumps(x) for x in meta], dtype=h5py.string_dtype()))
    lv = np.array([x['leftover'] for x in meta]); rmv = np.array([x['removed'] for x in meta])
    print(f'\nwrote {OUT}  {B.shape}  {os.path.getsize(OUT)/1e9:.2f} GB')
    print(f'removed fraction : p50 {np.median(rmv):.3f}  p90 {np.percentile(rmv,90):.3f}')
    print(f'leftover z (99.9): p50 {np.median(lv):.2f}  p90 {np.percentile(lv,90):.2f}  max {lv.max():.2f}')
