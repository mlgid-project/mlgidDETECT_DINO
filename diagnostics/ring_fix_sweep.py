"""Sweep the realbkg ring knobs against the real-41 ring targets.

Targets, measured on real 41 (legacy sim in brackets) on the processed image -- realbkg-41-gap-split
and ring_targets.py:
  rings/frame        broad 6-45, median 14, none ring-free   (legacy: 34% ring-free, 44% >= 21)
  organic            38% ring-free, median 1                  -> keep a ring-free share for organic
  ring box width p50 7.6 px                                   (6.8)
  FWHM / box p50     1.48                                     (1.79)    realbkg r6: 2.46
  ring contrast p50  0.22, p10 0.07                           (0.18)    realbkg r6: 0.14
  rings with a segment on them 15%                            (2%)      realbkg r6: 40%
Also reports segment stats (must not move) and unlabelled background bands vs labelled rings.

  BASE=config/... python diagnostics/ring_fix_sweep.py        (CPU; loads the hkl bank per variant)
"""
import os, sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_shape_compare as PS
from band_probe import band_stats

N = int(os.environ.get('SWEEP_FRAMES', 150))
BASE = os.environ.get('BASE', 'config/DINO/DINO_4scale_swin_realbkg_r6.py')
COUNTS = dict(realbkg_p_ring=0.7, realbkg_n_powder=(1, 2), realbkg_rings_cap=(1, 30))
VARIANTS = [
    ('r6 as is', {}),
    ('counts', dict(COUNTS)),
    ('counts+wbox2.1', dict(COUNTS, realbkg_ring_w_coef=2.1)),
    ('+segkeep0.3', dict(COUNTS, realbkg_ring_w_coef=2.1, realbkg_seg_on_ring_keep=0.3)),
    ('+segkeep0.3+rc4', dict(COUNTS, realbkg_ring_w_coef=2.1, realbkg_seg_on_ring_keep=0.3,
                             realbkg_ring_contrast_min=4.0)),
    ('+segkeep0.3+rc8', dict(COUNTS, realbkg_ring_w_coef=2.1, realbkg_seg_on_ring_keep=0.3,
                             realbkg_ring_contrast_min=8.0)),
]
if os.environ.get('VARIANTS'):
    VARIANTS = eval(os.environ['VARIANTS'])


def med(v, p=50):
    v = [x for x in v if np.isfinite(x)]
    return float(np.percentile(v, p)) if v else float('nan')


def run(tag, ov):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    nr, R, S, C, B, nseg = [], [], [], [], [], []
    for img, m, gt in PS.realbkg6(N, config=BASE, overrides=ov):
        ring = PS.is_ring_geom(gt, m) if len(gt) else np.zeros(0, bool)
        nr.append(int(ring.sum())); nseg.append(int((~ring).sum()))
        for b, isr in zip(gt, ring):
            st = PS.profile_stats(img, m, b)
            if st is not None:
                (R if isr else S).append(st)
        C += PS.crowding(gt, ring, m)
        bs = band_stats(img, m, gt)
        if bs is not None:
            B.append(bs[0.10])
    nr = np.array(nr)
    h = np.histogram(nr, bins=[0, 1, 6, 11, 21, 46, 1000])[0] / len(nr)
    print(f'\n== {tag}: {ov}')
    print(f'   rings/frame mean {nr.mean():5.1f} p50 {np.median(nr):3.0f} p90 {np.percentile(nr,90):3.0f} '
          f'max {nr.max():3d} | 0 / 1-5 / 6-10 / 11-20 / 21-45 / 46+: ' + ' '.join(f'{x:.2f}' for x in h)
          + f' | segs/frame {np.mean(nseg):.1f} (ring-free frames {np.mean(np.array(nseg)[nr == 0]) if (nr == 0).any() else float("nan"):.1f}, '
          f'>=11 rings {np.mean(np.array(nseg)[nr >= 11]) if (nr >= 11).any() else float("nan"):.1f})')
    print(f'   RING box w p50 {med([d["w"] for d in R]):.1f} | FWHM/box p50 {med([d["fwhm"]/d["w"] for d in R]):.2f} '
          f'| contrast p10 {med([d["con"] for d in R],10):.3f} p50 {med([d["con"] for d in R]):.3f} '
          f'| with a segment on them {np.mean([c["n_ov_seg"] > 0 for c in C]) if C else float("nan"):.0%} '
          f'| ring gap p10 {med([c["gap"] for c in C],10):.1f} p50 {med([c["gap"] for c in C]):.1f}')
    print(f'   SEG box w p50 {med([d["w"] for d in S]):.1f} | FWHM p50 {med([d["fwhm"] for d in S]):.1f} '
          f'| contrast p50 {med([d["con"] for d in S]):.3f} | unlabelled bands/frame {np.mean(B):.2f} '
          f'vs labelled rings {nr.mean():.1f}', flush=True)


if __name__ == '__main__':
    for tag, ov in VARIANTS:
        run(tag, ov)
