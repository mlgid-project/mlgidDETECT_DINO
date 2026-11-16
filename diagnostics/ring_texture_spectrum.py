"""Along-chi ring texture BY SCALE: real 41 vs legacy vs realbkg r7 with candidate textures.

texture_scan.py found the cause of the realbkg 41 failure: realbkg rings are perfectly smooth along
chi (patchiness 0.003 vs real 41 0.106), and as little as sd 0.15 of smooth log-normal texture makes
conv3 fragment its own sim's rings (2.7 -> 7.6 boxes/ring, AP 0.93 -> 0.45). But that texture at sd
>= 0.3 also fragments the legacy-trained baseline, which handles real 41 fine -- so its CHARACTER is
wrong. This measures what real ring texture looks like, scale by scale, to calibrate against.

Per ring (processed image, central 3 columns, valid rows only), the along-chi profile p(y) is split
into octave bands with differences of Gaussian smoothings: band k = G(s_k) p - G(2 s_k) p, s_k in
1, 2, 4, 8, 16, 32 px; reported as the band's rms / mean(p). Bands 1-2 px ~ pixel noise, 8-32 px ~
patches a detector reads as segments.

Candidates are r7 + ring_chi_texture sd at several correlation lengths. CPU only.
"""
import os, sys, random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_shape_compare as PS

N = int(os.environ.get('SP_FRAMES', 60))
R7 = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'
BASE = dict(realbkg_ring_min_gap=8.0, realbkg_rings_cap=(2, 60), realbkg_mosaic_seed=1234)
SCALES = [1, 2, 4, 8, 16, 32]


def spectrum(frames):
    rows = []
    for img, m, gt in frames:
        if not len(gt):
            continue
        for b, r in zip(gt, PS.is_ring_geom(gt, m)):
            if not r:
                continue
            xc = int(np.clip((b[0] + b[2]) / 2, 1, img.shape[1] - 2))
            valid = m[:, xc - 1:xc + 2].all(1)
            if valid.sum() < 128:
                continue
            p = np.median(img[valid, xc - 1:xc + 2], axis=1).astype(np.float64)
            mu = max(p.mean(), 1e-6)
            rows.append([np.sqrt(np.mean((gaussian_filter1d(p, s, mode='nearest')
                                          - gaussian_filter1d(p, 2 * s, mode='nearest')) ** 2)) / mu
                         for s in SCALES])
    a = np.array(rows)
    return np.median(a, 0), len(a)


def show(tag, frames):
    med, n = spectrum(frames)
    print(f'{tag:<34} ' + '  '.join(f'{s:>2}-{2*s:<2}px {v:.4f}' for s, v in zip(SCALES, med)) + f'   n {n}',
          flush=True)


def r7(ov):
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    return list(PS.realbkg6(N, config=R7, overrides=dict(BASE, **ov)))


if __name__ == '__main__':
    print('along-chi ring texture, rms per octave band / ring mean (processed image)')
    show('real 41', PS.real(f'{PS.CUR}/41.h5', False))
    show('real organic', PS.real(f'{PS.CUR}/organic_labeled.h5', True))
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    show('legacy', PS.legacy(N))
    show('r7 uniform', r7({}))
    cands = os.environ.get('SP_CANDS')
    cands = eval(cands) if cands else [(0.15, (3, 20)), (0.3, (3, 20)), (0.3, (1, 3)), (0.6, (1, 3)),
                                       (1.0, (1, 3)), (0.3, (16, 64)), (0.6, (16, 64))]
    for sd, corr in cands:
        show(f'r7 texture sd {sd} corr {corr}', r7(dict(realbkg_ring_chi_texture=sd, realbkg_ring_chi_corr=corr)))
