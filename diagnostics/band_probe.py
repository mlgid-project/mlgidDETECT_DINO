"""Do the realbkg training BACKGROUNDS carry unlabelled ring-like bands?

blur41_probe.py: blurring real 41 radially makes the realbkg model WORSE, not better (conv3 41 AP
0.380 -> 0.262 -> 0.114 at sigma 0/1.5/3 px, ring recall 0.776 -> 0.570 -> 0.280), while ring_fp
shows it never emits full-height ring boxes on 41 -- it sprays fragments over ring regions. The
working hypothesis: the model learned that a smooth chi-invariant band is BACKGROUND. In the polar
image any radial structure of the background (the donor envelope, Si/air scatter, and in v1 the
rejected-bank envelope with smeared real diffraction) is exactly such a band, and it never has a box.

Per frame, on the image the network sees:
  b(x)  = median over the BACKGROUND pixels of column x (valid mask minus every GT box, +-3 px)
  hp(x) = b(x) - running median of b over 41 px            (radial structure, offset removed)
  bands = peaks of hp with prominence >= T and FWHM <= 30 px  -> unlabelled ring-like features
reported per frame for T = 0.05 and 0.10 (real 41 labelled rings: contrast p10 0.07, p50 0.22).

Sources: real 41 / organic, legacy, physics4, realbkg v1 AS conv3 TRAINED IT (r3 config, envelope
from the rejected bank, i.e. USE_ENV_BANK=True), realbkg v2 (run 6). CPU only.
Caveat: 41's labels were never audited for completeness, so its own count may include real rings
nobody boxed -- read it as an upper bound on what the real data looks like.
"""
import os, sys
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_shape_compare as PS
import random, torch

N = int(os.environ.get('BAND_FRAMES', 120))


def band_stats(img, mask, gt):
    bg = mask.copy()
    for x0, y0, x1, y1 in gt:
        bg[int(max(0, np.floor(y0))):int(np.ceil(y1)) + 1, int(max(0, np.floor(x0) - 3)):int(np.ceil(x1)) + 4] = False
    vcol = mask.sum(0)
    ok = (bg.sum(0) >= np.maximum(0.3 * vcol, 20))
    if ok.sum() < 100:
        return None
    b = np.full(img.shape[1], np.nan)
    for x in np.where(ok)[0]:
        b[x] = np.median(img[bg[:, x], x])
    xs = np.arange(len(b))
    bi = np.interp(xs, xs[ok], b[ok])
    hp = bi - median_filter(bi, size=41, mode='nearest')
    hp[~ok] = 0.0
    out = {'std': float(np.std(hp[ok]))}
    for T in (0.05, 0.10):
        pk, pr = find_peaks(hp, prominence=T, width=(1, 30), rel_height=0.5)
        out[T] = int(ok[pk].sum())
    return out


def realbkg_v1_as_trained(n):
    from realbkg_sim.mosaic_background import MosaicBackground
    MosaicBackground.USE_ENV_BANK = True        # what conv1-conv5 trained on (fixed in 02de206)
    return PS.realbkg6(n, config='config/DINO/DINO_4scale_swin_realbkg_r3.py')


def realbkg_v2(n):
    from realbkg_sim.mosaic_background import MosaicBackground
    MosaicBackground.USE_ENV_BANK = False
    return PS.realbkg6(n)


SRC = [('real41', lambda: PS.real(f'{PS.CUR}/41.h5', False)),
       ('organic', lambda: PS.real(f'{PS.CUR}/organic_labeled.h5', True)),
       ('legacy', lambda: PS.legacy(N)),
       ('physics4', lambda: PS.physics4(N)),
       ('realbkg_v1_conv3', lambda: realbkg_v1_as_trained(N)),
       ('realbkg_v2_run6', lambda: realbkg_v2(N))]

if __name__ == '__main__':
    only = os.environ.get('BAND_ONLY')
    print('unlabelled radial bands in the BACKGROUND, per frame (processed image, 0-1)')
    for name, fn in SRC:
        if only and name not in only.split(','):
            continue
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        st, rings = [], []
        for img, m, gt in fn():
            s = band_stats(img, m, gt)
            if s is not None:
                st.append(s); rings.append(int(PS.is_ring_geom(gt, m).sum()) if len(gt) else 0)
        a05 = np.array([s[0.05] for s in st]); a10 = np.array([s[0.10] for s in st])
        sd = np.array([s['std'] for s in st])
        print(f'{name:<18} frames {len(st):4d} | labelled rings/frame {np.mean(rings):5.1f} | '
              f'unlabelled bands/frame  prom>=0.05: mean {a05.mean():5.2f} p90 {np.percentile(a05,90):4.0f}  '
              f'prom>=0.10: mean {a10.mean():5.2f} p90 {np.percentile(a10,90):4.0f} | '
              f'frames with >=1 band(0.10) {np.mean(a10 > 0):4.0%} | radial hp std p50 {np.median(sd):.4f}',
              flush=True)
