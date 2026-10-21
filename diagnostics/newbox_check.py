"""Do the re-fitted validation sets (datasets/newbox/*_visfix.h5) load correctly through the real loaders?

User, 2026-10-21: stored radius_width / angle_width are 2*sigma, drawn box = centre +- width/2 = +-sigma;
visibility was copied back from the old files by peak id; rings may carry angle_width = inf (clamped by
util/pygidloader.py). This loads every set the way evaluation does (PyGIDDataset / H5GIWAXSDataset +
standard_preprocessing) and reports, per set:
  * frames, GT boxes used, geometric ring / segment split, is_ring-flag split (pygid files)
  * box sizes in px (segments: half-width radial / angular, percentiles) -- these should equal sigma;
    realbkg's measured priors are sigma_q median 3.9 px, sigma_chi median 20.3 px (57 real frames)
  * non-finite or out-of-image boxes, confidence mix
  * image checks: NaN fraction of the converted image, valid fraction of the mask, and for sets that are
    the same frames as an old file: max |image difference| over the valid pixels
CPU only; no model.   python diagnostics/newbox_check.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.exp_preprocess import standard_preprocessing

D = G.CUR
SETS = [
    ('organic OLD  organic_labeled.h5', f'{D}/organic_labeled.h5'),
    ('organic NEW  organics_visfix.h5', f'{D}/newbox/organics_visfix.h5'),
    ('41 OLD  41.h5 (what runs used)', f'{D}/41.h5'),
    ('41 OLD  41_pygid.h5', f'{D}/41_pygid.h5'),
    ('41 NEW  perovskites_nan_wedge_visfix.h5', f'{D}/newbox/perovskites_nan_wedge_visfix.h5'),
]
P = lambda v: 'n/a' if not len(v) else 'p10 %.2f p50 %.2f p90 %.2f' % tuple(np.percentile(v, [10, 50, 90]))


def load(path):
    cfg = G._config(path)
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=True)
          if detect_dataset_type(path) == 'pygid' else
          H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    fr = []
    try:
        for ic in ds.iter_images():
            pl = ic.polar_labels
            fr.append(dict(img=np.asarray(ic.converted_polar_image[:, 0], np.float32)[0],
                           mask=np.asarray(ic.converted_mask).reshape(*G.POLAR).astype(bool),
                           gt=np.asarray(pl.boxes, np.float32).reshape(-1, 4),
                           conf=np.asarray(pl.confidences, np.float32),
                           flag=(np.asarray(pl.is_ring, bool) if getattr(pl, 'is_ring', None) is not None and len(pl.is_ring) else None)))
    finally:
        if callable(getattr(ds, 'close', None)):
            ds.close()
    return fr


def report(name, fr):
    gt = np.concatenate([f['gt'] for f in fr]) if fr else np.zeros((0, 4))
    ring = np.concatenate([G.is_ring_geom(f['gt'], f['mask']) if len(f['gt']) else np.zeros(0, bool) for f in fr])
    flag = [f['flag'] for f in fr]
    hw = (gt[:, 2] - gt[:, 0]) / 2; hh = (gt[:, 3] - gt[:, 1]) / 2
    bad = (~np.isfinite(gt).all(1)).sum()
    out = ((gt[:, 0] < -1) | (gt[:, 2] > 1025) | (gt[:, 1] < -1) | (gt[:, 3] > 513)).sum() if len(gt) else 0
    if len(gt):
        ov = np.stack([-gt[:, 0], -gt[:, 1], gt[:, 2] - 1024, gt[:, 3] - 512], 1)      # px beyond left/top/right/bottom
        o_any = (ov > 1).any(1)
        print(f'  boxes sticking out >1 px: {int(o_any.sum())} (rings {int((o_any & ring).sum())}, segments {int((o_any & ~ring).sum())}) | '
              f'by side L/T/R/B {[int((ov[:, i] > 1).sum()) for i in range(4)]} | overhang px p50/p90/max '
              f'{np.percentile(ov.max(1)[o_any], 50) if o_any.any() else 0:.1f}/{np.percentile(ov.max(1)[o_any], 90) if o_any.any() else 0:.1f}/{ov.max():.1f}')
    conf = np.concatenate([f['conf'] for f in fr]) if fr else np.zeros(0)
    print(f'\n=== {name}\n  frames {len(fr)} | GT boxes {len(gt)} | geometric rings {int(ring.sum())} segments {int((~ring).sum())}'
          f'{"" if any(x is None for x in flag) else " | is_ring flag rings " + str(int(sum(x.sum() for x in flag)))}'
          f' | non-finite {int(bad)} | outside image {int(out)}')
    print(f'  segments half-width px: radial {P(hw[~ring])} | angular {P(hh[~ring])}')
    print(f'  rings    half-width px: radial {P(hw[ring])} | angular {P(hh[ring])}')
    print(f'  confidence 1.0/0.5/0.1: {[int((np.isclose(conf, c)).sum()) for c in (1.0, 0.5, 0.1)]}')
    nan = np.mean([np.isnan(f['img']).mean() for f in fr]); val = np.mean([f['mask'].mean() for f in fr])
    print(f'  image NaN fraction (mean) {nan:.4f} | mask valid fraction (mean) {val:.3f}')
    return fr


if __name__ == '__main__':
    res = {}
    for name, path in SETS:
        if not os.path.exists(path):
            print(f'\n=== {name}: MISSING {path}'); continue
        try:
            res[name] = report(name, load(path))
        except Exception as e:                      # keep going so one bad set does not hide the rest
            import traceback; print(f'\n=== {name}: FAILED {e!r}'); traceback.print_exc()
    for a, b in (('organic OLD  organic_labeled.h5', 'organic NEW  organics_visfix.h5'),
                 ('41 OLD  41_pygid.h5', '41 NEW  perovskites_nan_wedge_visfix.h5'),
                 ('41 OLD  41.h5 (what runs used)', '41 NEW  perovskites_nan_wedge_visfix.h5')):
        if a in res and b in res and len(res[a]) == len(res[b]):
            d = []
            for fa, fb in zip(res[a], res[b]):
                m = fa['mask'] & fb['mask'] & np.isfinite(fa['img']) & np.isfinite(fb['img'])
                d.append(float(np.abs(fa['img'] - fb['img'])[m].max()) if m.any() else np.nan)
            print(f'\nimage diff  [{a}]  vs  [{b}]: max |diff| per frame {P(np.array(d)[np.isfinite(d)])} (frames {len(d)}); '
                  f'mask valid-fraction difference mean {np.mean([abs(x["mask"].mean() - y["mask"].mean()) for x, y in zip(res[a], res[b])]):.4f}')
