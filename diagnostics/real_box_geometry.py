"""Geometry of the REAL labelled boxes on both eval gates, in the 512x1024 polar frame.

The simulator gives every ring a full-height box (`bx[rg,1]=0; bx[rg,3]=HEIGHT`). This asks what
the real ring boxes actually do, because the detector wedge means the valid chi span at a given q
is usually well short of all 512 rows -- so a full-height ring box can be systematically too tall
on the very class that dominates 41.

Rings are identified by GEOMETRY (box covers >=70% of the valid chi span at its q), not by
`is_ring`: that field is not populated for 41.h5 and defaults to all-False, which reports 41 as
ring-free. See the eval-dataset-facts note.

Loads through the same PyGIDDataset / H5GIWAXSDataset path main.py evaluates with. No model.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset

SETS = {'organic': '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5',
        '41':      '/mnt/lustre/work/schreiber/szb389/datasets/41.h5'}
RING_FRAC = 0.70

def load(p):
    cfg = Config()
    cfg.INPUT_DATASET = p
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    if detect_dataset_type(p) == 'pygid':
        return PyGIDDataset(cfg, path=p, preprocess_func=standard_preprocessing,
                            buffer_size=5, load_labels=True)
    return H5GIWAXSDataset(cfg, path=p, preprocess_func=standard_preprocessing, buffer_size=5)

def pct(x, q):
    return np.percentile(x, q) if len(x) else float('nan')

for name, path in SETS.items():
    ds = load(path)
    H = 512
    rh, rspan, rcov, sh, sw, nfr = [], [], [], [], [], 0
    for ic in ds.iter_images():
        img = np.asarray(ic.converted_polar_image)[0, 0]
        valid = img > 1e-6
        span = valid.sum(0).astype(float)                 # valid chi rows per q column
        b = np.asarray(ic.polar_labels.boxes, float)
        if b.ndim != 2 or len(b) == 0:
            continue
        nfr += 1
        xc = np.clip(((b[:, 0] + b[:, 2]) / 2).astype(int), 0, img.shape[1] - 1)
        h = b[:, 3] - b[:, 1]
        w = b[:, 2] - b[:, 0]
        s = np.maximum(span[xc], 1.0)
        isring = h >= RING_FRAC * s
        rh += list(h[isring]); rspan += list(s[isring]); rcov += list((h / s)[isring])
        sh += list(h[~isring]); sw += list(w[~isring])
    rh, rspan, rcov = map(np.array, (rh, rspan, rcov))
    sh, sw = np.array(sh), np.array(sw)
    n = len(rh) + len(sh)
    print(f'=== {name}: {nfr} frames, {n} boxes, {len(rh)} rings ({100*len(rh)/max(n,1):.0f}%), '
          f'{len(sh)} segments   [ring = height >= {RING_FRAC:.0%} of valid chi span]')
    print(f'  ring box HEIGHT px        p10 {pct(rh,10):6.0f}  p50 {pct(rh,50):6.0f}  '
          f'p90 {pct(rh,90):6.0f}  max {rh.max() if len(rh) else float("nan"):6.0f}')
    print(f'  valid chi span at its q   p10 {pct(rspan,10):6.0f}  p50 {pct(rspan,50):6.0f}  '
          f'p90 {pct(rspan,90):6.0f}')
    print(f'  ring height / 512         p10 {pct(rh/H,10):6.2f}  p50 {pct(rh/H,50):6.2f}  '
          f'p90 {pct(rh/H,90):6.2f}')
    print(f'  ring height / valid span  p10 {pct(rcov,10):6.2f}  p50 {pct(rcov,50):6.2f}  '
          f'p90 {pct(rcov,90):6.2f}')
    if len(rh):
        print(f'  rings that are FULL height (>=505 px): {100*np.mean(rh >= 505):4.1f}%   '
              f'>= 0.95 of valid span: {100*np.mean(rcov >= 0.95):4.1f}%')
    if len(rh):
        # A full-height (512 px) PREDICTION nested in a GT ring box of height h scores IoU = h/512
        # at the same q and width, so GT rings shorter than 256 px cannot be matched at IoU 0.5 by
        # a model the simulator taught to draw rings full height.
        for thr in (0.5, 0.75):
            bad = np.mean(rh < thr*H)
            print(f'  GT rings a 512-px prediction CANNOT match at IoU {thr}: {100*bad:4.1f}% of rings'
                  f'  = {100*bad*len(rh)/max(n,1):4.1f}% of all boxes on this set')
    print(f'  segment box  w px         p10 {pct(sw,10):6.1f}  p50 {pct(sw,50):6.1f}  p90 {pct(sw,90):6.1f}')
    print(f'  segment box  h px         p10 {pct(sh,10):6.1f}  p50 {pct(sh,50):6.1f}  p90 {pct(sh,90):6.1f}')
    print()
print('SIMULATOR, for comparison: every ring box is y=0..512, i.e. height 512 and')
print('height/512 = 1.00 regardless of the valid chi span at that q.')
