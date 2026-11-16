"""Are 41's labels complete? Measured the same way for both gates.

Why this matters: the recorded physics-sim failure on 41 is PRECISION AT EQUAL RECALL -- 507 false
positives sitting a median 74 px from any real peak -- while physics models BEAT the baseline on
organic recall by +0.20. There is a note establishing that ORGANIC has no unlabelled peaks. No
equivalent check was ever run on 41. If 41 carries real peaks its annotators did not mark, then a
model that finds more peaks is penalised for being right, and a less thorough simulator scores
better on that gate for the wrong reason.

Method, no model involved: find local maxima in the preprocessed image, score each one against its
own local background (median and MAD of a surrounding annulus), and ask what fraction of the
significant ones fall outside every ground-truth box. Reported across a range of thresholds because
the answer should not hinge on one cut. Also reports what fraction of GT boxes contain a maximum at
all, which is the sanity check on the detector itself: if labels sit on maxima, the detector works.
"""
import os, sys
import numpy as np
from scipy.ndimage import maximum_filter
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset

SETS = (('organic', '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'),
        ('41',      '/mnt/lustre/work/schreiber/szb389/datasets/41.h5'))
THR = (2.0, 3.0, 4.0, 5.0, 6.0)
NBH = 5          # local-maximum neighbourhood (px)
ANN = 12         # annulus half-size for the local background

def local_z(img, mask, ys, xs):
    H, W = img.shape
    z = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs)):
        r0, r1 = max(y-ANN, 0), min(y+ANN, H-1)+1
        c0, c1 = max(x-ANN, 0), min(x+ANN, W-1)+1
        win = img[r0:r1, c0:c1]; wm = mask[r0:r1, c0:c1].copy()
        # exclude the 5x5 core so the peak cannot set its own background
        wm[max(y-2-r0, 0):y+3-r0, max(x-2-c0, 0):x+3-c0] = False
        v = win[wm]
        if v.size < 20:
            continue
        bkg = np.median(v); mad = np.median(np.abs(v-bkg))*1.4826
        if mad <= 0:
            continue
        z[i] = (img[y, x]-bkg)/mad
    return z

for name, path in SETS:
    cfg = Config(); cfg.INPUT_DATASET = path
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1; cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    tot_cand = {t: 0 for t in THR}
    out_cand = {t: 0 for t in THR}
    gt_tot = gt_hit = nfr = 0
    for ic in ds.iter_images():
        img = np.asarray(ic.converted_polar_image)[0, 0].astype(float)
        mask = img > 1e-6
        b = np.asarray(ic.polar_labels.boxes, float)
        if b.ndim != 2 or len(b) == 0:
            continue
        nfr += 1
        peak = (img == maximum_filter(img, size=NBH)) & mask & (img > 0)
        ys, xs = np.nonzero(peak)
        if len(ys) > 20000:                      # keep the brightest if a frame is pathological
            k = np.argsort(-img[ys, xs])[:20000]; ys, xs = ys[k], xs[k]
        z = local_z(img, mask, ys, xs)
        ok = np.isfinite(z)
        ys, xs, z = ys[ok], xs[ok], z[ok]
        # inside ANY gt box?
        inside = np.zeros(len(ys), bool)
        for x0, y0, x1, y1 in b:
            inside |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        for t in THR:
            s = z >= t
            tot_cand[t] += int(s.sum()); out_cand[t] += int((s & ~inside).sum())
        # does each GT box contain a maximum at all?
        gt_tot += len(b)
        for x0, y0, x1, y1 in b:
            gt_hit += int(((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).any())
    print(f'=== {name}: {nfr} frames, {gt_tot} GT boxes')
    print(f'    GT boxes containing a local maximum: {100*gt_hit/max(gt_tot,1):.1f}%  '
          f'(detector sanity -- labels should sit on maxima)')
    print(f'    {"z thr":>6} {"maxima":>8} {"outside all GT":>15} {"share":>8} {"per frame":>10}')
    for t in THR:
        print(f'    {t:>6.1f} {tot_cand[t]:>8d} {out_cand[t]:>15d} '
              f'{100*out_cand[t]/max(tot_cand[t],1):>7.1f}% {out_cand[t]/max(nfr,1):>10.1f}')
    print()
print('If 41 shows many significant maxima outside every box and organic does not, then 41 is')
print('under-labelled and part of its "precision" gap is the gate, not the model.')
