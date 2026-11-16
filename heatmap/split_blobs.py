"""Post-processing check: split one detection into two where the raw pixels say a close pair was merged into one blob.
No network change, no training. CPU only, input = the pickle from heatmap/dump_dets.py (+nms detections, floor 0.1, top-225).

  python heatmap/split_blobs.py <dump.pkl> [--sets sim] [--tau 30] [--sep_min 1.5] [--sep_max 8] [--ratio_min 0.3] [--score_min 0.2]
  python heatmap/split_blobs.py <dump.pkl> --sweep_tau 10 20 30 50 80     (sim and 41: how many splits / close-pair gain / extra false positives per tau)
  python heatmap/split_blobs.py <dump.pkl> --final --sets organic      (final test set, ONE evaluation with the parameters given; see the rule below)

Rule (to keep the final test honest): the parameters are chosen on the development sets 'sim' (simulator frames; NOTE they contain no GT pair closer than
5 px, only 5-10 px ones) and '41' (the real mixed-frame set). 'organic' is the final test set: it is only touched with --final, once, with the parameters
printed in the output; no tuning on it. (On sim the GT confidences are set to 1.0/0.5/0.1 in turn only so that the evaluator's AP strata exist; the
strata are meaningless there.)

Method, per segment detection with score >= score_min (rings and ring-like tall boxes are skipped):
  1. patch = the box grown by 25 %, at most 41 x 41 px, invalid pixels (mask) ignored;
  2. M1 = background + one axis-aligned 2D Gaussian (6 parameters); M2 = background + two Gaussians with a shared width (9 parameters),
     started from M1 shifted along the broader axis (4 starts, best kept); least squares (scipy);
  3. split if BIC(M1) - BIC(M2) > tau, and the two fitted centres are sep_min..sep_max px apart, and the weaker amplitude is >= ratio_min of the stronger;
  4. the detection is replaced by two boxes of the same size at the fitted centres; score_i = score * A_i / max(A_1, A_2) (the stronger keeps the score).
Output: baseline and split result with the evaluation's own numbers (AP, recall/precision/ring recall at score > 0.3, recall by NN distance)."""
import os, sys, types, argparse, pickle, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
from scipy.optimize import least_squares
from heatmap import evaluation as E


def g2(xx, yy, a, x0, y0, sx, sy):
    return a * np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2))


def fit_patch(img, valid, box, grow=1.25, max_half=20):
    """Returns None or dict(dbic, c1 [x,y], c2 [x,y], a1, a2, sep) in full-image pixel coordinates."""
    H, W = img.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hx = int(np.clip(round((box[2] - box[0]) / 2 * grow), 4, max_half)); hy = int(np.clip(round((box[3] - box[1]) / 2 * grow), 4, max_half))
    x0, x1 = int(max(cx - hx, 0)), int(min(cx + hx + 1, W)); y0, y1 = int(max(cy - hy, 0)), int(min(cy + hy + 1, H))
    if x1 - x0 < 7 or y1 - y0 < 7:
        return None
    p = img[y0:y1, x0:x1].astype(np.float64); v = valid[y0:y1, x0:x1]
    if v.sum() < 30:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xv, yv, pv = xx[v], yy[v], p[v]
    n = len(pv)
    bg0, a0 = float(np.percentile(pv, 10)), float(pv.max() - np.percentile(pv, 10))
    if a0 <= 1e-6:
        return None
    sx0, sy0 = max((x1 - x0) / 5.0, 1.0), max((y1 - y0) / 5.0, 1.0)
    lo1 = [-np.inf, 0, x0, y0, 0.7, 0.7]; hi1 = [np.inf, np.inf, x1, y1, (x1 - x0), (y1 - y0)]
    r1 = least_squares(lambda t: t[0] + g2(xv, yv, *t[1:]) - pv, [bg0, a0, cx, cy, sx0, sy0], bounds=(lo1, hi1), max_nfev=60)
    rss1 = float((r1.fun ** 2).sum()); t1 = r1.x
    best = None
    for dx, dy in ((0.8, 0), (0, 0.8), (0.6, 0.6), (0.6, -0.6)):
        s = [t1[0], t1[1] / 2, t1[1] / 2, t1[2] - dx * t1[4], t1[3] - dy * t1[5], t1[2] + dx * t1[4], t1[3] + dy * t1[5], t1[4], t1[5]]
        lo = [-np.inf, 0, 0, x0, y0, x0, y0, 0.7, 0.7]; hi = [np.inf, np.inf, np.inf, x1, y1, x1, y1, (x1 - x0), (y1 - y0)]
        s = np.clip(s, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            r = least_squares(lambda t: t[0] + g2(xv, yv, t[1], t[3], t[4], t[7], t[8]) + g2(xv, yv, t[2], t[5], t[6], t[7], t[8]) - pv,
                              s, bounds=(lo, hi), max_nfev=60)
        except Exception:
            continue
        rss = float((r.fun ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, r.x)
    if best is None:
        return None
    rss2, t2 = best
    bic1 = n * np.log(max(rss1, 1e-12) / n) + 6 * np.log(n); bic2 = n * np.log(max(rss2, 1e-12) / n) + 9 * np.log(n)
    c1, c2 = np.array([t2[3], t2[4]]), np.array([t2[5], t2[6]])
    return dict(dbic=bic1 - bic2, c1=c1, c2=c2, a1=float(t2[1]), a2=float(t2[2]), sep=float(np.hypot(*(c1 - c2))))


def fits_frame(f, score_min):
    """One fit per eligible detection (the expensive part, independent of tau / sep / ratio): list of None or fit dict, per detection."""
    b, s, c = f['boxes'], f['scores'], f['cls']
    img = np.asarray(f['img'], np.float32); valid = np.asarray(f['mask'], bool)
    colvalid = valid.sum(0)
    out = []
    for k in range(len(b)):
        bk = b[k]
        ok = c[k] == 0 and s[k] >= score_min and (bk[3] - bk[1]) < 0.7 * max(int(colvalid[int(np.clip((bk[0] + bk[2]) / 2, 0, img.shape[1] - 1))]), 1)
        out.append(fit_patch(img, valid, bk) if ok else None)
    return out


def split_frame(f, fits, tau, sep_min, sep_max, ratio_min, stats):
    b, s = f['boxes'], f['scores']
    ob, os_ = [], []
    for k in range(len(b)):
        r = fits[k]; bk = b[k]
        stats['fitted'] += int(r is not None)
        if r is not None and r['dbic'] > tau and sep_min <= r['sep'] <= sep_max and min(r['a1'], r['a2']) >= ratio_min * max(r['a1'], r['a2']):
            w, h = bk[2] - bk[0], bk[3] - bk[1]; am = max(r['a1'], r['a2'])
            for cc, aa in ((r['c1'], r['a1']), (r['c2'], r['a2'])):
                ob.append([cc[0] - w / 2, cc[1] - h / 2, cc[0] + w / 2, cc[1] + h / 2]); os_.append(s[k] * aa / am)
            stats['splits'] += 1
        else:
            ob.append(bk); os_.append(s[k])
    return np.asarray(ob, np.float32).reshape(-1, 4), np.asarray(os_, np.float32)


def run_set(frames, fits, params, label):
    stats = dict(fitted=0, splits=0)
    base, new, gts = [], [], []
    for f, ft in zip(frames, fits):
        base.append((np.asarray(f['boxes'], np.float32).reshape(-1, 4), np.asarray(f['scores'], np.float32)))
        new.append(split_frame(f, ft, stats=stats, **params))
        gts.append(dict(gt=np.asarray(f['gt'], np.float32).reshape(-1, 4), gtconf=np.asarray(f['gtconf'], np.float32), mask=np.asarray(f['mask'], bool)))
    rb = E.evaluate_dets(base, gts, None); rn = E.evaluate_dets(new, gts, None)
    print(f'\n== {label}: {len(frames)} frames, {stats["fitted"]} detections fitted, {stats["splits"]} split', flush=True)
    print(E.format_result('baseline +nms', label, rb), flush=True)
    print(E.format_result('with blob splitting', label, rn), flush=True)
    return rb, rn, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dump'); p.add_argument('--sets', nargs='*', default=['sim'])
    p.add_argument('--tau', type=float, default=30.0); p.add_argument('--sep_min', type=float, default=1.5); p.add_argument('--sep_max', type=float, default=8.0)
    p.add_argument('--ratio_min', type=float, default=0.3); p.add_argument('--score_min', type=float, default=0.2)
    p.add_argument('--sweep_tau', type=float, nargs='*'); p.add_argument('--final', action='store_true',
                   help='allow the real sets (organic, 41): one evaluation with the given parameters, no tuning')
    a = p.parse_args()
    d = pickle.load(open(a.dump, 'rb'))
    real = [s for s in a.sets if s not in ('sim', '41')]
    if real and not a.final:
        sys.exit('refusing to run on the final test set (organic) without --final (choose the parameters on sim and 41 first)')
    for f in d['sets'].get('sim', []):
        f['gtconf'] = np.array([(1.0, 0.5, 0.1)[i % 3] for i in range(len(f['gt']))], np.float32)
    print(f'{d["run"]} (epoch {d["epoch"]})', flush=True)
    for ds in a.sets:
        t0 = time.time()
        fits = [fits_frame(f, a.score_min) for f in d['sets'][ds]]
        print(f'{ds}: fitted {sum(x is not None for fr in fits for x in fr)} detections in {time.time() - t0:.0f} s', flush=True)
        taus = a.sweep_tau if (a.sweep_tau and ds != 'organic') else [a.tau]
        for tau in taus:
            prm = dict(tau=tau, sep_min=a.sep_min, sep_max=a.sep_max, ratio_min=a.ratio_min)
            print(f'parameters: {prm}, score_min {a.score_min}', flush=True)
            run_set(d['sets'][ds], fits, prm, ds)


if __name__ == '__main__':
    main()
