"""What kind of real objects does a model miss, and what kind of predictions are wrong? CPU only, input = pickle(s) from heatmap/dump_dets.py.

  python heatmap/miss_analysis.py <dump.pkl> [<dump2.pkl> ...] [--sets 41 organic] [--thr 0.3]

For every ground-truth box (score threshold --thr, the evaluation's q-matcher at min IoU 0.1) it computes: ring / segment, width and height (px), position (q column,
chi row), contrast in the processed image the network sees (core mean minus surrounding-annulus mean, in annulus standard deviations), crowding (other GT centres
within 40 px), distance to the nearest invalid pixel, 'on ring' (a segment whose radial extent overlaps a ring GT). It prints the HIT RATE per quantile bin of each
feature, so a model that misses 'narrow', 'faint' or 'near the edge' objects shows it. For predictions it prints where the false positives sit
(on a GT ring's column, near a GT peak, elsewhere) per predicted class, and a per-frame table. With two or more pickles the tables are printed side by side
(first = reference, e.g. the legacy-sim model; second = the model being diagnosed)."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, uniform_filter
from heatmap import evaluation as E


def contrast(img, mask, b):
    H, W = img.shape
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, x1 = max(x0, 0), min(max(x1, x0 + 1), W); y0, y1 = max(y0, 0), min(max(y1, y0 + 1), H)
    core = img[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    if core.size == 0:
        return np.nan
    pad = 6
    X0, X1, Y0, Y1 = max(x0 - pad, 0), min(x1 + pad, W), max(y0 - pad, 0), min(y1 + pad, H)
    ann = np.ones((Y1 - Y0, X1 - X0), bool); ann[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    ann &= mask[Y0:Y1, X0:X1]
    a = img[Y0:Y1, X0:X1][ann]
    if a.size < 8:
        return np.nan
    return float((core.mean() - a.mean()) / max(a.std(), 1e-3))


def gt_table(frames, thr):
    rows = []
    for fi, f in enumerate(frames):
        gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
        if len(gt) == 0:
            continue
        img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool)
        ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else E.is_ring_geom(gt, mask)   # simulator frames carry the simulator's own ring flag
        sel = f['scores'] > thr
        b = f['boxes'][sel]; sc = f['scores'][sel]
        hit = np.zeros(len(gt), bool); tpscore = np.full(len(gt), np.nan)
        if len(b):
            _, ri, ci = E.matcher(torch.tensor(gt).float(), torch.tensor(b).float())
            hit[np.asarray(ri, int)] = True; tpscore[np.asarray(ri, int)] = sc[np.asarray(ci, int)]
        dist = distance_transform_edt(mask)
        cx = (gt[:, 0] + gt[:, 2]) / 2; cy = (gt[:, 1] + gt[:, 3]) / 2
        for i in range(len(gt)):
            on_ring = False
            if not ring[i] and ring.any():
                r = np.flatnonzero(ring)
                on_ring = bool(((gt[i, 0] < gt[r, 2]) & (gt[i, 2] > gt[r, 0])).any())
            near = int((np.hypot(cx - cx[i], cy - cy[i]) < 40).sum() - 1)
            rows.append(dict(frame=fi, ring=bool(ring[i]), w=float(gt[i, 2] - gt[i, 0]), h=float(gt[i, 3] - gt[i, 1]), x=float(cx[i]), y=float(cy[i]),
                             con=contrast(img, mask, gt[i]), crowd=near, edge=float(dist[int(np.clip(cy[i], 0, 511)), int(np.clip(cx[i], 0, 1023))]),
                             on_ring=on_ring, hit=bool(hit[i]), tp=float(tpscore[i])))
    return rows


def binned(rows, key, sel, nb=4):
    v = np.array([r[key] for r in rows if sel(r)], float); h = np.array([r['hit'] for r in rows if sel(r)], float)
    ok = np.isfinite(v)
    v, h = v[ok], h[ok]
    if len(v) < 8:
        return '   (too few)'
    q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
    out = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (v >= lo) & ((v < hi) if hi < q[-1] else (v <= hi))
        out.append(f'[{lo:.3g},{hi:.3g}] n={int(m.sum())} hit {h[m].mean():.2f}')
    return '  ' + ' | '.join(out)


def fp_table(frames, thr):
    cnt = {}
    nfr = 0
    for f in frames:
        gt = np.asarray(f['gt'], np.float32).reshape(-1, 4); mask = np.asarray(f['mask'], bool)
        sel = f['scores'] > thr
        b, c, sc = f['boxes'][sel], f['cls'][sel], f['scores'][sel]
        nfr += 1
        if len(b) == 0:
            continue
        matched = np.zeros(len(b), bool)
        if len(gt):
            _, ri, ci = E.matcher(torch.tensor(gt).float(), torch.tensor(b).float()); matched[np.asarray(ci, int)] = True
        ring = (np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else E.is_ring_geom(gt, mask)) if len(gt) else np.zeros(0, bool)
        gc = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], 1) if len(gt) else np.zeros((0, 2))
        pc = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        for j in np.flatnonzero(~matched):
            on_ring = bool(ring.any() and ((pc[j, 0] > gt[ring, 0]) & (pc[j, 0] < gt[ring, 2])).any())
            near = bool(len(gc) and (np.hypot(*(gc - pc[j]).T) < 8).any())
            k = ('cls ring' if c[j] == 1 else 'cls segment', 'on a GT ring column' if on_ring else ('near a GT peak (<8 px)' if near else 'elsewhere'))
            cnt[k] = cnt.get(k, 0) + 1
    return cnt, nfr


def main():
    p = argparse.ArgumentParser(); p.add_argument('dumps', nargs='+'); p.add_argument('--sets', nargs='*', default=['41', 'organic']); p.add_argument('--thr', type=float, default=0.3)
    a = p.parse_args()
    for pk in a.dumps:
        d = pickle.load(open(pk, 'rb'))
        print(f'\n################ {d["run"]} (epoch {d["epoch"]})  score > {a.thr}', flush=True)
        for ds in a.sets:
            if ds not in d['sets']:
                continue
            rows = gt_table(d['sets'][ds], a.thr)
            R = [r for r in rows if r['ring']]; S = [r for r in rows if not r['ring']]
            print(f'\n== {ds}: {len(rows)} GT ({len(R)} ring, {len(S)} segment)')
            print(f'  hit rate: all {np.mean([r["hit"] for r in rows]):.3f} | rings {np.mean([r["hit"] for r in R]) if R else float("nan"):.3f} | segments {np.mean([r["hit"] for r in S]):.3f}'
                  f" | segments on a ring {np.mean([r['hit'] for r in S if r['on_ring']]) if any(r['on_ring'] for r in S) else float('nan'):.3f} (n={sum(r['on_ring'] for r in S)})"
                  f" | isolated segments {np.mean([r['hit'] for r in S if not r['on_ring']]):.3f}")
            tps = [r['tp'] for r in rows if r['hit']]
            print(f'  median score of the matched predictions: rings {np.nanmedian([r["tp"] for r in R if r["hit"]]) if any(r["hit"] for r in R) else float("nan"):.2f}, segments {np.nanmedian([r["tp"] for r in S if r["hit"]]) if any(r["hit"] for r in S) else float("nan"):.2f}')
            for key, lab in (('w', 'box width px'), ('h', 'box height px'), ('x', 'q column'), ('y', 'chi row'), ('con', 'contrast (sd of the annulus)'), ('crowd', 'GT centres within 40 px'), ('edge', 'px to nearest invalid pixel')):
                print(f'  SEGMENT hit rate by {lab}:' + binned(rows, key, lambda r: not r['ring']))
            for key, lab in (('con', 'contrast'), ('x', 'q column'), ('w', 'box width px')):
                print(f'  RING hit rate by {lab}:' + binned(rows, key, lambda r: r['ring']))
            cnt, nfr = fp_table(d['sets'][ds], a.thr)
            print(f'  false positives per frame ({nfr} frames):  ' + ' | '.join(f'{k[0]}, {k[1]}: {v / nfr:.1f}' for k, v in sorted(cnt.items())))
            fr = {}
            for r in rows:
                fr.setdefault(r['frame'], []).append(r)
            if ds == '41':
                per = sorted(((np.mean([r['hit'] for r in v]), len(v), sum(r['ring'] for r in v), k) for k, v in fr.items()))
                print('  per-frame recall (worst 8): ' + ', '.join(f'f{k}:{h:.2f} (n={n}, rings {nr})' for h, n, nr, k in per[:8]))
                print('  per-frame recall (best 5):  ' + ', '.join(f'f{k}:{h:.2f} (n={n}, rings {nr})' for h, n, nr, k in per[-5:]))


if __name__ == '__main__':
    main()
