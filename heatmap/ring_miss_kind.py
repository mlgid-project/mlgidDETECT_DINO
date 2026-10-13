"""Why are real 41 rings missed? For every GT ring (geometric criterion) that is NOT matched at score > thr: is there a detection of class ring / segment centred within --dx px in q, and
how do its height / IoU compare with the GT box? Split at --x_split. CPU only; input = dump_dets pickle(s).
  python heatmap/ring_miss_kind.py <dump.pkl> [<dump2.pkl>] [--thr 0.3] [--dx 10] [--x_split 781]"""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def main():
    p = argparse.ArgumentParser(); p.add_argument('dumps', nargs='+'); p.add_argument('--thr', type=float, default=0.3); p.add_argument('--dx', type=float, default=10.0); p.add_argument('--x_split', type=float, default=781)
    a = p.parse_args()
    for path in a.dumps:
        d = pickle.load(open(path, 'rb'))
        print(f'\n#### {d["run"]} (epoch {d["epoch"]}), score > {a.thr}')
        res = {'x<': dict(n=0, hit=0, ring_near=0, seg_near=0, none=0, ious=[], hdet=[], hgt=[]), 'x>=': dict(n=0, hit=0, ring_near=0, seg_near=0, none=0, ious=[], hdet=[], hgt=[])}
        for f in d['sets']['41']:
            gt = np.asarray(f['gt'], np.float32).reshape(-1, 4); mask = np.asarray(f['mask'], bool)
            if not len(gt):
                continue
            k = f['scores'] > a.thr; b = f['boxes'][k]; c = f['cls'][k]
            ring = E.is_ring_geom(gt, mask)
            hit = np.zeros(len(gt), bool)
            if len(b):
                _, ri, _ = E.matcher(torch.tensor(gt).float(), torch.tensor(b).float()); hit[np.asarray(ri, int)] = True
            for i in np.where(ring)[0]:
                g = gt[i]; xc = (g[0] + g[2]) / 2; r = res['x>=' if xc >= a.x_split else 'x<']
                r['n'] += 1
                if hit[i]:
                    r['hit'] += 1; continue
                near = np.where(np.abs((b[:, 0] + b[:, 2]) / 2 - xc) <= a.dx)[0] if len(b) else []
                if len(near) == 0:
                    r['none'] += 1; continue
                j = near[np.argmax([iou(g, b[m]) for m in near])]
                (r.__setitem__('ring_near', r['ring_near'] + 1) if c[j] == 1 else r.__setitem__('seg_near', r['seg_near'] + 1))
                r['ious'].append(iou(g, b[j])); r['hdet'].append(b[j][3] - b[j][1]); r['hgt'].append(g[3] - g[1])
        for nm, r in res.items():
            n = max(r['n'], 1)
            print(f'  GT rings {nm} {a.x_split:g}: {r["n"]} | matched {r["hit"]} ({r["hit"] / n:.2f}) | missed: ring-class det within {a.dx:g} px but unmatched {r["ring_near"]} ({r["ring_near"] / n:.2f}) | '
                  f'segment-class det near {r["seg_near"]} ({r["seg_near"] / n:.2f}) | nothing near {r["none"]} ({r["none"] / n:.2f})')
            if r['ious']:
                print(f'      of the unmatched-but-near: median IoU {np.median(r["ious"]):.3f}, median det height {np.median(r["hdet"]):.0f} px vs GT height {np.median(r["hgt"]):.0f} px')


if __name__ == '__main__':
    main()
