"""Why are close-pair peaks missed? Splitting one blob into two can only help where the detector put ONE box on a pair.

  python heatmap/blob_diag.py <dump.pkl> [--sets sim organic 41] [--thr 0.3] [--near 6]

For every ground-truth segment whose nearest other GT segment is closer than 5 px (and 5-10 px), at score > thr, after the same matcher
as the evaluation (heatmap.evaluation.matcher), it says which case applies:
  hit                      the GT peak is matched
  missed, neighbour hit    its close neighbour is matched but it is not: typically ONE detection for the pair (a splitting candidate)
  missed, both missed, a detection within --near px of it: a box sits on the pair but matched neither (localisation / box shape)
  missed, nothing near     no detection (score > thr) within --near px: the detector did not fire at all (splitting cannot help)
CPU only. Input: the pickle from heatmap/dump_dets.py."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E


def centres(b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def matched(gt, b):
    hit = np.zeros(len(gt), bool)
    if len(b) and len(gt):
        _, ri, _ = E.matcher(torch.tensor(gt).float(), torch.tensor(b).float())
        hit[np.asarray(ri, int)] = True
    return hit


def diag(frames, thr, near):
    cnt = {}
    for lo, hi, nm in ((0, 5, '<5 px'), (5, 10, '5-10 px')):
        c = dict(n=0, hit=0, miss_nb_hit=0, miss_both_near=0, miss_nothing=0)
        for f in frames:
            gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
            if len(gt) == 0:
                continue
            k = (f['scores'] > thr) & (f['cls'] == 0)
            b = f['boxes'][k]
            hit = matched(gt, f['boxes'][f['scores'] > thr])
            eu, _, ring = E.nn_distances(gt, f['mask'])
            gc = centres(gt); dc = centres(b)
            seg = np.where(~ring)[0]
            for i in np.where((eu >= lo) & (eu < hi))[0]:
                c['n'] += 1
                if hit[i]:
                    c['hit'] += 1; continue
                o = seg[seg != i]
                j = o[np.argmin(np.hypot(*(gc[o] - gc[i]).T))] if len(o) else None
                dmin = np.hypot(*(dc - gc[i]).T).min() if len(dc) else np.inf
                if j is not None and hit[j]:
                    c['miss_nb_hit'] += 1
                elif dmin <= near:
                    c['miss_both_near'] += 1
                else:
                    c['miss_nothing'] += 1
        cnt[nm] = c
    return cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dump'); p.add_argument('--sets', nargs='*'); p.add_argument('--thr', type=float, default=0.3)
    p.add_argument('--near', type=float, default=6.0)
    a = p.parse_args()
    d = pickle.load(open(a.dump, 'rb'))
    print(f'{d["run"]} (epoch {d["epoch"]}), score > {a.thr}, "near" = {a.near} px')
    for ds in (a.sets or list(d['sets'])):
        for nm, c in diag(d['sets'][ds], a.thr, a.near).items():
            n = max(c['n'], 1)
            print(f'  {ds:7s} NN {nm:8s}: {c["n"]:4d} GT | hit {c["hit"]} ({c["hit"] / n:.3f}) | missed, neighbour hit {c["miss_nb_hit"]} ({c["miss_nb_hit"] / n:.3f}) | '
                  f'missed, both missed, box near {c["miss_both_near"]} ({c["miss_both_near"] / n:.3f}) | missed, nothing near {c["miss_nothing"]} ({c["miss_nothing"] / n:.3f})')


if __name__ == '__main__':
    main()
