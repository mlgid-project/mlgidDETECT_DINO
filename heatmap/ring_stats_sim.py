"""Ring boxes of the real-background simulator vs the real 41 ground truth: width, position (q column), count per frame, height share. Needs GPU + ~13 GB RAM.

  python heatmap/ring_stats_sim.py --dump <dets_final.pkl> --n 150 --cfg "name:key=val;key=val" [--cfg ...]
Real 41 rings = the frame's GT boxes taller than 70 % of the valid height of their column (E.is_ring_geom). Prints quantiles (5/25/50/75/95 %) per source."""
import os, sys, types, argparse, ast, copy, pickle, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E

Q = (5, 25, 50, 75, 95)


def qs(v):
    return ' '.join(f'{x:7.1f}' for x in np.percentile(v, Q)) if len(v) else 'none'


def report(name, w, xc, h, nring, nseg, nframes, sw=None, sh=None):
    print(f'\n== {name}: {nframes} frames, rings/frame {nring / nframes:.1f}, segments/frame {nseg / nframes:.1f}')
    print(f'   quantiles 5/25/50/75/95 %:      {"  ".join(str(q).rjust(5) for q in Q)}')
    print(f'   ring box width px            {qs(w)}')
    print(f'   ring centre q column         {qs(xc)}')
    print(f'   ring box height px           {qs(h)}')
    if sw is not None and len(sw):
        print(f'   SEGMENT box width px         {qs(sw)}')
        print(f'   SEGMENT box height px        {qs(sh)}')
    for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 800), (800, 1100)):
        k = (xc >= lo) & (xc < hi)
        print(f'   rings with q column {lo:4d}-{hi:4d}: {k.sum() / nframes:5.2f} per frame, median width {np.median(w[k]) if k.any() else float("nan"):5.1f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dump', required=True); p.add_argument('--n', type=int, default=150); p.add_argument('--cfg', action='append', required=True)
    p.add_argument('--seed', type=int, default=5); p.add_argument('--preset', default='r6')
    p.add_argument('--init', default='', help="constructor options that cannot be set afterwards, e.g. \"mask_keep=[0,3,6,7]\"")
    a = p.parse_args()
    d = pickle.load(open(a.dump, 'rb'))['sets']['41']
    W, XC, H, nr, ns = [], [], [], 0, 0
    SW, SH = [], []
    for f in d:
        gt = np.asarray(f['gt'], np.float32).reshape(-1, 4); mask = np.asarray(f['mask'], bool)
        if not len(gt):
            continue
        ring = E.is_ring_geom(gt, mask)
        W += list((gt[ring, 2] - gt[ring, 0])); XC += list((gt[ring, 0] + gt[ring, 2]) / 2); H += list(gt[ring, 3] - gt[ring, 1])
        SW += list(gt[~ring, 2] - gt[~ring, 0]); SH += list(gt[~ring, 3] - gt[~ring, 1])
        nr += int(ring.sum()); ns += int((~ring).sum())
    report('real 41 GT', np.array(W), np.array(XC), np.array(H), nr, ns, len(d), np.array(SW), np.array(SH))
    from realsim.factory import make_realbkg_sim
    init = {k: ast.literal_eval(v) for k, v in (x.split('=', 1) for x in a.init.split(';') if x)}
    sim = make_realbkg_sim(device='cuda', preset=a.preset, overrides=init)
    base = {k: copy.deepcopy(getattr(sim, k)) for k in dir(sim) if not k.startswith('_') and isinstance(getattr(sim, k), (int, float, str, bool, tuple, type(None), dict)) and k not in ('device',)}
    for spec in a.cfg:
        name, _, kv = spec.partition(':')
        for k, v in base.items():
            setattr(sim, k, copy.deepcopy(v))
        for item in [x for x in kv.split(';') if x]:
            k, v = item.split('=', 1)
            if k.startswith('R_'):
                sim.R[k[2:]] = ast.literal_eval(v)
            else:
                setattr(sim, k, ast.literal_eval(v))
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        W, XC, H, nr, ns, n = [], [], [], 0, 0, 0
        SW, SH = [], []
        while n < a.n:
            try:
                out = sim.simulate_img()
            except Exception:
                continue
            if out is None:
                continue
            img, boxes, mask, is_ring = out
            b = boxes.float().cpu().numpy().reshape(-1, 4); r = is_ring.cpu().numpy().astype(bool)
            W += list(b[r, 2] - b[r, 0]); XC += list((b[r, 0] + b[r, 2]) / 2); H += list(b[r, 3] - b[r, 1])
            SW += list(b[~r, 2] - b[~r, 0]); SH += list(b[~r, 3] - b[~r, 1])
            nr += int(r.sum()); ns += int((~r).sum()); n += 1
        report(name, np.array(W), np.array(XC), np.array(H), nr, ns, n, np.array(SW), np.array(SH))


if __name__ == '__main__':
    main()
