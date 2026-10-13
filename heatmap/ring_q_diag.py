"""Where do the simulator's rings sit along q, and why? Draws powder entries from the hkl bank and reports (a) the entry's own q range / reflection count,
(b) the frame q_max values the simulator uses, (c) the resulting ring x position = q / q_max * 1024 if q < 0.995 q_max, as drawn by _draw_peaks.
  python heatmap/ring_q_diag.py [--n 3000]
Needs the bank (13 GB RAM, GPU not required but the factory builds on cuda)."""
import os, sys, types, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch

P = (5, 25, 50, 75, 95)


def pq(v):
    return ' '.join(f'{x:7.2f}' for x in np.percentile(v, P))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--n', type=int, default=3000); a = a.parse_args()
    from realsim.factory import make_realbkg_sim
    sim = make_realbkg_sim(device='cuda')
    ids = sim.phys.powder_ids
    print(f'{len(ids)} powder entries, {len(sim.phys.oriented_ids)} oriented entries')
    qmax_all = np.asarray(sim.qmax, float)
    print(f'frame q_max (donor frames): quantiles 5/25/50/75/95 %: {pq(qmax_all)}')
    random.seed(0); np.random.seed(0)
    qlo, qhi, nref, xs = [], [], [], []
    for _ in range(a.n):
        q, _c, I = sim.phys._entry(int(np.random.choice(ids)))
        q = np.asarray(q.cpu() if torch.is_tensor(q) else q, float)
        qlo.append(q.min()); qhi.append(q.max()); nref.append(len(q))
        qm = float(np.random.choice(qmax_all))
        v = q < qm * 0.995
        if v.sum():
            k = min(int(v.sum()), random.randint(*sim.rings_cap))
            idx = np.random.choice(np.where(v)[0], size=k, replace=False)
            xs += list(q[idx] / qm * 1024)
    xs = np.array(xs)
    print(f'entry q min   (A^-1)  5/25/50/75/95 %: {pq(qlo)}')
    print(f'entry q max   (A^-1)  5/25/50/75/95 %: {pq(qhi)}')
    print(f'entry reflections     5/25/50/75/95 %: {pq(nref)}')
    print(f'ring x position (px)  5/25/50/75/95 %: {pq(xs)}')
    for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 800), (800, 1100)):
        print(f'  share of rings with x {lo:4d}-{hi:4d}: {((xs >= lo) & (xs < hi)).mean():.3f}')
    print('real 41 for comparison (ring centres): median 571, shares  0-200 0.106 | 200-400 0.179 | 400-600 0.257 | 600-800 0.238 | 800+ 0.218')


if __name__ == '__main__':
    main()
