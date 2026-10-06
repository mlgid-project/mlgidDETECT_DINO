"""Targets for the realbkg ring fix: per-frame ring counts on the real gates, and what the bank's
powder entries can supply.

realbkg-41-gap-split found the realbkg models fragment 41/legacy-style rings. Before changing the
ring draw, measure:
  * the per-frame distribution of geometric rings on real 41 and organic (not just the mean), and
    of the current realbkg r6 sim and the legacy sim, so a wide per-frame draw can be matched
  * how many reflections a powder entry has below the frame's q_max -- whether 1-3 entries can
    produce 41's dense forests (frame 33: ~45 rings) at all
  * the radial spacing between neighbouring rings in a frame, real vs sim
CPU only; loads the hkl bank.
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_shape_compare as PS
import torch


def rings_per_frame(frames):
    n, gaps, widths = [], [], []
    for img, m, gt in frames:
        if not len(gt):
            n.append(0); continue
        r = PS.is_ring_geom(gt, m)
        n.append(int(r.sum()))
        rb = gt[r]
        if len(rb) > 1:
            rb = rb[np.argsort(rb[:, 0])]
            gaps += list(rb[1:, 0] - rb[:-1, 2])
        widths += list(rb[:, 2] - rb[:, 0]) if len(rb) else []
    return np.array(n), np.array(gaps), np.array(widths)


def report(tag, frames):
    n, g, w = rings_per_frame(frames)
    hist = np.histogram(n, bins=[0, 1, 3, 6, 11, 21, 31, 46, 1000])[0] / len(n)
    print(f'{tag:<10} frames {len(n):4d} | rings/frame mean {n.mean():5.1f} p10 {np.percentile(n,10):4.0f} '
          f'p50 {np.percentile(n,50):4.0f} p90 {np.percentile(n,90):4.0f} max {n.max():3d} | '
          f'fraction with 0 / 1-2 / 3-5 / 6-10 / 11-20 / 21-30 / 31-45 / 46+: '
          + ' '.join(f'{h:.2f}' for h in hist))
    if len(g):
        print(f'{"":<10} neighbour ring gap px (edge-edge) p10 {np.percentile(g,10):6.1f} p50 {np.percentile(g,50):6.1f} '
              f'p90 {np.percentile(g,90):6.1f} | ring box width p50 {np.median(w):.1f}', flush=True)


if __name__ == '__main__':
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    report('real41', PS.real(f'{PS.CUR}/41.h5', False))
    report('organic', PS.real(f'{PS.CUR}/organic_labeled.h5', True))
    report('legacy', PS.legacy(200))
    report('realbkg6', PS.realbkg6(200))

    # what the bank's powder entries can supply
    from physics_simulation import PhysicsSimulation
    from simulation import SimulationConfig
    ph = PhysicsSimulation(f'{PS.CUR}/cif_library_organic/bank/bank_organic_hkl.npz',
                           sim_config=SimulationConfig(), device='cpu')
    for qmax in (3.6, 4.45, 6.0, 7.8):
        cnt = []
        for i in np.random.choice(ph.powder_ids, 3000):
            q, _, I = ph._entry(int(i))
            q = np.asarray(q.cpu() if torch.is_tensor(q) else q, float)
            cnt.append(int((q < qmax * 0.995).sum()))
        cnt = np.array(cnt)
        print(f'powder entries, reflections below q_max {qmax}: p10 {np.percentile(cnt,10):.0f} '
              f'p50 {np.percentile(cnt,50):.0f} p90 {np.percentile(cnt,90):.0f} | '
              f'>=13: {np.mean(cnt>=13):.0%}  >=30: {np.mean(cnt>=30):.0%}', flush=True)
