"""Ring:segment composition of the two simulators (MODIFICATIONS.md section L1).

The two eval gates want OPPOSITE ring rates -- 41 is 0.704 ring:segment, organic 0.035 -- so the
simulator's own rate decides which gate it fits. This measures it for the legacy sim and the
physics-CIF sim, with the geometric criterion (`is_ring_geom`, postproc_diag.py:115): a box is a
ring when it spans >= 70% of the VALID chi rows at its own radius. 41.h5 does not populate
`is_ring` at all, so geometry is the only trustworthy label on the real side.

Pass -n / --n-powder to sweep the physics sim's powder-entry range, the lever itself.

Needs CUDA:  python diagnostics/ring_rate_sims.py [-n 120] [--n-powder 0 3]
"""
import argparse
import random
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from simulation import FastSimulation, SimulationConfig          # noqa: E402
from physics_simulation import PhysicsSimulation                 # noqa: E402

BANK = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
BOX_COEF = (2.8, 1.3)   # current convention; the OLD one gives a different, non-comparable rate


def is_ring_geom(boxes, mask, frac=0.70):
    H, W = mask.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes):
        xc = int(np.clip((x0 + x1) / 2, 0, W - 1))
        out.append((y1 - y0) >= frac * max(int(mask[:, xc].sum()), 1))
    return np.array(out, bool)


def measure(sim, tag, n, seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gr, gs, fr, nb = [], [], [], []
    for _ in range(n):
        img, boxes, mask, flag = sim.simulate_img()
        m = mask.detach().cpu().numpy().astype(bool).squeeze()
        r = is_ring_geom(boxes.detach().cpu().numpy(), m)
        gr.append(int(r.sum())); gs.append(int((~r).sum())); nb.append(len(boxes))
        fr.append(int(np.asarray(flag.detach().cpu()).sum()))
    gr, gs, fr = np.array(gr), np.array(gs), np.array(fr)
    print(f'{tag:<34} rings/frame {gr.mean():6.2f}  segs/frame {gs.mean():7.2f}  '
          f'ring:seg {gr.sum() / max(gs.sum(), 1):6.3f}  objects/frame {np.mean(nb):7.1f}  '
          f'flagged-ring/frame {fr.mean():5.2f}  0-ring frames {int((gr == 0).sum())}/{n}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=120, help='frames per simulator')
    ap.add_argument('--bank', default=BANK)
    ap.add_argument('--n-powder', type=int, nargs=2, default=None, metavar=('LO', 'HI'),
                    help='also measure the physics sim at this powder-entry range')
    a = ap.parse_args()

    def cfg():
        c = SimulationConfig(); c.a_coef, c.w_coef = BOX_COEF; return c

    print(f'{"":<34} (ring == box spans >= 70% of the valid chi rows at its radius)')
    print(f'{"REAL 41":<34} rings/frame  16.90  segs/frame   24.00  ring:seg  0.704')
    print(f'{"REAL organic":<34} rings/frame   3.50  segs/frame   98.60  ring:seg  0.035')
    measure(FastSimulation(sim_config=cfg(), device='cuda'), 'sim legacy', a.n)
    measure(PhysicsSimulation(a.bank, sim_config=cfg(), device='cuda', unify_contrast=True),
            'sim physics (default N_POWDER)', a.n)
    if a.n_powder is not None:
        lo, hi = a.n_powder
        measure(PhysicsSimulation(a.bank, sim_config=cfg(), device='cuda', unify_contrast=True,
                                  n_powder=(lo, hi)), f'sim physics n_powder=({lo}, {hi})', a.n)


if __name__ == '__main__':
    main()
