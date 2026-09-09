"""Wiring check for the physics-CIF generator (MODIFICATIONS.md section I).

Asserts PhysicsSimulation honours the FastSimulation 4-tuple contract that SimulationDataset and
collate_fn depend on, in both contrast modes, and reports how physics intensities actually differ
from the standard sim's uniform draws -- the thing the track is for.

Needs CUDA. Usage: python diagnostics/physics_smoke.py [--bank path] [-n 12]
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from simulation import FastSimulation, SimulationConfig, HEIGHT, WIDTH   # noqa: E402
from physics_simulation import PhysicsSimulation                          # noqa: E402


def check(sim, n, tag):
    n_boxes, rings, per_img, outside = [], 0, [], []
    for i in range(n):
        img, boxes, mask, is_ring = sim.simulate_img()
        assert img.shape == (HEIGHT, WIDTH), f'{tag}: img shape {tuple(img.shape)}'
        assert img.dtype == torch.float32, f'{tag}: img dtype {img.dtype}'
        assert torch.isfinite(img).all(), f'{tag}: non-finite pixels'
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0, \
            f'{tag}: range [{float(img.min()):.3f}, {float(img.max()):.3f}]'
        assert mask.shape == (HEIGHT, WIDTH) and mask.dtype == torch.bool, f'{tag}: bad mask'
        assert len(boxes) > 0, f'{tag}: empty boxes'
        assert boxes.shape[1] == 4, f'{tag}: boxes shape {tuple(boxes.shape)}'
        assert len(is_ring) == len(boxes), f'{tag}: is_ring/boxes length mismatch'
        # the DINO matcher asserts this on every target box (util/box_ops.py:53)
        assert bool(((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()), \
            f'{tag}: degenerate/inverted box survived'
        assert float(boxes[:, [0, 2]].min()) >= 0 and float(boxes[:, [0, 2]].max()) <= WIDTH, \
            f'{tag}: x out of canvas'
        assert float(boxes[:, [1, 3]].min()) >= 0 and float(boxes[:, [1, 3]].max()) <= HEIGHT, \
            f'{tag}: y out of canvas'
        # Report, don't assert: the standard sim's own tail (digitalize_img -> normalize) runs
        # over the whole canvas, so masked pixels are not necessarily 0 there either. What
        # matters is that physics is no worse than the generator it is diluting.
        if bool((~mask).any()):
            outside.append(float(img[~mask].abs().max()))
        n_boxes.append(len(boxes))
        rings += int(is_ring.sum())
        per_img.append(float(img.mean()))
    out = max(outside) if outside else 0.0
    print(f'  {tag:<28} OK  boxes/img {np.mean(n_boxes):6.1f}  rings {rings / sum(n_boxes):5.1%}'
          f'  mean px {np.mean(per_img):.3f}  max|px| outside mask {out:.3f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='/mnt/lustre/work/schreiber/szb389/datasets/'
                                      'cif_library_organic/bank/bank_gate.npz')
    ap.add_argument('-n', type=int, default=12)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'needs CUDA'

    # the config the run actually trains under
    cfg = SimulationConfig()
    cfg.a_coef, cfg.w_coef = 2.80, 1.30
    print(f'bank {args.bank}\nbox convention a_coef {cfg.a_coef} w_coef {cfg.w_coef}\n')

    phys_out = {}
    for unify in (False, True):
        sim = PhysicsSimulation(args.bank, sim_config=cfg, unify_contrast=unify)
        phys_out[unify] = check(sim, args.n, f'physics unify_contrast={unify}')
    std_out = check(FastSimulation(sim_config=cfg), args.n, 'standard sim (control)')
    for unify, v in phys_out.items():
        assert v <= max(std_out, 1e-5) + 1e-6, (
            f'physics unify_contrast={unify} leaves {v:.3f} outside the mask vs the standard '
            f'sim\'s {std_out:.3f} -- physics is dirtier than what it dilutes')

    # the actual point of the track: intensity SHAPE, normalized per pattern
    d = np.load(args.bank, allow_pickle=False)
    # materialise once: npz members are lazily decompressed, so slicing d['intensity'] inside the
    # loop re-inflates the whole array on every entry
    inten = d['intensity'][:].astype(float)
    phys = []
    for s, c in zip(d['entry_start'][:4000], d['entry_count'][:4000]):
        v = inten[s:s + c]
        v = v[v > 0]
        if len(v) >= 5:
            phys.append(v / v.max())
    phys = np.concatenate(phys)
    rng = np.random.default_rng(0)
    unif = np.concatenate([(lambda v: v / v.max())(rng.uniform(2, 50, 100)) for _ in range(4000)])
    print(f'\nintensity shape, I/Imax pooled over patterns:')
    print(f'  {"physics (structure factors)":<30} median {np.median(phys):.3f}  '
          f'frac<0.1 {float((phys < 0.1).mean()):.3f}')
    print(f'  {"standard sim (uniform 2-50)":<30} median {np.median(unif):.3f}  '
          f'frac<0.1 {float((unif < 0.1).mean()):.3f}')
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
