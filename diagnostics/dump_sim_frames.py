"""Render simulator frames with their ground-truth boxes drawn on top -- the visual counterpart to
the composition tables in diagnostics/ring_rate_sims.py.

The numbers say branch A is 41-shaped and branch B is organic-shaped. This shows whether the
frames LOOK it, which is the part a table cannot tell you.

Needs CUDA (the simulators render on GPU), so run it through an existing allocation:
  srun --jobid=<a running job> --overlap --gres=gpu:1 python diagnostics/dump_sim_frames.py

USAGE
  python diagnostics/dump_sim_frames.py                      # physics6 branches + control + real
  python diagnostics/dump_sim_frames.py --frames 6 --out DIR
"""
import argparse, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import random
import torch
from simulation import SimulationConfig                               # noqa: E402
from physics_simulation import PhysicsSimulation                      # noqa: E402
from diagnostics.ring_rate_sims import is_ring_geom                   # noqa: E402

BANK = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz'
ORGANIC = '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'

# the dino_physics6_1 mixture, one panel row per branch so each is visible on its own
BRANCHES = [('A_41-shaped',      [(1.0, (1, 3), (1, 1))]),
            ('B_organic-shaped', [(1.0, (0, 1), (3, 5))]),
            ('C_broad',          [(1.0, (0, 4), (1, 6))]),
            ('POOLED_physics6',  [(1.0, (1, 3), (1, 1)),
                                  (1.0, (0, 1), (3, 5)),
                                  (1.0, (0, 4), (1, 6))]),
            ('CONTROL_physics5', None)]


def cfg():
    c = SimulationConfig(); c.a_coef, c.w_coef = 2.80, 1.30; return c


def panel(ax, img, boxes, mask, title):
    ax.imshow(img, cmap='gray', origin='lower', aspect='auto', vmin=0, vmax=1)
    rings = is_ring_geom(boxes, mask) if len(boxes) else np.zeros(0, bool)
    for (x0, y0, x1, y1), r in zip(boxes, rings):
        # red = ring (spans >= 70% of the valid chi rows at its radius), cyan = segment
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec='red' if r else 'cyan', lw=0.6, alpha=0.9))
    ax.set_title(f'{title}\n{int(rings.sum())} rings / {int((~rings).sum())} segs',
                 fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def sim_frames(frame_types, n, seed):
    sim = PhysicsSimulation(BANK, sim_config=cfg(), device='cuda', unify_contrast=True,
                            n_powder=(0, 3), real_tail_only=True, frame_types=frame_types)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = []
    for _ in range(n):
        img, b, mask, _ = sim.simulate_img()
        out.append((img.detach().cpu().numpy(),
                    b.detach().cpu().numpy(),
                    mask.detach().cpu().numpy().astype(bool).squeeze()))
    return out


def real_frames(n):
    from util.configuration import Config
    from util.exp_preprocess import standard_preprocessing
    from util.pygidloader import PyGIDDataset
    c = Config(); c.PREPROCESSING_POLAR_SHAPE = [512, 1024]; c.INPUT_DATASET = ORGANIC
    ds = PyGIDDataset(c, path=ORGANIC, preprocess_func=standard_preprocessing,
                      buffer_size=5, load_labels=True)
    out = []
    for ic in ds.iter_images():
        if ic.polar_labels is None:
            continue
        b = np.asarray(ic.polar_labels.boxes)
        if b.ndim != 2 or len(b) == 0:
            continue
        img = ic.converted_polar_image[0, 0]
        out.append((img, b, img > 0))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=4, help='frames per row')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/tmp_diag/sim_frames')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [(tag, sim_frames(ft, a.frames, a.seed)) for tag, ft in BRANCHES]
    try:
        rows.append(('REAL_organic', real_frames(a.frames)))
    except Exception as e:                                   # real data is optional here
        print(f'[warn] real frames skipped: {e}')

    fig, axes = plt.subplots(len(rows), a.frames,
                             figsize=(3.1 * a.frames, 2.0 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (tag, frames) in enumerate(rows):
        for j in range(a.frames):
            ax = axes[i, j]
            if j < len(frames):
                img, b, m = frames[j]
                panel(ax, img, b, m, f'{tag}  #{j}')
            else:
                ax.axis('off')
    fig.suptitle('dino_physics6_1 frame-type mixture -- red = ring, cyan = segment', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid = os.path.join(a.out, 'physics6_frames.png')
    fig.savefig(grid, dpi=130); plt.close(fig)
    print(f'[out] {grid}')

    # one full-resolution frame per row, for looking at individual peaks
    for tag, frames in rows:
        if not frames:
            continue
        img, b, m = frames[0]
        f2, ax = plt.subplots(figsize=(13, 6.5))
        panel(ax, img, b, m, tag)
        f2.tight_layout()
        p = os.path.join(a.out, f'frame_{tag}.png')
        f2.savefig(p, dpi=130); plt.close(f2)
        print(f'[out] {p}')


if __name__ == '__main__':
    main()
