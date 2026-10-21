"""Legacy-sim frames with the OLD box convention (a_coef 2.80, w_coef 1.30) and the NEW one (+-1 sigma = a_coef
w_coef 2.0) side by side. Same seed before every frame, so both conventions see the same peaks (box edges are
the only thing that changes, apart from the sim's own tiny-box filter). Rings green, segments cyan.
Left: full frame. Right: the same zoom window for both.   python diagnostics/newbox_sim_images.py
"""
import os, sys, random
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import simulation as SIM

OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/15_newbox_legacy_conventions'
SEEDS = [int(s) for s in os.environ.get('NB_SEEDS', '3,11,21,34,55').split(',')]
CONV = [('OLD  a_coef 2.80 / w_coef 1.30', (2.8, 1.3)), ('NEW  +-1 sigma  (a_coef = w_coef = 2.0)', (2.0, 2.0))]


def frame(sim, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    while True:
        r = sim.simulate_img()
        if r is not None:
            img, bx, m, ring = r
            return (np.asarray(img.detach().cpu(), np.float32).reshape(512, 1024),
                    np.asarray(bx.detach().cpu(), np.float32).reshape(-1, 4),
                    np.asarray(ring.detach().cpu()).astype(bool).reshape(-1))


def show(ax, img, bx, ring, win=None, lw=0.8):
    lo, hi = np.percentile(img, [1, 99.7])
    ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1), cmap='gray', aspect='auto', vmin=0, vmax=1)
    for b, rg in zip(bx, ring):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   ec='#7CFC00' if rg else '#00E5FF', lw=lw))
    if win is not None:
        ax.set_xlim(win[0], win[2]); ax.set_ylim(win[3], win[1])
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    sims = []
    for name, (a, w) in CONV:
        sc = SIM.SimulationConfig(); sc.a_coef, sc.w_coef = a, w
        sims.append(SIM.FastSimulation(sim_config=sc, device='cpu'))
    for seed in SEEDS:
        fr = [frame(s, seed) for s in sims]
        (io, bo, ro), (inew, bn, rn) = fr
        # zoom window: 256 x 128 px around the segment with most neighbours (new boxes)
        seg = bn[~rn]
        if len(seg):
            c = np.stack([(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2], 1)
            d = ((c[:, None] - c[None]) ** 2).sum(-1)
            k = int(np.argmax((d < 60 ** 2).sum(1))); cx, cy = c[k]
        else:
            cx, cy = 512, 256
        x0 = int(np.clip(cx - 128, 0, 1024 - 256)); y0 = int(np.clip(cy - 64, 0, 512 - 128))
        win = (x0, y0, x0 + 256, y0 + 128)
        fig, axs = plt.subplots(2, 2, figsize=(16, 6.2), gridspec_kw=dict(width_ratios=[2.6, 1.3]))
        for row, ((name, _), (img, bx, rg)) in enumerate(zip(CONV, fr)):
            show(axs[row, 0], img, bx, rg)
            axs[row, 0].add_patch(plt.Rectangle((win[0], win[1]), 256, 128, fill=False, ec='#FF3B30', lw=1.0, ls='--'))
            axs[row, 0].set_title(f'{name}: {len(bx)} boxes ({int(rg.sum())} rings, {int((~rg).sum())} segments)', fontsize=10)
            show(axs[row, 1], img, bx, rg, win=win, lw=1.2)
            axs[row, 1].set_title('zoom (red window)', fontsize=10)
        fig.suptitle(f'legacy sim, seed {seed}: same frame, two label conventions (rings green, segments cyan)', fontsize=11)
        fig.tight_layout()
        fig.savefig(f'{OUT}/legacy_conventions_seed{seed:03d}.png', dpi=100)
        plt.close(fig)
        print(f'seed {seed}: old {len(bo)} boxes, new {len(bn)} boxes', flush=True)
    print('wrote', OUT)
