"""Review figures for the r7 ring fix: what the network sees, with and without boxes, next to real 41.

Writes to tmp_diag/sim2/images/12_ring_fix/:
  ringfix_frames_r6_vs_r7.png   4 frames each from r6 and r7, preprocessed, GT boxes drawn
  ringfix_r7_raw_vs_log.png     5 r7 frames, raw linear | log10 counts, no boxes
  ringfix_real41.png            3 real 41 frames with GT, same rendering, for comparison
"""
import os, sys, random
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_shape_compare as PS
import realbkg_simulation as RS

OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/12_ring_fix'
R6 = 'config/DINO/DINO_4scale_swin_realbkg_r6.py'
R7 = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'


def frames(config, n, seed, raw=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sim = PS.build_realbkg(config)
    snap = {}
    _ac = RS.apply_contrast
    def hook(total, mask, chain):
        snap['t'] = np.asarray(total).copy()
        return _ac(total, mask, chain)
    RS.apply_contrast = hook
    out = []
    try:
        while len(out) < n:
            r = sim.simulate_img()
            if r is None:
                continue
            img, bx, m, _ = r
            m = np.asarray(m).astype(bool); b = np.asarray(bx)
            out.append(dict(img=np.asarray(img), mask=m, gt=b, raw=snap['t'],
                            ring=PS.is_ring_geom(b, m) if len(b) else np.zeros(0, bool)))
    finally:
        RS.apply_contrast = _ac
    return out


def draw_boxes(a, f):
    a.imshow(f['img'], cmap='magma', aspect='auto', vmin=0, vmax=1)
    for b, r in zip(f['gt'], f['ring']):
        a.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                  ec='#7CFC00' if r else '#00E5FF', lw=0.6))
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f'{len(f["gt"])} boxes, {int(f["ring"].sum())} rings', fontsize=9)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    f6, f7 = frames(R6, 4, 3), frames(R7, 9, 3)
    fig, ax = plt.subplots(4, 2, figsize=(18, 15))
    for i in range(4):
        draw_boxes(ax[i, 0], f6[i]); draw_boxes(ax[i, 1], f7[i])
    ax[0, 0].set_title('r6 (run 6, current)\n' + ax[0, 0].get_title(), fontsize=10)
    ax[0, 1].set_title('r7 (ring fix)\n' + ax[0, 1].get_title(), fontsize=10)
    fig.suptitle('What the network sees, GT boxes drawn (green ring, cyan segment)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f'{OUT}/ringfix_frames_r6_vs_r7.png', dpi=85)
    plt.close(fig)

    fig, ax = plt.subplots(5, 2, figsize=(16, 20))
    for i, f in enumerate(f7[4:9]):
        t, m = f['raw'], f['mask']
        good = t[m & (t > 0)]
        ax[i, 0].imshow(t, cmap='magma', aspect='auto', vmin=0, vmax=t.max())
        ax[i, 0].set_title(f'r7 frame {i+1} - RAW linear counts (0 to max {t.max():.2g})', fontsize=10)
        v = np.where(m, np.log10(np.maximum(t, 1.0)), np.nan)
        ax[i, 1].imshow(v, cmap='magma', aspect='auto', vmin=np.percentile(np.log10(np.maximum(good, 1)), 1),
                        vmax=np.log10(max(t.max(), 10)))
        ax[i, 1].set_title(f'r7 frame {i+1} - log10 counts ({len(f["gt"])} peaks, {int(f["ring"].sum())} rings)',
                           fontsize=10)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle('r7 ring fix - raw linear | log10, no boxes (pre-contrast)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f'{OUT}/ringfix_r7_raw_vs_log.png', dpi=85)
    plt.close(fig)

    real = list(PS.real(f'{PS.CUR}/41.h5', False))
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    for a, k in zip(ax, [33, 14, 5]):
        img, m, gt = real[k]
        draw_boxes(a, dict(img=img, gt=gt, ring=PS.is_ring_geom(gt, m)))
        a.set_title(f'real 41 frame {k}: ' + a.get_title(), fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/ringfix_real41.png', dpi=85)
    print('wrote', OUT)
