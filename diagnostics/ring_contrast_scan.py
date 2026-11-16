"""Is RING CONTRAST what makes 41/legacy rings foreign to a realbkg model?

realbkg-41-gap-split: conv3 scores its own sim's rings at 0.96 but legacy rings at 0.36 and 41's at
0.30. The r7 ring fix matched count, segment overlap, box width, visible width and spacing to 41,
and conv3 STILL scores r7 rings at 0.95 -- so none of those is the feature. The largest remaining
mismatch is brightness: r7 ring contrast on the processed image reads 0.32-0.45 against real 41's
0.22 and legacy's 0.18.

Test: the SAME r7 frames (every seed fixed, background included) with only the rings' rendered
amplitude scaled by s. Labels are identical across s -- the gate runs on the unscaled amplitude.
If conv3's ring score falls from ~0.95 towards ~0.36 as ring contrast reaches 41's, contrast is it.

  python diagnostics/ring_contrast_scan.py name=/ckpt.pth [...]
"""
import os, sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
import peak_shape_compare as PS
from cross_sim_probe import score
from realbkg_simulation import RealBkgSimulation

N = int(os.environ.get('SCAN_FRAMES', 60))
CFG = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'
OV = dict(realbkg_ring_min_gap=8.0, realbkg_rings_cap=(2, 60), realbkg_mosaic_seed=1234)
SCALES = [float(v) for v in os.environ.get('SCAN_SCALES', '1.0,0.5,0.25,0.12,0.06').split(',')]

_render = RealBkgSimulation._render
RING_SCALE = {'s': 1.0}


def _render_scaled(self, x, y, s_q, s_c, amp, eta):
    # rings are the peaks _draw_widths gave a chi width of 1e4
    amp = np.where(np.asarray(s_c) >= 1e3, np.asarray(amp) * RING_SCALE['s'], amp)
    return _render(self, x, y, s_q, s_c, amp, eta)


RealBkgSimulation._render = _render_scaled


def frames_at(s):
    RING_SCALE['s'] = s
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    return list(PS.realbkg6(N, config=CFG, overrides=OV))


def ring_contrast(frames):
    c = []
    for img, m, gt in frames:
        if not len(gt):
            continue
        for b, r in zip(gt, PS.is_ring_geom(gt, m)):
            if r:
                st = PS.profile_stats(img, m, b)
                if st is not None:
                    c.append(st['con'])
    return np.percentile(c, 10), np.percentile(c, 50), len(c)


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    sets = {}
    for s in SCALES:
        sets[s] = frames_at(s)
        p10, p50, n = ring_contrast(sets[s])
        nb = np.mean([len(g) for _, _, g in sets[s]])
        print(f'ring scale {s:4.2f}: ring contrast p10 {p10:.3f} p50 {p50:.3f} (n {n}) | boxes/frame {nb:.1f}'
              f'   [real 41: p10 0.07 p50 0.22 | legacy p50 0.18]', flush=True)
    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep})', flush=True)
        for s in SCALES:
            print(f'  ring scale {s:4.2f}  {score(model, args, sets[s])}', flush=True)
        del model
