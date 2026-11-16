"""Transplant test: is along-chi RING TEXTURE what makes 41/legacy rings foreign to realbkg models?

Ruled out so far (realbkg-41-gap-split): ring count, segment overlap, box width, ring width, ring
spacing, blur, contrast (ring_contrast_scan: conv3 still 0.93 at 41's contrast), preprocessing.

The legacy sim multiplies its whole peak image by smooth Perlin noise before any background
(simulation.py: `img = mul_perlin(img)`, factor 1 +- up to 0.25, cells 16-64 px), so its rings are
PATCHY along chi -- as real rings are. realbkg renders every ring perfectly uniform along chi.
A model that only ever saw uniform rings may read each bright patch on a real ring as a segment.

Four arms, scored by the deployed postprocessing against each frame's own labels:
  A  r7 as is                              (conv3 0.95 on rings)
  B  r7 with legacy's mul_perlin applied to its rendered peak image
  C  legacy as is                          (conv3 0.36 on rings)
  D  legacy with mul_perlin switched off
Texture is the feature if conv3 collapses on B and recovers on D.

Also measures ring PATCHINESS on the processed image for real 41 and every arm: per ring, the
along-chi profile (median over the ring's central columns) -> coefficient of variation after
removing a 61-row running mean (i.e. patch-scale variation, not the slow chi trend).

  python diagnostics/transplant_probe.py name=/ckpt.pth [...]
"""
import os, sys, random
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
import peak_shape_compare as PS
from cross_sim_probe import score
import simulation as SIM
from realbkg_simulation import RealBkgSimulation

N = int(os.environ.get('TP_FRAMES', 60))
R7 = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'
OV = dict(realbkg_ring_min_gap=8.0, realbkg_rings_cap=(2, 60), realbkg_mosaic_seed=1234)

_render = RealBkgSimulation._render
_mul_perlin = SIM.mul_perlin
MODE = {'perlin_on_realbkg': False}


def _render_textured(self, x, y, s_q, s_c, amp, eta):
    img = _render(self, x, y, s_q, s_c, amp, eta)
    if MODE['perlin_on_realbkg']:
        img = _mul_perlin(torch.as_tensor(img, dtype=torch.float32)).numpy().astype(np.float64)
    return img


RealBkgSimulation._render = _render_textured


def realbkg_arm(textured):
    MODE['perlin_on_realbkg'] = textured
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    return list(PS.realbkg6(N, config=R7, overrides=OV))


def legacy_arm(perlin):
    SIM.mul_perlin = _mul_perlin if perlin else (lambda img: img)
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    try:
        return list(PS.legacy(N))
    finally:
        SIM.mul_perlin = _mul_perlin


def patchiness(frames):
    cv = []
    for img, m, gt in frames:
        if not len(gt):
            continue
        for b, r in zip(gt, PS.is_ring_geom(gt, m)):
            if not r:
                continue
            xc = int(np.clip((b[0] + b[2]) / 2, 1, img.shape[1] - 2))
            cols = slice(xc - 1, xc + 2)
            valid = m[:, cols].all(1)
            if valid.sum() < 80:
                continue
            prof = np.median(img[valid, cols], axis=1)
            trend = uniform_filter1d(prof, 61, mode='nearest')
            cv.append(float(np.std(prof - trend) / max(np.mean(prof), 1e-6)))
    return np.percentile(cv, 50) if cv else float('nan'), len(cv)


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    p, n = patchiness(list(PS.real(f'{PS.CUR}/41.h5', False)))
    print(f'real 41: ring patchiness (CV after detrend) p50 {p:.3f} (n {n})', flush=True)
    arms = {'A r7 uniform rings': realbkg_arm(False),
            'B r7 + legacy mul_perlin': realbkg_arm(True),
            'C legacy as is': legacy_arm(True),
            'D legacy, mul_perlin off': legacy_arm(False)}
    for k, fr in arms.items():
        p, n = patchiness(fr)
        print(f'{k:<26} ring patchiness p50 {p:.3f} (n {n}) | rings/frame '
              f'{np.mean([PS.is_ring_geom(g, m).sum() if len(g) else 0 for _, m, g in fr]):.1f}', flush=True)
    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep})', flush=True)
        for k, fr in arms.items():
            print(f'  {k:<26} {score(model, args, fr)}', flush=True)
        del model
