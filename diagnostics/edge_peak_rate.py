"""Does the simulator now produce cut-off peaks at the rate real data shows?

Measures the same quantity on SIMULATED frames that was measured on the labeled sets:
the fraction of GT boxes overlapping an invalid pixel (dark wedge or detector gap).

Real reference, measured 2026-09-06:
    41       1680 boxes   34.8% overlap the mask   100% of images have >= 1
    organic   817 boxes   20.1% overlap the mask   100% of images have >= 1

Run on a GPU node (the simulator is CUDA-only):  python diagnostics/edge_peak_rate.py [N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from simulation import FastSimulation, SimulationConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def rate(edge_peaks, n=N):
    cfg = SimulationConfig()
    cfg.a_coef, cfg.w_coef = 2.80, 1.30          # the box convention every run on this branch uses
    cfg.edge_peaks = edge_peaks
    sim = FastSimulation(sim_config=cfg, device='cuda')
    n_box = touch = imgs = imgs_with = clamped = 0
    for _ in range(n):
        img, boxes, mask, is_ring = sim.simulate_img()
        if boxes is None or not len(boxes):
            continue
        imgs += 1
        hit = 0
        m = mask.bool()
        for x0, y0, x1, y1 in boxes:
            xi0, yi0 = int(max(0, x0.floor())), int(max(0, y0.floor()))
            xi1, yi1 = int(min(m.shape[1], x1.ceil())), int(min(m.shape[0], y1.ceil()))
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            n_box += 1
            sub = m[yi0:yi1, xi0:xi1]
            if sub.numel() and not bool(sub.all()):
                touch += 1
                hit += 1
            #TRUNCATED: the box edge sits exactly on the wedge boundary, i.e. the label was
            #clamped to the mask instead of spanning the cut. This -- not the overlap rate --
            #is what the model learns from, and it is what edge_peaks changes.
            r = ((x0 + x1) / 2) / (1 + (m.shape[1] - 512) / 512)
            lo = float(sim.angle_limits.min(r.reshape(1))[0])
            hi = float(sim.angle_limits.max(r.reshape(1))[0])
            if abs(float(y0) - lo) < 0.75 or abs(float(y1) - hi) < 0.75:
                clamped += 1
        imgs_with += bool(hit)
    return imgs, n_box, touch, imgs_with, clamped


print(f'{N} simulated images each\n')
print(f'{"edge_peaks":12s} {"imgs":>5s} {"boxes":>7s} {"overlap mask":>15s} {"TRUNCATED at wedge":>20s}')
for flag in (False, True):
    imgs, n_box, touch, imgs_with, clamped = rate(flag)
    print(f'{str(flag):12s} {imgs:5d} {n_box:7d} {touch:8d} ({100*touch/max(n_box,1):5.1f}%) '
          f'{clamped:12d} ({100*clamped/max(n_box,1):5.1f}%)')
print('\nreal reference:   41  34.8% of boxes, 100% of images'
      '\n                  organic  20.1% of boxes, 100% of images')
