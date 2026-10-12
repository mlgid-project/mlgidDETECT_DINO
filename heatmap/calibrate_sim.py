"""Closed-loop calibration of the real-background simulator against real 41: for each setting, draw frames from ONE simulator instance (the 13 GB bank is loaded once),
measure ring and segment appearance in the processed image exactly as heatmap/appearance_compare.py does, and print it next to the real-41 quartiles.

  python heatmap/calibrate_sim.py --n 40 --cfg base: --cfg g05:ring_gain=0.5 --cfg 'g03_w:ring_gain=0.3;ring_sq_scale=0.7' --cfg seg3:seg_gain=3 [--preset r6] [--out file]
A setting is name:key=value;key=value (separated by semicolons; values are python literals; keys are attributes of RealBkgSimulation, e.g. ring_gain, seg_gain, ring_sq_scale, ring_w_coef,
ring_chi_texture, ring_chi_corr, ring_min_gap, seg_wide_frac, seg_wide_sigma, voigt_eta, contrast_min, snr_min). Settings accumulate on the preset and do NOT carry over
to the next --cfg. Two frame sets per setting: n ring-only frames (single_structure, p_ring 1) and n segment-only frames (p_ring 0), 'spots_cap' as set by --spots.
GPU memory: the simulator renders with torch on cuda:0 (small); RAM: ~13 GB. Do not run next to a training that holds the RAM."""
import os, sys, types, argparse, ast, time, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap.appearance_compare import ring_row, seg_row, noise_stats

TARGET_RING = dict(amp=(0.077, 0.205, 0.348), amp_sd=(0.341, 0.965, 1.579), snr=(9.9, 20.1, 33.1), fwhm=(8, 12, 19), cv8=(0.109, 0.180, 0.254), wf=(0.419, 0.623, 0.853), w=(4.97, 7.46, 10.4))
TARGET_SEG = dict(amp=(0.146, 0.252, 0.403), amp_sd=(0.875, 1.376, 2.008), fq=(7, 10, 17), fc=(11, 21, 47.2), w=(2.41, 4.54, 7.0), h=(17.0, 37.1, 75.3))
TARGET_NOISE = dict(sd=(0.089, 0.166, 0.211), lag_q=(0.051, 0.190, 0.794), lag_chi=(0.236, 0.344, 0.821))


def draw(sim, n, ring_frames):
    sim.single_structure = True
    sim.p_ring = 1.0 if ring_frames else 0.0
    R, S, N, got, tries, nbox = [], [], [], 0, 0, []
    while got < n and tries < 6 * n:
        tries += 1
        try:
            out = sim.simulate_img()
        except Exception:
            continue
        if out is None:
            continue
        img, boxes, mask, is_ring = out
        img = img.float().cpu().numpy(); mask = mask.cpu().numpy().astype(bool); boxes = boxes.float().cpu().numpy(); ring = is_ring.cpu().numpy().astype(bool)
        got += 1; nbox.append(len(boxes))
        ns = noise_stats(img, mask)
        if ns:
            N.append(ns)
        xs = np.sort((boxes[ring, 0] + boxes[ring, 2]) / 2)
        for b in boxes[ring]:
            r = ring_row(img, mask, b, xs)
            if r: R.append(r)
        for b in boxes[~ring][:60]:
            r = seg_row(img, mask, b)
            if r: S.append(r)
    return R, S, N, nbox


def line(rows, target):
    out = []
    for k, (lo, mid, hi) in target.items():
        v = np.array([r[k] for r in rows], float); v = v[np.isfinite(v)]
        if len(v) < 3:
            out.append(f'{k} -'); continue
        q = np.percentile(v, [25, 50, 75])
        out.append(f'{k} {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g} (real {lo:.3g}/{mid:.3g}/{hi:.3g})')
    return '\n      '.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=40); p.add_argument('--cfg', action='append', default=[]); p.add_argument('--preset', default='r6')
    p.add_argument('--spots', default='(10,60)'); p.add_argument('--rings', default='(8,30)'); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from realsim.factory import make_realbkg_sim
    t0 = time.time()
    sim = make_realbkg_sim(device='cuda', preset=a.preset)
    print(f'[calibrate] simulator ready in {time.time() - t0:.0f} s', flush=True)
    base = {k: copy.deepcopy(getattr(sim, k)) for k in ('ring_gain', 'seg_gain', 'ring_sq_scale', 'ring_w_coef', 'ring_chi_texture', 'ring_chi_corr', 'ring_min_gap', 'seg_wide_frac',
                                                        'seg_wide_sigma', 'voigt_eta', 'contrast_min', 'snr_min', 'rings_cap', 'spots_cap', 'sq_q_slope', 'ring_contrast_min', 'amplitude_mode', 'elongate', 'seg_sc_scale', 'intensity_decades')}
    sim.spots_cap = tuple(ast.literal_eval(a.spots)); sim.rings_cap = tuple(ast.literal_eval(a.rings))
    base['spots_cap'], base['rings_cap'] = sim.spots_cap, sim.rings_cap
    for spec in (a.cfg or ['base:']):
        name, _, kv = spec.partition(':')
        for k, v in base.items():
            setattr(sim, k, copy.deepcopy(v))
        for item in [x for x in kv.split(';') if x]:
            k, v = item.split('=', 1)
            setattr(sim, k, ast.literal_eval(v))
        t = time.time()
        R, _, NR, nbR = draw(sim, a.n, True)
        _, S, NS, nbS = draw(sim, a.n, False)
        print(f'\n===== {name}: {kv or "(preset)"}   [{time.time() - t:.0f} s] ring frames: {len(R)} rings, mean {np.mean(nbR):.1f} boxes/frame; segment frames: {len(S)} segments (first 60/frame), mean {np.mean(nbS):.1f} boxes/frame', flush=True)
        print('  RINGS (sim p25/median/p75, real 41 p25/median/p75)\n      ' + line(R, TARGET_RING))
        print('  SEGMENTS\n      ' + line(S, TARGET_SEG))
        print('  BACKGROUND\n      ' + line(NR + NS, TARGET_NOISE), flush=True)


if __name__ == '__main__':
    main()
