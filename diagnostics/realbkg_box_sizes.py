"""Box size the realbkg simulator actually emits, against the real labels and the legacy sim.

The two simulators build a box from the same sigmas with formulas that differ by a factor of two:

    legacy   (simulation.py, _boxes_from_positions)  full width 2*w_coef*sigma = 2.6 sigma_q
                                                     full height 2*a_coef*sigma = 5.6 sigma_chi
    realbkg  (realbkg_simulation.py:526, 583)        full width   w_coef*sigma  = 1.3 sigma_q
                                                     full height  a_coef*sigma  = 2.8 sigma_chi

box_coef_override=(2.8, 1.3) was tuned for the LEGACY formula, so if the sigmas mean the same
thing in both -- and `_render`'s exp(-u2/2) with u2=((X-x)/s_q)^2 says they do -- every realbkg
box is half the linear size it should be. This measures the emitted boxes so the question is
settled by numbers rather than by reading intent into the `/2.0`.
"""
import os, sys, argparse, random
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

import torch, argparse as _a
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
from util.slconfig import SLConfig
from simulation import SimulationConfig
from realbkg_simulation import RealBkgSimulation
from diagnostics.cache_realbkg_donors import load_into

cfg = SLConfig.fromfile(args.config)
a = _a.Namespace(**{k: v for k, v in cfg.items()})
sc = SimulationConfig(); sc.a_coef, sc.w_coef = getattr(cfg, 'box_coef_override', (2.80, 1.30))
RealBkgSimulation._load_donors = lambda self, *x, **k: load_into(
    self, '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')
sim = RealBkgSimulation(
    bank_path=a.physics_bank_path, donor_path=a.realbkg_donor_path, stats_path=a.realbkg_stats_path,
    sim_config=sc, device='cpu',
    n_oriented=tuple(a.realbkg_n_oriented), p_ring=float(a.realbkg_p_ring),
    mosaic=bool(a.realbkg_mosaic), mosaic_pool=int(a.realbkg_mosaic_pool),
    mosaic_refresh=int(a.realbkg_mosaic_refresh), mosaic_seed=getattr(a, 'realbkg_mosaic_seed', None),
    intensity_decades=a.realbkg_intensity_decades, amplitude_mode=a.realbkg_amplitude_mode,
    mask_bank=bool(a.realbkg_mask_bank), mask_keep=a.realbkg_mask_keep,
    unified_labels=bool(a.realbkg_unified_labels), contrast_min=float(a.realbkg_contrast_min),
    snr_min=float(a.realbkg_snr_min), ring_iou_max=float(a.realbkg_ring_iou_max),
    seg_iou_max=a.realbkg_seg_iou_max, max_peaks=a.realbkg_max_peaks,
    spots_cap=a.realbkg_spots_cap, rings_cap=a.realbkg_rings_cap,
    n_powder=tuple(a.realbkg_n_powder),
    ring_box_from_mask=bool(getattr(a, 'realbkg_ring_box_from_mask', False)),
    seg_wide_frac=float(getattr(a, 'realbkg_seg_wide_frac', 0.0)),
    seg_wide_sigma=getattr(a, 'realbkg_seg_wide_sigma', ((8.1, 0.45), (2.9, 0.45))))

grab = {}
_vis = sim._visibility
def vis_hook(amp, noise_at, s_q, s_c, is_ring, mask, x):
    grab['s_q'] = np.asarray(s_q).copy(); grab['s_c'] = np.asarray(s_c).copy()
    grab['rg'] = np.asarray(is_ring).copy()
    return _vis(amp, noise_at, s_q, s_c, is_ring, mask, x)
sim._visibility = vis_hook

sq, scm, sw, sh, rw = [], [], [], [], []
rh, rcov, nb, nr = [], [], [], []
n = 0
while n < args.frames:
    grab.clear()
    out = sim.simulate_img()
    if out is None or 's_q' not in grab:
        continue
    _i, bx, _m, rg = out
    b = np.asarray(bx, float); r = np.asarray(rg, bool)
    if len(b) == 0:
        continue
    n += 1
    w = b[:, 2]-b[:, 0]; h = b[:, 3]-b[:, 1]
    sw += list(w[~r]); sh += list(h[~r]); rw += list(w[r])
    rh += list(h[r])
    nb.append(len(b)); nr.append(int(r.sum()))
    msk = np.asarray(_m, bool)
    if r.any():
        cc = np.clip(((b[r, 0]+b[r, 2])/2).astype(int), 0, msk.shape[1]-1)
        span = msk[:, cc].sum(0).astype(float)
        rcov += list(h[r]/np.maximum(span, 1.0))
    g = grab['rg']
    sq += list(grab['s_q'][~g]); scm += list(grab['s_c'][~g])
    if n % 30 == 0:
        print(f'  {n} frames...', flush=True)

p = lambda x, q: np.percentile(x, q) if len(x) else float('nan')
sq, scm, sw, sh, rw = map(np.array, (sq, scm, sw, sh, rw))
print(f'\n{n} frames | {len(sw)} segment boxes, {len(rw)} ring boxes | config {args.config}')
print(f'  sigma_q   of segments      p10 {p(sq,10):7.2f}  p50 {p(sq,50):7.2f}  p90 {p(sq,90):7.2f}')
print(f'  sigma_chi of segments      p10 {p(scm,10):7.2f}  p50 {p(scm,50):7.2f}  p90 {p(scm,90):7.2f}')
print(f'\n  EMITTED segment box w px   p10 {p(sw,10):7.2f}  p50 {p(sw,50):7.2f}  p90 {p(sw,90):7.2f}')
print(f'  EMITTED segment box h px   p10 {p(sh,10):7.2f}  p50 {p(sh,50):7.2f}  p90 {p(sh,90):7.2f}')
print(f'  EMITTED ring box    w px   p10 {p(rw,10):7.2f}  p50 {p(rw,50):7.2f}  p90 {p(rw,90):7.2f}')
print(f'\n  box w / sigma_q            {p(sw,50)/max(p(sq,50),1e-9):.2f}   (legacy formula gives 2.60)')
print(f'  box h / sigma_chi          {p(sh,50)/max(p(scm,50),1e-9):.2f}   (legacy formula gives 5.60)')
rh, rcov, nb, nr = map(np.array, (rh, rcov, nb, nr))
print(f'\n  EMITTED ring box    h px   p10 {p(rh,10):7.1f}  p50 {p(rh,50):7.1f}  p90 {p(rh,90):7.1f}')
print(f'  ring h / valid chi span    p10 {p(rcov,10):7.2f}  p50 {p(rcov,50):7.2f}  p90 {p(rcov,90):7.2f}')
print(f'  rings FULL height (>=505): {100*np.mean(rh>=505):4.1f}%')
print(f'\n  boxes/frame    p10 {p(nb,10):6.1f}  p50 {p(nb,50):6.1f}  p90 {p(nb,90):6.1f}  '
      f'mean {nb.mean():6.1f}')
print(f'  segments/frame p10 {p(nb-nr,10):6.1f}  p50 {p(nb-nr,50):6.1f}  p90 {p(nb-nr,90):6.1f}')
print(f'  rings/frame    mean {nr.mean():5.2f}   ring-free frames {100*np.mean(nr==0):4.0f}%')
print('\nREAL GT, measured:')
print('  organic  rings 80.0% full height, h/span 1.03-1.14 | p50 66 boxes, 2.12 rings, 62% ring-free')
print('  41       rings  0.6% full height, h/span p50 0.98  | p50 20 boxes, 8.85 rings,  0% ring-free')
print('  organic segments  w p50 10.5  h p50  8.1')
print('  41      segments  w p50  4.6  h p50 34.2')
print('legacy sim segments h p50 23.9')
print('\nIf the /2.0 is wrong, DOUBLING these widths and heights is what the boxes should be:')
print(f'  doubled segment box w p50 {2*p(sw,50):7.2f}   h p50 {2*p(sh,50):7.2f}')
