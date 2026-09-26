"""Dynamic range of the finished simulator, measured over enough frames to be a distribution.

max/median over valid pixels of the RAW (pre-contrast) frame. Real labelled frames sit at
96 to 13,116; the simulator used to sit at 2.4 / 8.1 / 12.9 (min/median/max over 10 frames),
two to three decades short, and that gap was a declared blocker on any training run.

Builds the simulator from the real config exactly as main.py does, so what is measured is what
would train. Run as a job: the hkl bank needs ~15 GB.
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

import random, torch, argparse as _a
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
from util.slconfig import SLConfig
from simulation import SimulationConfig
import realbkg_simulation as RS
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
    n_powder=tuple(a.realbkg_n_powder))

snap = {}
_ac = RS.apply_contrast
def hook(total, mask, chain):
    snap['raw'] = np.asarray(total).copy(); snap['m'] = np.asarray(mask).copy()
    return _ac(total, mask, chain)
RS.apply_contrast = hook

dyn, nbox, nring, nseg = [], [], [], []
while len(dyn) < args.frames:
    r = sim.simulate_img()
    if r is None or 'raw' not in snap:
        continue
    raw, m = snap['raw'], snap['m'].astype(bool)
    dyn.append(float(raw[m].max()/max(np.median(raw[m]), 1e-12)))
    _i, bx, _mk, rg = r
    nbox.append(len(bx)); nring.append(int(rg.sum())); nseg.append(len(bx)-int(rg.sum()))
    if len(dyn) % 20 == 0:
        print(f'  {len(dyn):3d} frames...', flush=True)

d = np.array(dyn); b = np.array(nbox); rr = np.array(nring); ss = np.array(nseg)
REAL = np.array([96, 151, 1388, 2837, 4997, 7476, 9540, 13116])
print(f'\nDYNAMIC RANGE (max/median, raw), {len(d)} frames')
for tag, x in (('sim', d), ('real labelled (8 frames)', REAL)):
    print(f'  {tag:<26s} min {x.min():9.1f}  p10 {np.percentile(x,10):9.1f}  '
          f'p50 {np.median(x):9.1f}  p90 {np.percentile(x,90):9.1f}  max {x.max():9.1f}')
inside = ((d >= REAL.min()) & (d <= REAL.max())).mean()
print(f'  sim frames inside real\'s observed range [{REAL.min()}, {REAL.max()}]: {100*inside:.0f}%')
print(f'  sim frames BELOW real\'s floor: {100*(d < REAL.min()).mean():.0f}%   '
      f'ABOVE real\'s ceiling: {100*(d > REAL.max()).mean():.0f}%')
print(f'\nCOMPOSITION, {len(b)} frames')
print(f'  boxes/frame    mean {b.mean():6.1f}  p50 {int(np.median(b)):4d}  max {b.max():4d}   '
      f'[real organic p50 66 max 168 | 41 p50 20 max 65]')
print(f'  rings/frame    mean {rr.mean():6.2f}  ring-free {100*(rr==0).mean():.0f}%          '
      f'[real organic 2.12 / 62% | 41 8.85 / 0%]')
print(f'  segments/frame mean {ss.mean():6.1f}                          '
      f'[real organic 63.6 | 41 16.5]')
print(f'  corr(rings, segments) = {np.corrcoef(rr, ss)[0,1]:+.2f}      [real organic -0.48 | 41 -0.25]')
