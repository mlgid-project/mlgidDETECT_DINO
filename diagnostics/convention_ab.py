"""What the unified labelling convention dropped, measured on IDENTICAL frames.

conv1/conv2 are the only runs that ever set use_realbkg_sim, so the convention never had a
same-simulator control: it shipped together with the whole real-background image source, and the
AP regression cannot be attributed to either from the curves alone. This measures the convention
alone.

Both label paths in simulate_img() share every RNG draw up to `eta = random.uniform(...)` -- the
background pick, the peak draw, the widths and the amplitudes all happen before the branch. So
snapshotting the RNG state and running the frame twice, once with unified_labels on and once off,
gives two labellings of the SAME peaks on the SAME background. Mosaic refresh is disabled for the
duration so the donor pool cannot shift between the two halves of a pair.

The internals are read by wrapping _visibility, _nms and _render rather than by editing them, so
what is measured is the shipped code path.
"""
import os, sys, argparse, random, copy
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=150)
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

import torch, argparse as _a
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
sim.mosaic_refresh = 0          # no slot swap between the two halves of a pair

# ---------------------------------------------------------------- instrumentation
rec = {}
_vis, _nms, _render = sim._visibility, sim._nms, sim._render

def vis_hook(amp, noise_at, s_q, s_c, is_ring, mask, x):
    con, snr = _vis(amp, noise_at, s_q, s_c, is_ring, mask, x)
    rec['amp'] = np.asarray(amp).copy(); rec['con'] = np.asarray(con).copy()
    rec['snr'] = np.asarray(snr).copy(); rec['rg'] = np.asarray(is_ring).copy()
    rec['noise'] = np.asarray(noise_at).copy()
    return con, snr

def nms_hook(bx, sel, amp, iou_max):
    keep = _nms(bx, sel, amp, iou_max)
    killed = int((np.asarray(sel) & ~keep).sum())
    rec.setdefault('nms', []).append((float(iou_max) if iou_max is not None else None,
                                      int(np.asarray(sel).sum()), killed))
    return keep

def render_hook(x, y, s_q, s_c, amp, eta):
    rec['n_render'] = len(x); rec['amp_render'] = np.asarray(amp).copy()
    return _render(x, y, s_q, s_c, amp, eta)

sim._visibility, sim._nms, sim._render = vis_hook, nms_hook, render_hook

def run_once(unified):
    rec.clear()
    sim.unified_labels = unified
    out = sim.simulate_img()
    d = dict(rec)
    d['none'] = out is None
    if out is not None:
        _i, bx, _m, rg = out
        d['n_box'] = len(bx); d['n_ring'] = int(np.asarray(rg).sum())
        d['bx'] = np.asarray(bx).copy()
    return d

# ---------------------------------------------------------------- paired sweep
U, O = [], []
tries = 0
while len(U) < args.frames and tries < args.frames * 40:
    tries += 1
    st_py, st_np = random.getstate(), np.random.get_state()
    fm = sim._frames_made
    u = run_once(True)
    random.setstate(st_py); np.random.set_state(st_np); sim._frames_made = fm
    o = run_once(False)
    if 'con' not in u or 'con' not in o:
        continue                                  # frame aborted before the gate in one path
    if len(u['con']) != len(o['con']) or not np.allclose(u['amp'], o['amp']):
        raise SystemExit('paths diverged before the branch -- pairing is invalid')
    U.append(u); O.append(o)
    if len(U) % 25 == 0:
        print(f'  {len(U):4d} pairs ({tries} draws)...', flush=True)

n = len(U)
print(f'\n{n} paired frames from {tries} draws, config {args.config}')
print(f'gate: contrast>={sim.contrast_min} snr>={sim.snr_min} | '
      f'IoU seg {sim.seg_iou_max} ring {sim.ring_iou_max} | max_peaks {sim.max_peaks}\n')

def col(d, k, default=0):
    return np.array([x.get(k, default) if x.get(k, None) is not None else default for x in d], float)

# frames the conventions disagree about keeping at all
u_none = np.array([x['none'] for x in U]); o_none = np.array([x['none'] for x in O])
print('FRAMES DISCARDED  (simulate_img returned None)')
print(f'  unified   {u_none.sum():4d} / {n}      historical {o_none.sum():4d} / {n}')
print(f'  unified discards a frame the historical path keeps: '
      f'{int((u_none & ~o_none).sum())}   (its keep.sum()<3 early return)\n')

ok = ~u_none & ~o_none
pk  = np.array([len(x['con']) for x in U])[ok]
gate = np.array([int(((x['con'] >= sim.contrast_min) & (x['snr'] >= sim.snr_min)).sum())
                 for x in U])[ok]
ub  = col(U, 'n_box')[ok]; ob = col(O, 'n_box')[ok]
ur  = col(U, 'n_ring')[ok]; orr = col(O, 'n_ring')[ok]
uren = col(U, 'n_render')[ok]; oren = col(O, 'n_render')[ok]

def row(tag, x):
    print(f'  {tag:<34s} mean {x.mean():7.1f}  p50 {np.median(x):7.1f}  '
          f'min {x.min():6.0f}  max {x.max():6.0f}')

print(f'PER FRAME, {ok.sum()} frames both conventions kept')
row('peaks drawn inside the mask', pk)
row('pass the brightness gate', gate)
print()
row('PAINTED  unified', uren)
row('PAINTED  historical', oren)
row('BOXES    unified', ub)
row('BOXES    historical', ob)
print()
row('unlabelled-but-painted  unified', uren - ub)
row('unlabelled-but-painted  historical', oren - ob)
print(f'\n  historical: {100*np.mean((oren-ob)/np.maximum(oren,1)):.1f}% of painted peaks carry NO box'
      f'  (median {100*np.median((oren-ob)/np.maximum(oren,1)):.1f}%)')
print(f'  unified:    {100*np.mean((uren-ub)/np.maximum(uren,1)):.1f}% -- 0 by construction\n')

print('BOX COUNT, unified vs historical, same peaks')
d = ub - ob
print(f'  unified keeps {d.mean():+.1f} boxes/frame on average ({100*d.mean()/max(ob.mean(),1e-9):+.1f}%)')
print(f'  rings    unified {ur.mean():6.2f}  historical {orr.mean():6.2f}')
print(f'  segments unified {(ub-ur).mean():6.2f}  historical {(ob-orr).mean():6.2f}')

print('\nWHAT REMOVED THE BOXES  (mean per frame)')
for tag, src in (('unified', U), ('historical', O)):
    ring_k = seg_k = 0.0; cnt = 0
    for x, keep in zip(src, ok):
        if not keep:
            continue
        cnt += 1
        for iou, nsel, killed in x.get('nms', []):
            if iou == sim.ring_iou_max:
                ring_k += killed
            else:
                seg_k += killed
    cnt = max(cnt, 1)
    print(f'  {tag:<12s} gate rejects {np.mean(pk-gate):6.2f}   '
          f'ring NMS {ring_k/cnt:5.2f}   segment NMS {seg_k/cnt:5.2f}')

# brightness of what each convention labels, in units of local noise
uc = np.concatenate([x['con'][((x['con'] >= sim.contrast_min) & (x['snr'] >= sim.snr_min))]
                     for x, k in zip(U, ok) if k])
allc = np.concatenate([x['con'] for x, k in zip(U, ok) if k])
print(f'\nCONTRAST (amp / local noise) of the peaks in the frame')
for tag, x in (('all peaks drawn', allc), ('gate-passing (labelled)', uc)):
    print(f'  {tag:<24s} p10 {np.percentile(x,10):8.2f}  p50 {np.median(x):8.2f}  '
          f'p90 {np.percentile(x,90):8.2f}')
print('  real labelled peaks      p10     1.50  p50     3.10   (measured, organic+41)')
