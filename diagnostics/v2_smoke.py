"""Pre-flight for the background-v2 run: wiring, per-frame cost, and freshness.

Builds the simulator exactly as main.py does from the r6 config, then checks the three things
that would waste a 72 h allocation: that v2 is actually on, that a frame still costs something
sane now the background is rebuilt per frame, and that consecutive frames really do get different
backgrounds. Also renders a few finished frames so the peaks can be seen sitting on the new
backgrounds rather than judged from statistics alone.
"""
import os, sys, time, random, hashlib
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import torch, argparse as _a
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(11); np.random.seed(11); torch.manual_seed(11)
from util.slconfig import SLConfig
from simulation import SimulationConfig
import realbkg_simulation as RS
from realbkg_simulation import RealBkgSimulation
from diagnostics.cache_realbkg_donors import load_into

CFG = 'config/DINO/DINO_4scale_swin_realbkg_r6.py'
cfg = SLConfig.fromfile(CFG)
a = _a.Namespace(**{k: v for k, v in cfg.items()})
sc = SimulationConfig(); sc.a_coef, sc.w_coef = getattr(cfg, 'box_coef_override', (2.80, 1.30))
RealBkgSimulation._load_donors = lambda self, *x, **k: load_into(
    self, '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')

t0 = time.perf_counter()
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
    seg_wide_sigma=getattr(a, 'realbkg_seg_wide_sigma', ((8.1, 0.45), (2.9, 0.45))),
    bkg_v2=bool(getattr(a, 'realbkg_bkg_v2', False)),
    surrogate_frac=float(getattr(a, 'realbkg_surrogate_frac', 0.0)),
    v2_canvas=tuple(getattr(a, 'realbkg_v2_canvas', (1536, 3072))),
    v2_refresh=int(getattr(a, 'realbkg_v2_refresh', 200)),
    v2_n_pc=int(getattr(a, 'realbkg_v2_n_pc', 6)))
print(f'construction {time.perf_counter()-t0:.1f} s\n')

assert sim.bkg_v2, 'bkg_v2 did NOT reach the simulator'
from realbkg_sim.mosaic_background import MosaicBackground
assert MosaicBackground.USE_ENV_BANK is False, 'envelope still drawn from the rejected 189 bank'
print('WIRING OK: bkg_v2 on, USE_ENV_BANK False, surrogate_frac '
      f'{sim.surrogate_frac:.0%}, canvas {sim.v2_canvas}, refresh {sim.v2_refresh}\n')

# per-frame cost and background freshness
grab = {}
_c = sim._compose
def hook(bkg, peaks, mask, coef):
    grab['b'] = hashlib.md5(np.ascontiguousarray(bkg).tobytes()).hexdigest()[:12]
    return _c(bkg, peaks, mask, coef)
sim._compose = hook

out, hs, ts = [], [], []
while len(out) < 24:
    grab.clear(); t = time.perf_counter()
    r = sim.simulate_img()
    if r is None:
        continue
    ts.append(time.perf_counter()-t); hs.append(grab.get('b')); out.append(r)
ts = np.array(ts)
print(f'PER-FRAME COST  median {np.median(ts)*1000:7.1f} ms   mean {ts.mean()*1000:7.1f} ms')
print(f'  (v1 was ~300 ms/frame with the background taken from a cached pool)')
print(f'BACKGROUND FRESHNESS  {len(set(hs))}/{len(hs)} distinct backgrounds in consecutive frames\n')
nb = np.array([len(r[1]) for r in out]); nr = np.array([int(r[3].sum()) for r in out])
print(f'COMPOSITION  boxes/frame p50 {np.median(nb):.0f}  rings/frame mean {nr.mean():.2f}  '
      f'ring-free {100*np.mean(nr == 0):.0f}%')

OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/09_background_v2'
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(2, 3, figsize=(16, 6))
for i, axx in enumerate(ax.ravel()):
    img, bx, mk, rg = out[i]
    axx.imshow(np.asarray(img), cmap='magma', aspect='auto', vmin=0, vmax=1)
    for b, isr in zip(np.asarray(bx), np.asarray(rg)):
        axx.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                    ec=('#7CFC00' if isr else '#00E5FF'), lw=0.7))
    axx.set_xticks([]); axx.set_yticks([])
    axx.set_title(f'{len(bx)} boxes, {int(rg.sum())} rings', fontsize=9)
fig.suptitle('Run 6 simulator: physics peaks on the v2 background (cyan = segment, green = ring)',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT}/run6_frames.png', dpi=105)
print('\nwrote run6_frames.png')
