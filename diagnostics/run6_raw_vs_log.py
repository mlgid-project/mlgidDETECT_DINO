"""Five fresh run-6 training frames, NO boxes: raw linear counts | log10 counts, side by side.

Builds the simulator exactly as main.py does from the r6 config (construction copied from
v2_smoke.py) and captures the frame `_compose` returns -- the linear image BEFORE the contrast
chain (clip/log/HE) that the network actually receives.
"""
import os, sys, time, random, hashlib
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import torch, argparse as _a
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
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

grab = {}
_c = sim._compose
def hook(bkg, peaks, mask, coef):
    t = _c(bkg, peaks, mask, coef); grab['t'] = t; grab['m'] = mask; return t
sim._compose = hook

frames = []
while len(frames) < 5:
    grab.clear()
    r = sim.simulate_img()
    if r is None:
        continue
    frames.append((grab['t'].copy(), np.asarray(grab['m']).astype(bool), len(r[1]), int(r[3].sum())))

OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/10_run6_raw_vs_log'
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(5, 2, figsize=(16, 20))
for i, (t, m, nb, nr) in enumerate(frames):
    good = t[m & (t > 0)]
    print(f'frame {i}: counts p1 {np.percentile(good,1):.0f}  median {np.median(good):.0f}  '
          f'p99.5 {np.percentile(good,99.5):.0f}  max {t.max():.3g}  | {nb} labelled peaks, {nr} rings')
    a = ax[i, 0]
    im = a.imshow(t, cmap='magma', aspect='auto', vmin=0, vmax=t.max())
    a.set_title(f'frame {i+1} - RAW linear counts (0 to max {t.max():.2g})', fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.025, pad=0.01)
    a = ax[i, 1]
    v = np.where(m, np.log10(np.maximum(t, 1.0)), np.nan)
    im = a.imshow(v, cmap='magma', aspect='auto', vmin=np.percentile(np.log10(np.maximum(good, 1)), 1),
                  vmax=np.log10(max(t.max(), 10)))
    a.set_title(f'frame {i+1} - log10 counts  ({nb} labelled peaks, {nr} rings)', fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.025, pad=0.01, label='log10 counts')
    for a in ax[i]:
        a.set_xticks([]); a.set_yticks([])
fig.suptitle('Run 6 simulator (config r6, v2 background) - 5 fresh frames, no boxes\n'
             'left: raw linear counts, right: log10 counts (masked area blank). '
             'Neither is the network input, which adds clip + histogram equalisation.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(f'{OUT}/run6_raw_vs_log.png', dpi=100)
np.savez_compressed(f'{OUT}/run6_raw_frames.npz', *[f[0].astype(np.float32) for f in frames])
print('wrote run6_raw_vs_log.png + run6_raw_frames.npz')
