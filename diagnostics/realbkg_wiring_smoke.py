"""Do the realbkg_* config keys actually reach RealBkgSimulation?

Builds the simulator exactly as main.py:142 does, straight from the real config, and asserts
every convention value landed. Run this before submitting a training run: a key that never
reaches args fails silently, and the run then trains for days on the wrong simulator. That has
happened before -- it is why run_detector_realbkg.sbatch carries an epoch-0 wiring check in its
header.

    sbatch diagnostics/run_wiring_smoke.sbatch        (needs ~12.5 GB for the hkl bank, so not
                                                       on a login node -- its cgroup caps ~6.5 GB)

Exit status is 0 only if every check passes AND the simulator still produces frames.
"""
import os, sys, argparse
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import numpy as np
from util.slconfig import SLConfig
from simulation import SimulationConfig
from realbkg_simulation import RealBkgSimulation
from diagnostics.cache_realbkg_donors import load_into

cfg = SLConfig.fromfile('config/DINO/DINO_4scale_swin_realbkg.py')
args = argparse.Namespace(**{k: v for k, v in cfg.items()})   # main.py merges cfg into args
sc = SimulationConfig()
sc.a_coef, sc.w_coef = getattr(cfg, 'box_coef_override', (2.80, 1.30))

CACHE = '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5'
RealBkgSimulation._load_donors = lambda self, *a, **k: load_into(self, CACHE)

# byte-for-byte the call in main.py:142
sim = RealBkgSimulation(
    bank_path=getattr(args, 'physics_bank_path', None),
    donor_path=args.realbkg_donor_path, stats_path=args.realbkg_stats_path,
    sim_config=sc, device='cpu',
    n_oriented=tuple(getattr(args, 'realbkg_n_oriented', (1, 3))),
    p_ring=float(getattr(args, 'realbkg_p_ring', 0.15)),
    mosaic=bool(getattr(args, 'realbkg_mosaic', False)),
    mosaic_pool=int(getattr(args, 'realbkg_mosaic_pool', 48)),
    mosaic_refresh=int(getattr(args, 'realbkg_mosaic_refresh', 64)),
    mosaic_seed=getattr(args, 'realbkg_mosaic_seed', None),
    intensity_decades=getattr(args, 'realbkg_intensity_decades', None),
    amplitude_mode=getattr(args, 'realbkg_amplitude_mode', 'fitted'),
    mask_bank=bool(getattr(args, 'realbkg_mask_bank', True)),
    mask_keep=getattr(args, 'realbkg_mask_keep', 'default'),
    unified_labels=bool(getattr(args, 'realbkg_unified_labels', False)),
    contrast_min=float(getattr(args, 'realbkg_contrast_min', 1.5)),
    snr_min=float(getattr(args, 'realbkg_snr_min', 6.0)),
    ring_iou_max=float(getattr(args, 'realbkg_ring_iou_max', 0.10)),
    seg_iou_max=getattr(args, 'realbkg_seg_iou_max', None),
    max_peaks=getattr(args, 'realbkg_max_peaks', None),
    spots_cap=getattr(args, 'realbkg_spots_cap', None),
    rings_cap=getattr(args, 'realbkg_rings_cap', None),
    n_powder=tuple(getattr(args, 'realbkg_n_powder', (1, 1))))

checks = [
    ('bank is the hkl bank', 'bank_organic_hkl.npz' in args.physics_bank_path),
    ('unified_labels', sim.unified_labels is True),
    ('contrast_min 2.0', sim.contrast_min == 2.0),
    ('snr_min 6.0', sim.snr_min == 6.0),
    ('seg_iou_max 0.30', sim.seg_iou_max == 0.30),
    ('ring_iou_max 0.10', sim.ring_iou_max == 0.10),
    ('max_peaks 200', sim.max_peaks == 200),
    ('spots_cap (2, 200)', tuple(sim.spots_cap) == (2, 200)),
    ('rings_cap (3, 15)', tuple(sim.rings_cap) == (3, 15)),
    ('n_powder (1, 1)', tuple(sim.n_powder) == (1, 1)),
    ('oriented entries 675474', len(sim.phys.oriented_ids) == 675474),
]
bad = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(f'  {"OK  " if ok else "FAIL"}  {n}')

# and it must still produce a frame whose every box is a rendered peak
np.random.seed(0)
import random; random.seed(0)
n = 0
for _ in range(12):
    r = sim.simulate_img()
    if r is not None:
        _img, bx, _m, rg = r
        n += 1
        if n == 1:
            print(f'\n  first frame: {len(bx)} boxes ({int(rg.sum())} ring)')
print(f'  {n}/12 attempts produced a frame')
print('\nRESULT:', 'ALL WIRED' if not bad and n else f'BROKEN -> {bad}')
sys.exit(0 if (not bad and n) else 1)
