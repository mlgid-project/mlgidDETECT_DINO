"""Does the model actually see new images each epoch?

A randomisation failure would explain the decay in [[realbkg-overfits-its-backgrounds]] far better
than background diversity does, and this codebase has had exactly that bug before: main.py's
comment records dino_physics3_1 resuming at epoch 41 and regenerating epochs 0-40 verbatim, so it
trained 82 epochs on ~41,000 images seen twice.

Replicates main.py's construction and its per-epoch reseed exactly
(`_epoch_seed = seed + 1000*(epoch+1)`, seed = args.seed + rank = 42), builds the dataset ONCE as
main.py now does, and then for several consecutive epochs hashes:
  * the final preprocessed image, for frame-level repetition
  * the donor BACKGROUND array handed to _compose, for how many distinct backgrounds an epoch sees
  * which bank entries were drawn, for peak-configuration repetition

Reports repeats within an epoch and overlap between epochs.
"""
import os, sys, argparse, random, hashlib
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg_r3.py')
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--per-epoch', type=int, default=40)
ap.add_argument('--base-seed', type=int, default=42)
args = ap.parse_args()

import torch, argparse as _a
from util.slconfig import SLConfig
from simulation import SimulationConfig
from realbkg_simulation import RealBkgSimulation
from diagnostics.cache_realbkg_donors import load_into

cfg = SLConfig.fromfile(args.config)
a = _a.Namespace(**{k: v for k, v in cfg.items()})
sc = SimulationConfig(); sc.a_coef, sc.w_coef = getattr(cfg, 'box_coef_override', (2.80, 1.30))
RealBkgSimulation._load_donors = lambda self, *x, **k: load_into(
    self, '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')

# main.py seeds ONCE at process start, then reseeds per epoch inside the loop
seed = args.base_seed
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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

H = lambda arr: hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]

grab = {}
_compose = sim._compose
def compose_hook(bkg, peaks, mask, coef):
    grab['bkg'] = H(bkg)
    return _compose(bkg, peaks, mask, coef)
sim._compose = compose_hook
_draw = sim._draw_peaks
def draw_hook(qmax):
    out = _draw(qmax)
    grab['ents'] = tuple(getattr(sim, 'last_entries', ()) or ())
    return out
sim._draw_peaks = draw_hook

per_epoch = {}
for ep in range(args.epochs):
    _es = seed + 1000 * (ep + 1)          # exactly main.py:603-606
    random.seed(_es); np.random.seed(_es % (2**32)); torch.manual_seed(_es)
    imgs, bkgs = [], []
    while len(imgs) < args.per_epoch:
        grab.clear()
        out = sim.simulate_img()
        if out is None:
            continue
        imgs.append(H(np.asarray(out[0])))
        bkgs.append(grab.get('bkg', 'NA'))
    per_epoch[ep] = (imgs, bkgs)
    print(f'  epoch {ep}: seed {_es}  {len(set(imgs))}/{len(imgs)} distinct images, '
          f'{len(set(bkgs))} distinct backgrounds', flush=True)

print('\nIMAGE-LEVEL REPETITION')
allimg = [h for ep in per_epoch for h in per_epoch[ep][0]]
print(f'  total frames {len(allimg)}, distinct {len(set(allimg))}')
for i in range(args.epochs):
    for j in range(i+1, args.epochs):
        ov = set(per_epoch[i][0]) & set(per_epoch[j][0])
        print(f'  epoch {i} vs {j}: {len(ov)} identical frames'
              + ('   <-- REPEAT' if ov else ''))

print('\nBACKGROUND REUSE  (the 48-slot mosaic pool)')
allbkg = [h for ep in per_epoch for h in per_epoch[ep][1]]
print(f'  distinct backgrounds across all {len(allbkg)} frames: {len(set(allbkg))}')
for i in range(args.epochs):
    for j in range(i+1, args.epochs):
        ov = set(per_epoch[i][1]) & set(per_epoch[j][1])
        print(f'  epoch {i} vs {j}: {len(ov)} shared backgrounds')
print(f'\n  At 1000 images/epoch and mosaic_refresh={sim.mosaic_refresh}, a pool slot is replaced')
print(f'  ~{1000//max(sim.mosaic_refresh,1)} times per epoch out of {len(sim.bkg)} slots.')
