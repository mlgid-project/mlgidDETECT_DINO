"""Does the 70x peak-brightness mismatch survive preprocessing?

Raw measurement: simulated labelled peaks sit at p50 222x local noise, real labelled peaks at 3.1x.
But `CHAIN` ends in histogram equalisation, a RANK transform, so absolute ratios may be largely
discarded before the network sees anything. This measures the same quantity on the PROCESSED image
for both, which is what the detector is actually fed:

    z = (value at the peak centre - local background) / local noise

local background and noise are the median and MAD of a mask-valid annulus around the peak, outside
its own box, so a bright neighbour cannot contribute. If sim and real agree here, the raw 222x is
cosmetic and amplitude_mode is not worth changing; if they disagree, it is a real defect.
"""
import os, sys, argparse, random
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=80)
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

import torch, argparse as _a

def zstats(img, mask, boxes, pad=6):
    """z of each box centre against a mask-valid annulus around it, outside the box."""
    H, W = img.shape
    out = []
    for b in np.asarray(boxes, float):
        x0, y0, x1, y1 = b
        cx, cy = int(round((x0+x1)/2)), int(round((y0+y1)/2))
        if not (0 <= cx < W and 0 <= cy < H) or not mask[cy, cx]:
            continue
        hw, hh = max((x1-x0)/2, 1.0), max((y1-y0)/2, 1.0)
        r0 = int(max(cy-hh-pad, 0)); r1 = int(min(cy+hh+pad, H-1))+1
        c0 = int(max(cx-hw-pad, 0)); c1 = int(min(cx+hw+pad, W-1))+1
        win = img[r0:r1, c0:c1]; wm = mask[r0:r1, c0:c1].copy()
        # knock out the box itself, keep the surrounding annulus
        br0 = int(max(y0, r0))-r0; br1 = int(min(y1, r1-1))-r0+1
        bc0 = int(max(x0, c0))-c0; bc1 = int(min(x1, c1-1))-c0+1
        wm[max(br0,0):max(br1,0), max(bc0,0):max(bc1,0)] = False
        v = win[wm]
        if v.size < 12:
            continue
        bkg = np.median(v)
        mad = np.median(np.abs(v-bkg))*1.4826
        if mad <= 0:
            continue
        out.append((float(img[cy, cx])-bkg)/mad)
    return out

def show(tag, z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if not len(z):
        print(f'  {tag:<34s} (no samples)'); return
    print(f'  {tag:<34s} n {len(z):6d}  p10 {np.percentile(z,10):8.2f}  '
          f'p50 {np.median(z):8.2f}  p90 {np.percentile(z,90):8.2f}')

# ----------------------------------------------------------------- real
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
print('POST-PREPROCESSING z OF LABELLED PEAKS  (what the network sees)\n')
for name, path in (('organic', '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'),
                   ('41', '/mnt/lustre/work/schreiber/szb389/datasets/41.h5')):
    cfg = Config(); cfg.INPUT_DATASET = path
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1; cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    z = []
    for ic in ds.iter_images():
        img = np.asarray(ic.converted_polar_image)[0, 0].astype(float)
        msk = img > 1e-6
        b = np.asarray(ic.polar_labels.boxes, float)
        if b.ndim == 2 and len(b):
            z += zstats(img, msk, b)
    show(f'REAL {name}', z)

# ----------------------------------------------------------------- sim
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

cap = {}
_ac = RS.apply_contrast
def hook(total, mask, chain):
    img = _ac(total, mask, chain)
    cap['img'] = np.asarray(img).copy(); cap['m'] = np.asarray(mask).copy().astype(bool)
    return img
RS.apply_contrast = hook

z, n = [], 0
while n < args.frames:
    cap.clear()
    out = sim.simulate_img()
    if out is None or 'img' not in cap:
        continue
    _i, bx, _m, rg = out
    b = np.asarray(bx, float)
    if len(b) == 0:
        continue
    n += 1
    z += zstats(cap['img'], cap['m'], b)
    if n % 20 == 0:
        print(f'    {n} sim frames...', flush=True)
show(f'SIM ({os.path.basename(args.config)})', z)
print('\nRAW, for reference: sim labelled peaks p50 222x local noise, real p50 3.1x.')
print('If the SIM and REAL rows above are close, histogram equalisation absorbs that gap')
print('and amplitude_mode is NOT the defect. If they are far apart, it is.')
