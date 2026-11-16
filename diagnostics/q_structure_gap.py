"""Where in q do the sim's peaks sit, against each real gate?

The 41 gap tracks the BANK, not the backgrounds: 41 scores 0.761 with parametric peaks on
synthetic backgrounds (lr4e5_1), 0.567-0.644 with CIF peaks on the SAME synthetic backgrounds
(physics3_2/4_1/5_1), and 0.39-0.43 with CIF peaks on real ones. And the recorded failure mode is
PRECISION at equal recall -- physics-trained models over-predict on 41.

The bank is `cif_library_organic`, an ORGANIC CIF library. 41 is a 2D perovskite. Different
lattices put reflections at different q and, more importantly, at different SPACINGS. The model
sees pixels, so the quantity that matters is the box centre's x (= q/qmax * WIDTH) and the
same-frame nearest-neighbour spacing in x -- the lattice signature it learns to expect.

Measures, for real organic / real 41 / the simulator:
  * marginal distribution of box-centre x
  * nearest-neighbour |dx| to another peak in the SAME frame (all peaks, and segments only)
  * the same for rings alone, since 41 is 42% rings
No model, no GPU.
"""
import os, sys, argparse, random
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg_r3.py')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

RING_FRAC = 0.70
WIDTH = 1024

def nn_gaps(xs):
    """nearest-neighbour |dx| within one frame"""
    if len(xs) < 2:
        return []
    s = np.sort(np.asarray(xs, float))
    d = np.diff(s)
    g = np.minimum(np.r_[d, np.inf], np.r_[np.inf, d])
    return list(g[np.isfinite(g)])

def summarise(tag, x, gap_all, gap_seg, gap_ring, nring, ntot):
    f = lambda v, q: np.percentile(v, q) if len(v) else float('nan')
    print(f'  {tag}')
    print(f'    box-centre x        p10 {f(x,10):7.1f}  p50 {f(x,50):7.1f}  p90 {f(x,90):7.1f}'
          f'   (0-1023 = q/qmax)')
    print(f'    NN gap, all peaks   p10 {f(gap_all,10):7.1f}  p50 {f(gap_all,50):7.1f}  '
          f'p90 {f(gap_all,90):7.1f}  n {len(gap_all)}')
    print(f'    NN gap, segments    p10 {f(gap_seg,10):7.1f}  p50 {f(gap_seg,50):7.1f}  '
          f'p90 {f(gap_seg,90):7.1f}')
    print(f'    NN gap, rings       p10 {f(gap_ring,10):7.1f}  p50 {f(gap_ring,50):7.1f}  '
          f'p90 {f(gap_ring,90):7.1f}')
    print(f'    rings {100*nring/max(ntot,1):.0f}% of {ntot} boxes')

# ------------------------------------------------------------------ real
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
print('Q-SPACE STRUCTURE OF THE LABELS\n')
for name, path in (('REAL organic', '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'),
                   ('REAL 41', '/mnt/lustre/work/schreiber/szb389/datasets/41.h5')):
    cfg = Config(); cfg.INPUT_DATASET = path
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1; cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    X, GA, GS, GR, nr, nt = [], [], [], [], 0, 0
    for ic in ds.iter_images():
        img = np.asarray(ic.converted_polar_image)[0, 0]
        span = (img > 1e-6).sum(0).astype(float)
        b = np.asarray(ic.polar_labels.boxes, float)
        if b.ndim != 2 or len(b) == 0:
            continue
        xc = (b[:, 0]+b[:, 2])/2
        h = b[:, 3]-b[:, 1]
        col = np.clip(xc.astype(int), 0, WIDTH-1)
        rg = h >= RING_FRAC*np.maximum(span[col], 1.0)
        X += list(xc); nr += int(rg.sum()); nt += len(b)
        GA += nn_gaps(xc); GS += nn_gaps(xc[~rg]); GR += nn_gaps(xc[rg])
    summarise(name, np.array(X), GA, GS, GR, nr, nt)
    print()

# ------------------------------------------------------------------ sim
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

X, GA, GS, GR, nr, nt, n = [], [], [], [], 0, 0, 0
while n < args.frames:
    out = sim.simulate_img()
    if out is None:
        continue
    _i, bx, _m, rg = out
    b = np.asarray(bx, float); r = np.asarray(rg, bool)
    if len(b) == 0:
        continue
    n += 1
    xc = (b[:, 0]+b[:, 2])/2
    X += list(xc); nr += int(r.sum()); nt += len(b)
    GA += nn_gaps(xc); GS += nn_gaps(xc[~r]); GR += nn_gaps(xc[r])
    if n % 30 == 0:
        print(f'    {n} sim frames...', flush=True)
summarise(f'SIM ({os.path.basename(args.config)})', np.array(X), GA, GS, GR, nr, nt)
