"""Are the realbkg sim's peaks too BLURRY, or its rings too CROWDED, compared with real 41?

The realbkg models find 41's peaks (uncapped recall 0.84-0.87) but score them barely above their
false positives (TP-FP separation 0.11-0.19; legacy-sim lr4e5_1 0.65, physics4 0.40), with the FP
excess concentrated in ring-heavy frames (realbkg-41-gap-split). The user's two hypotheses:
  (1) the simulated Gaussians are too blurry, so the model never learns what a sharp 41 ring is
  (2) simulated rings overlap too much with other features, so it learns to be unsure about them

Measured on the image the NETWORK SEES (after preprocessing), for real 41, real organic, and the
three training simulators -- legacy (lr4e5_1: 41 0.761), physics (physics4_1: 0.642, same CIF bank
as realbkg on synthetic backgrounds) and realbkg run 6 (0.38-0.44):

  box geometry      radial width (x) of ring and segment boxes; segment chi height
  visible shape     a radial profile through each box (median over its valid chi rows):
                    FWHM in px, FWHM / box width, peak contrast above the profile's flanks,
                    edge steepness (max |d profile / dx| / contrast, i.e. 1/edge length)
  crowding          per ring: other GT boxes whose radial span overlaps it, fraction of its chi
                    extent covered by them, and the edge-to-edge radial gap to the nearest ring

Rings are GEOMETRIC everywhere (>= 70% of the valid chi rows at their radius): 41.h5 never sets
is_ring. No model, CPU only.
"""
import os, sys, argparse, json, random, time
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import torch
import argparse as _a

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--only', default='real41,organic,legacy,physics4,realbkg6')
ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg_r6.py')
args = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset
CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
RUNS = f'{CUR}/DINO_BACKBONE_curation/detector_runs'


def is_ring_geom(boxes, mask, frac=0.70):
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes):
        xc = int(np.clip((x0 + x1) / 2, 0, mask.shape[1] - 1))
        out.append((y1 - y0) >= frac * max(int(mask[:, xc].sum()), 1))
    return np.array(out, bool)


# ------------------------------------------------------------------------------- frame sources
def real(path, pygid):
    c = Config(); c.PREPROCESSING_POLAR_SHAPE = [512, 1024]; c.INPUT_DATASET = path
    ds = (PyGIDDataset(c, path=path, preprocess_func=standard_preprocessing, buffer_size=3, load_labels=True)
          if pygid else H5GIWAXSDataset(c, path=path, preprocess_func=standard_preprocessing, buffer_size=3))
    try:
        for ic in ds.iter_images():
            yield (np.asarray(ic.converted_polar_image, np.float32).reshape(-1, 512, 1024)[0],
                   np.asarray(ic.converted_mask).reshape(512, 1024).astype(bool),
                   np.asarray(ic.polar_labels.boxes, np.float32).reshape(-1, 4))
    finally:
        if callable(getattr(ds, 'close', None)):
            ds.close()


def from_sim(sim, n):
    k = 0
    while k < n:
        r = sim.simulate_img()
        if r is None:
            continue
        img, bx, m, _ = r
        k += 1
        yield (np.asarray(img.detach().cpu(), np.float32).reshape(512, 1024),
               np.asarray(m.detach().cpu()).reshape(512, 1024).astype(bool),
               np.asarray(bx.detach().cpu(), np.float32).reshape(-1, 4))


def sim_cfg():
    from simulation import SimulationConfig
    c = SimulationConfig(); c.a_coef, c.w_coef = 2.8, 1.3
    return c


def legacy(n):
    from simulation import FastSimulation
    return from_sim(FastSimulation(sim_config=sim_cfg(), device='cpu'), n)


def physics4(n):
    from physics_simulation import PhysicsSimulation
    a = json.load(open(f'{RUNS}/dino_physics4_1/config_args_all.json'))
    sim = PhysicsSimulation(a['physics_bank_path'], sim_config=sim_cfg(), device='cpu',
                            unify_contrast=bool(a.get('unify_contrast', False)),
                            n_powder=a.get('physics_n_powder'),
                            real_tail_only=bool(a.get('real_tail_only', False)),
                            frame_types=a.get('physics_frame_types'))
    return from_sim(sim, n)


def build_realbkg(config='config/DINO/DINO_4scale_swin_realbkg_r6.py', overrides=None, device='cpu'):
    """The realbkg simulator exactly as main.py constructs it from `config` (+ overrides)."""
    from util.slconfig import SLConfig
    from realbkg_simulation import RealBkgSimulation
    from diagnostics.cache_realbkg_donors import load_into
    cfg = SLConfig.fromfile(config)
    a = _a.Namespace(**{k: v for k, v in cfg.items()})
    for k, v in (overrides or {}).items():
        setattr(a, k, v)
    RealBkgSimulation._load_donors = lambda self, *x, **k: load_into(
        self, f'{CUR}/realbkg_donors_selected.h5')
    return RealBkgSimulation(
        bank_path=a.physics_bank_path, donor_path=a.realbkg_donor_path, stats_path=a.realbkg_stats_path,
        sim_config=sim_cfg(), device=device,
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
        v2_n_pc=int(getattr(a, 'realbkg_v2_n_pc', 6)),
        seg_on_ring_keep=float(getattr(a, 'realbkg_seg_on_ring_keep', 1.0)),
        ring_w_coef=getattr(a, 'realbkg_ring_w_coef', None),
        ring_sq_scale=float(getattr(a, 'realbkg_ring_sq_scale', 1.0)),
        ring_min_gap=float(getattr(a, 'realbkg_ring_min_gap', 0.0)),
        ring_chi_texture=float(getattr(a, 'realbkg_ring_chi_texture', 0.0)),
        ring_chi_corr=tuple(getattr(a, 'realbkg_ring_chi_corr', (3.0, 20.0))))


def realbkg6(n, config='config/DINO/DINO_4scale_swin_realbkg_r6.py', overrides=None):
    return from_sim(build_realbkg(config, overrides), n)



# --------------------------------------------------------------------------------- measurement
def profile_stats(img, mask, b):
    """Radial profile through box b: median over the box's valid chi rows, over x in the box
    padded by max(8, 1.5*width) each side. Returns fwhm, contrast, steepness, or None."""
    x0, y0, x1, y1 = b
    w = max(x1 - x0, 1.0)
    pad = max(8.0, 1.5 * w)
    xa, xb = int(max(0, np.floor(x0 - pad))), int(min(img.shape[1], np.ceil(x1 + pad)))
    ya, yb = int(max(0, np.floor(y0))), int(min(img.shape[0], np.ceil(y1)))
    if xb - xa < 6 or yb - ya < 1:
        return None
    sub, sm = img[ya:yb, xa:xb], mask[ya:yb, xa:xb]
    cols = sm.mean(0) > 0.5
    rows = sm.all(1)
    if rows.sum() < 1 or cols.mean() < 0.8:
        return None
    prof = np.median(sub[rows], axis=0)
    ia, ib = int(np.floor(x0)) - xa, int(np.ceil(x1)) - xa
    ia, ib = max(ia, 0), min(ib, len(prof))
    if ib <= ia:
        return None
    flank = np.concatenate([prof[:max(ia, 1)], prof[min(ib, len(prof) - 1):]])
    base = np.percentile(flank, 30)
    pk_i = ia + int(np.argmax(prof[ia:ib]))
    con = prof[pk_i] - base
    if con <= 0.02:
        return dict(fwhm=np.nan, con=con, steep=np.nan, w=w)
    half = base + 0.5 * con
    l = pk_i
    while l > 0 and prof[l - 1] >= half:
        l -= 1
    r = pk_i
    while r < len(prof) - 1 and prof[r + 1] >= half:
        r += 1
    steep = np.max(np.abs(np.diff(prof[max(l - 3, 0):r + 4]))) / con if r - l < len(prof) else np.nan
    return dict(fwhm=float(r - l + 1), con=float(con), steep=float(steep), w=w)


def crowding(gt, ring, mask):
    out = []
    for i in np.where(ring)[0]:
        g = gt[i]
        others = np.delete(np.arange(len(gt)), i)
        ov = others[(gt[others, 0] < g[2]) & (gt[others, 2] > g[0])]
        ov_seg = ov[~ring[ov]]
        cov = np.zeros(512, bool)
        for j in ov_seg:
            cov[int(max(gt[j, 1], 0)):int(min(np.ceil(gt[j, 3]), 512))] = True
        xc = int(np.clip((g[0] + g[2]) / 2, 0, 1023))
        valid = mask[:, xc]
        cov_frac = (cov & valid).sum() / max(valid.sum(), 1)
        rr = gt[ring]
        gaps = [max(r[0] - g[2], g[0] - r[2]) for r in rr if not np.allclose(r, g)]
        out.append(dict(n_ov=len(ov), n_ov_seg=len(ov_seg), cov=float(cov_frac),
                        gap=float(min(gaps)) if gaps else np.nan, w=float(g[2] - g[0])))
    return out


def P(v, ps=(10, 50, 90)):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return '      n/a'
    return '  '.join(f'p{p} {np.percentile(v, p):7.2f}' for p in ps) + f'   n {len(v)}'


def measure(tag, frames):
    t0 = time.time()
    R, S, C = [], [], []
    nf = nr = ns = 0
    for img, m, gt in frames:
        nf += 1
        if not len(gt):
            continue
        ring = is_ring_geom(gt, m)
        nr += ring.sum(); ns += (~ring).sum()
        for b, isr in zip(gt, ring):
            st = profile_stats(img, m, b)
            if st is None:
                continue
            if not isr:
                st['h'] = float(b[3] - b[1])
            (R if isr else S).append(st)
        C += crowding(gt, ring, m)
    print(f'\n===== {tag}: {nf} frames, {nr/nf:.1f} rings + {ns/nf:.1f} segments per frame '
          f'({time.time()-t0:.0f} s)')
    for name, L in (('RINGS', R), ('SEGMENTS', S)):
        print(f'  {name}')
        print(f'    box radial width px     {P([d["w"] for d in L])}')
        if name == 'SEGMENTS':
            print(f'    box chi height px       {P([d["h"] for d in L])}')
        print(f'    visible FWHM px         {P([d["fwhm"] for d in L])}')
        print(f'    FWHM / box width        {P([d["fwhm"]/d["w"] for d in L])}')
        print(f'    contrast (0-1 image)    {P([d["con"] for d in L])}')
        print(f'    edge steepness /px      {P([d["steep"] for d in L])}')
    print('  RING CROWDING')
    print(f'    other boxes overlapping radially     {P([d["n_ov"] for d in C])}')
    print(f'    of which segments                     {P([d["n_ov_seg"] for d in C])}')
    print(f'    ring chi extent covered by segments   {P([d["cov"] for d in C])}')
    print(f'    gap to nearest ring px (edge-edge)    {P([d["gap"] for d in C])}')
    print(f'    rings with ANY segment on them        '
          f'{np.mean([d["n_ov_seg"] > 0 for d in C]):.0%}' if C else '    no rings', flush=True)


SRC = {'real41': lambda: real(f'{CUR}/41.h5', False),
       'organic': lambda: real(f'{CUR}/organic_labeled.h5', True),
       'legacy': lambda: legacy(args.frames),
       'physics4': lambda: physics4(args.frames),
       'realbkg6': lambda: realbkg6(args.frames, config=args.config)}
if __name__ == '__main__':
  for name in args.only.split(','):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    measure(name, SRC[name]())
