"""Dump N frames of the REAL-BACKGROUND simulator as a pyGID/NeXus file, RAW (pre-contrast).

WHY RAW. `RealBkgSimulation.simulate_img()` returns the model input, which is
apply_contrast(total, mask, {'clip': (5,99.5), 'log': True, 'he': True}) -- rank-preserving but
with every physical intensity destroyed. This script captures `total` itself, the linear frame
    total = donor_background + peaks + counting_noise(peaks)
by intercepting apply_contrast, so what lands in the file is exactly what the simulator built
before any contrast step.

WHAT IS IN THE FILE, per entry:
  data/img_gid_q                        reciprocal-space raw image (inverse polar resampling)
  data/q_xy, data/q_z                   axes, set so PyGIDDataset recovers the donor's own q_max
  data/analysis/frame00000/fitted_peaks pygid PEAK_DTYPE record -- the GT boxes
  polar/image, polar/mask               THE LOSSLESS ORIGINAL: the raw polar frame, untouched
  polar/boxes, polar/is_ring            GT boxes as xyxy polar pixels
  polar/amplitude, polar/sigma_q, polar/sigma_chi

The polar group is the authoritative copy. The reciprocal conversion is lossy by construction --
a polar rectangle reaches r_max at every chi but the reciprocal square only reaches
r = (n-1)/max(|cos chi|, |sin chi|), so ~18% of the frame (the high-q wedge near chi = 0 and 90)
has no home and is zero. Real detector frames have the same property. Use `polar/` to judge the
simulator; use `img_gid_q` when you want the file to load through the normal pygid path.

BOX CONVENTION. box FULL extent = coef * sigma, coef = (a_coef, w_coef) = (2.80, 1.30) for
(chi, q). radius_width / angle_width in fitted_peaks are those full extents in q / degrees, so a
viewer that draws radius +- radius_width/2 reproduces the GT box exactly.

  python diagnostics/dump_realbkg_pygid.py [--frames 20] [--n 1024] [--out PATH] [--seed 0]
"""
import os, sys, json, argparse, datetime
import numpy as np
import cv2
import h5py
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

HEIGHT, WIDTH = 512, 1024

from diagnostics.dump_sim_pygid import PEAK_DTYPE, polar_to_reciprocal


def boxes_to_peaks(boxes, is_ring, q_max, amp, s_q, s_c):
    """Polar pixel boxes -> fitted_peaks. Inverse of util/pygidloader.py:_load_fittedpeaks."""
    p = np.zeros(len(boxes), dtype=PEAK_DTYPE)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    p['radius']       = (x0 + x1)/2/WIDTH*q_max
    p['radius_width'] = (x1 - x0)/WIDTH*q_max          # FULL extent = w_coef * sigma_q
    p['angle']        = (y0 + y1)/2*90.0/HEIGHT
    p['angle_width']  = (y1 - y0)*90.0/HEIGHT          # FULL extent = a_coef * sigma_chi
    p['q_xy'] = p['radius']*np.cos(np.radians(p['angle']))
    p['q_z']  = p['radius']*np.sin(np.radians(p['angle']))
    p['theta'] = p['angle']
    p['is_ring'] = is_ring
    p['visibility'] = 3                                 # ground truth
    p['score'] = 1.0
    p['amplitude'] = amp
    p['A'] = s_q                                        # rendered sigma, polar px
    p['B'] = s_c
    p['id'] = np.arange(len(boxes))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--n', type=int, default=1024, help='reciprocal grid size (real files: 1641)')
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/datasets/realbkgsim_raw_20.h5')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--donor-cache',
                    default='/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    import realbkg_simulation as RS
    from realbkg_simulation import RealBkgSimulation

    cfg = SLConfig.fromfile(os.path.join(_REPO, args.config))
    sc = SimulationConfig()
    coefs = getattr(cfg, 'box_coef_override', None) or (2.80, 1.30)
    sc.a_coef, sc.w_coef = float(coefs[0]), float(coefs[1])
    print(f'box convention: a_coef (chi) = {sc.a_coef}, w_coef (q) = {sc.w_coef}')

    # Skip the 444-donor scan when a cache of the SELECTED pool exists: _load_donors is where
    # essentially all the startup cost lives (see diagnostics/cache_realbkg_donors.py).
    from diagnostics.cache_realbkg_donors import load_into, DEFAULT as DONOR_CACHE
    _ld = RealBkgSimulation._load_donors
    if os.path.exists(args.donor_cache):
        RealBkgSimulation._load_donors = lambda self, *a, **k: load_into(self, args.donor_cache)

    sim = RealBkgSimulation(
        bank_path=cfg.physics_bank_path, donor_path=cfg.realbkg_donor_path,
        stats_path=cfg.realbkg_stats_path, sim_config=sc, device='cpu',
        n_oriented=tuple(getattr(cfg, 'realbkg_n_oriented', (1, 3))),
        p_ring=float(getattr(cfg, 'realbkg_p_ring', 0.15)))
    RealBkgSimulation._load_donors = _ld

    # ---- intercept the pre-contrast frame and the per-peak parameters -------------
    snap = {}
    _ac = RS.apply_contrast
    def _ac_hook(total, mask, chain):
        snap['total'] = np.asarray(total).copy(); snap['mask'] = np.asarray(mask).copy()
        return _ac(total, mask, chain)
    RS.apply_contrast = _ac_hook

    _render = RealBkgSimulation._render
    def _render_hook(self, x, y, s_q, s_c, amp, eta):
        snap['cand'] = dict(x=np.asarray(x).copy(), y=np.asarray(y).copy(),
                            s_q=np.asarray(s_q).copy(), s_c=np.asarray(s_c).copy(),
                            amp=np.asarray(amp).copy(), eta=float(eta))
        return _render(self, x, y, s_q, s_c, amp, eta)
    RealBkgSimulation._render = _render_hook

    _draw = RealBkgSimulation._draw_peaks
    def _draw_hook(self, qmax):
        snap['qmax'] = float(qmax)
        return _draw(self, qmax)
    RealBkgSimulation._draw_peaks = _draw_hook

    def match(boxes, is_ring, c, a_coef, w_coef):
        """Recover per-peak amp/sigma for the KEPT boxes by rebuilding candidate boxes."""
        hw, hh = w_coef*c['s_q']/2.0, a_coef*c['s_c']/2.0
        cb = np.stack([c['x']-hw, c['y']-hh, c['x']+hw, c['y']+hh], 1).astype(np.float32)
        out = np.full(len(boxes), -1, int)
        used = np.zeros(len(cb), bool)
        for i, b in enumerate(boxes):
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            d = np.abs((cb[:, 0]+cb[:, 2])/2 - cx)
            if not is_ring[i]:                       # rings have y overwritten to the full height
                d = d + np.abs((cb[:, 1]+cb[:, 3])/2 - cy)
            d[used] = np.inf
            j = int(np.argmin(d)); out[i] = j; used[j] = True
        return out

    frames = []
    while len(frames) < args.frames:
        snap.clear()
        r = sim.simulate_img()
        if r is None:
            continue
        _img, boxes, mask, is_ring = r
        boxes = boxes.cpu().numpy(); is_ring = is_ring.cpu().numpy()
        pol = snap['total'].astype(np.float32)
        m = snap['mask'].astype(bool)
        c = snap['cand']
        j = match(boxes, is_ring, c, sc.a_coef, sc.w_coef)
        frames.append(dict(pol=pol, mask=m, boxes=boxes, is_ring=is_ring, qmax=snap['qmax'],
                           amp=c['amp'][j].astype(np.float32), s_q=c['s_q'][j].astype(np.float32),
                           s_c=c['s_c'][j].astype(np.float32), eta=c['eta']))
        f = frames[-1]
        print(f"  frame {len(frames)-1:2d}: {len(boxes):3d} boxes ({int(is_ring.sum()):2d} ring) "
              f"q_max {f['qmax']:.2f}  I range [{pol[m].min():.3g}, {pol[m].max():.3g}]  "
              f"eta {f['eta']:.2f}", flush=True)

    RS.apply_contrast = _ac
    RealBkgSimulation._render = _render
    RealBkgSimulation._draw_peaks = _draw

    n = args.n
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, 'w') as f:
        for i, fr in enumerate(frames):
            qmax = fr['qmax']
            q_axis = np.arange(n, dtype=np.float64)/(n-1)*(qmax/np.sqrt(2.0))
            rec = polar_to_reciprocal(fr['pol'], n)

            e = f.create_group(f'entry_sim{i:02d}'); e.attrs['NX_class'] = 'NXentry'
            e.create_dataset('definition', data='NXgid')
            d = e.create_group('data'); d.attrs['NX_class'] = 'NXdata'
            d.attrs['signal'] = 'img_gid_q'
            d.attrs['axes'] = np.array(['frame_num', 'q_z', 'q_xy'], dtype=object)
            d.create_dataset('img_gid_q', data=rec[None].astype(np.float32),
                             compression='gzip', compression_opts=4)
            d.create_dataset('q_xy', data=q_axis)
            d.create_dataset('q_z', data=q_axis)
            d.create_dataset('frame_num', data=np.array([0], dtype=np.int64))
            d.create_dataset('filename', data=f'realbkgsim_{i:02d}')
            an = d.create_group('analysis'); an.attrs['NX_class'] = 'NXparameters'
            g = an.create_group('frame00000'); g.attrs['NX_class'] = 'NXparameters'
            peaks = boxes_to_peaks(fr['boxes'], fr['is_ring'], qmax, fr['amp'], fr['s_q'], fr['s_c'])
            g.create_dataset('fitted_peaks', data=peaks)
            g.create_dataset('detected_peaks', data=peaks)

            p = e.create_group('polar')          # lossless: what the simulator actually built
            p.create_dataset('image', data=fr['pol'], compression='gzip', compression_opts=4)
            p.create_dataset('mask', data=fr['mask'], compression='gzip', compression_opts=4)
            p.create_dataset('boxes', data=fr['boxes'].astype(np.float32))
            p.create_dataset('is_ring', data=fr['is_ring'])
            p.create_dataset('amplitude', data=fr['amp'])
            p.create_dataset('sigma_q', data=fr['s_q'])
            p.create_dataset('sigma_chi', data=fr['s_c'])
            p.attrs['shape'] = [HEIGHT, WIDTH]
            p.attrs['q_max'] = qmax
            p.attrs['note'] = ('boxes are xyxy in polar pixels; x = q/q_max*1024, '
                               'y = chi/90*512. FULL box extent = coef*sigma, coef=(2.80,1.30).')

            pr = e.create_group('process'); pr.attrs['NX_class'] = 'NXprocess'
            pr.create_dataset('program', data='diagnostics/dump_realbkg_pygid.py')
            pr.create_dataset('date', data=datetime.datetime.now().isoformat())
            pr.create_dataset('NOTE', data=(
                'SIMULATED, RAW (pre-contrast): linear donor background + peaks + counting noise. '
                'No clip/log/HE has been applied. img_gid_q is an inverse-polar resampling and '
                'loses the high-q wedge near chi=0/90; entry/polar/image is the lossless original.'))
            pr.create_dataset('settings', data=json.dumps(dict(
                config=args.config, seed=args.seed, n=n, q_max=qmax, eta=fr['eta'],
                a_coef=sc.a_coef, w_coef=sc.w_coef,
                donors=cfg.realbkg_donor_path, stats=cfg.realbkg_stats_path,
                bank=cfg.physics_bank_path)))

    nb = np.array([len(x['boxes']) for x in frames])
    nr = np.array([int(x['is_ring'].sum()) for x in frames])
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"  {len(frames)} frames | boxes/frame min {nb.min()} p50 {int(np.median(nb))} max {nb.max()}"
          f" | rings/frame min {nr.min()} p50 {int(np.median(nr))} max {nr.max()}")
    print(f"  entry_simNN/polar/image  = raw polar frame (lossless, pre-contrast)")
    print(f"  entry_simNN/data/img_gid_q = raw reciprocal frame (lossy resampling)")
    print(f"  entry_simNN/data/analysis/frame00000/fitted_peaks = GT boxes, pygid PEAK_DTYPE")


if __name__ == '__main__':
    main()
