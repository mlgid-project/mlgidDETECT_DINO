"""Append one extra frame carrying a deliberately very bright peak to the review HDF5.

WHY. The twenty review frames top out at 8,820 counts in the brightest pixel and 2,260 counts of
peak amplitude, because amplitude is drawn from the lognormal fitted to real labelled peaks
(amp/local-noise, mu 1.364, sigma 1.002) and multiplied by the background's own noise -- on a
500-count background that tail simply does not reach tens of thousands. A reviewer still wants to
see what a saturating-class Bragg peak looks like coming out of this pipeline, so this script
forces one.

HOW. The single brightest peak of the frame has its amplitude overridden to `--amp` counts; every
other peak, the background, the noise, the widths, the gate and the box convention are untouched,
so the frame is produced by the ordinary pipeline in every other respect. The override is recorded
in the frame's `process/settings` so nobody mistakes it for a natural draw. It is placed on the
highest-exposure background available, which is where such a peak is least unrealistic: the local
background and noise there are largest, so the implied contrast over noise is as small as it can
be made.

  python diagnostics/add_bright_frame.py [--amp 32000] [--h5 PATH]
"""
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

HEIGHT, WIDTH = 512, 1024
from diagnostics.dump_realbkg_pygid import boxes_to_peaks, polar_to_reciprocal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amp', type=float, default=32000.0, help='forced amplitude, counts')
    ap.add_argument('--h5', default='/mnt/lustre/work/schreiber/szb389/datasets/mosaicsim_raw.h5')
    ap.add_argument('--entry', default=None, help='group name; default = next free entry_simNN')
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--tries', type=int, default=12)
    ap.add_argument('--compact', action='store_true',
                    help='force the bright peak to be a compact spot rather than whatever width '
                         'the model happened to draw')
    ap.add_argument('--sigma-q', type=float, default=3.0, help='with --compact, radial sigma px')
    ap.add_argument('--sigma-chi', type=float, default=8.0,
                    help='with --compact, azimuthal sigma px (real spots: p50 22.5, min 1.5)')
    a = ap.parse_args()

    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    import realbkg_simulation as RS
    from realbkg_simulation import RealBkgSimulation
    from realbkg_sim.mosaic_background import MosaicBackground

    cfg = SLConfig.fromfile(os.path.join(_REPO, a.config))
    sc = SimulationConfig()
    sc.a_coef, sc.w_coef = (getattr(cfg, 'box_coef_override', None) or (2.80, 1.30))

    # brightest exposure class: the last group of the exposure-sorted partition
    mb = MosaicBackground(seed=a.seed)
    grp = mb.exposure_partition(20)[-1]
    print(f'background from donors {[int(mb.rows[j].get("id", j)) for j in grp]} '
          f'@ {np.median(mb.med[grp]):.0f} counts/px', flush=True)
    b, m = mb.background(pool=grp)
    nz = RealBkgSimulation._noise_map(b.astype(np.float64), m)
    cf = nz/np.sqrt(np.maximum(cv2.GaussianBlur(b, (0, 0), 16.0), 1e-6))
    qp = os.path.join(os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389'),
                      'datasets/realbkg_donors_mm/qmax.npy')
    qs = np.load(qp) if os.path.exists(qp) else np.array([4.45], np.float32)
    pool = (b[None].astype(np.float32), m[None], nz[None].astype(np.float32),
            cf[None].astype(np.float32), np.asarray([float(qs[0])], np.float32))

    _ld = RealBkgSimulation._load_donors

    def install(self, *args, **kw):
        self.bkg, self.mask, self.noise, self.coef, self.qmax = pool
        self.meta = [dict(source='mosaic')]
        return True
    RealBkgSimulation._load_donors = install
    sim = RealBkgSimulation(bank_path=cfg.physics_bank_path, donor_path=cfg.realbkg_donor_path,
                            stats_path=cfg.realbkg_stats_path, sim_config=sc, device='cpu',
                            n_oriented=(2, 2), p_ring=0.0)
    RealBkgSimulation._load_donors = _ld

    # ---- the overrides. `_assign_amplitudes` runs AFTER `_draw_widths`, and it is the only
    # place that knows which peak is the brightest, so the widths are fixed up from there: the
    # arrays `simulate_img` is holding are mutated in place, which is what makes the box and the
    # render agree with the amplitude.
    held = {}
    _dw = RealBkgSimulation._draw_widths

    def dw(self, n, is_ring):
        r = _dw(self, n, is_ring)
        held['w'] = r
        return r
    RealBkgSimulation._draw_widths = dw

    _aa = RealBkgSimulation._assign_amplitudes

    def boost(self, rel, noise_at, n_label):
        amp = _aa(self, rel, noise_at, n_label).copy()
        k = int(np.argmax(amp))
        amp[k] = float(a.amp)
        if a.compact and 'w' in held:
            s_q, s_c, el = held['w']
            s_q[k] = float(a.sigma_q)
            s_c[k] = float(a.sigma_chi)
            # elongation divides the amplitude afterwards to conserve flux; exempt this peak so
            # the forced amplitude is exactly what lands in the image
            if el.get('on') and el.get('f') is not None:
                el['f'][k] = 1.0
        held['k'] = k
        return amp
    RealBkgSimulation._assign_amplitudes = boost

    snap = {}
    _ac = RS.apply_contrast

    def hook(total, mask, chain):
        snap['total'] = np.asarray(total).copy(); snap['mask'] = np.asarray(mask).copy()
        out = np.asarray(_ac(total, mask, chain), dtype=np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    RS.apply_contrast = hook

    _render = RealBkgSimulation._render

    def rh(self, x, y, s_q, s_c, amp, eta):
        snap['cand'] = dict(x=np.asarray(x).copy(), y=np.asarray(y).copy(),
                            s_q=np.asarray(s_q).copy(), s_c=np.asarray(s_c).copy(),
                            amp=np.asarray(amp).copy())
        return _render(self, x, y, s_q, s_c, amp, eta)
    RealBkgSimulation._render = rh

    _draw = RealBkgSimulation._draw_peaks

    def dh(self, qmax):
        snap['qmax'] = float(qmax)
        return _draw(self, qmax)
    RealBkgSimulation._draw_peaks = dh

    def match(boxes, is_ring, c):
        hw, hh = sc.w_coef*c['s_q']/2.0, sc.a_coef*c['s_c']/2.0
        cb = np.stack([c['x']-hw, c['y']-hh, c['x']+hw, c['y']+hh], 1).astype(np.float32)
        out = np.full(len(boxes), -1, int); used = np.zeros(len(cb), bool)
        for i, bb in enumerate(boxes):
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            d = np.abs((cb[:, 0]+cb[:, 2])/2 - cx)
            if not is_ring[i]:
                d = d + np.abs((cb[:, 1]+cb[:, 3])/2 - cy)
            d[used] = np.inf
            j = int(np.argmin(d)); out[i] = j; used[j] = True
        return out

    fr = None
    for t in range(a.tries):
        snap.clear()
        r = sim.simulate_img()
        if r is None:
            continue
        _img, boxes, mask, is_ring = r
        boxes = boxes.cpu().numpy(); is_ring = is_ring.cpu().numpy()
        c = snap['cand']; j = match(boxes, is_ring, c)
        pol = snap['total'].astype(np.float32); msk = snap['mask'].astype(bool)
        imax = float(pol[msk].max())
        print(f'  try {t}: {len(boxes)} boxes, I max {imax:.0f}', flush=True)
        if imax >= 0.9*a.amp:
            fr = dict(pol=pol, mask=msk, boxes=boxes, is_ring=is_ring, qmax=snap['qmax'],
                      amp=c['amp'][j].astype(np.float32), s_q=c['s_q'][j].astype(np.float32),
                      s_c=c['s_c'][j].astype(np.float32))
            break
    RS.apply_contrast = _ac
    RealBkgSimulation._render = _render
    RealBkgSimulation._draw_peaks = _draw
    RealBkgSimulation._assign_amplitudes = _aa
    RealBkgSimulation._draw_widths = _dw
    if fr is None:
        raise SystemExit('no frame reached the requested brightness')

    pol, msk = fr['pol'], fr['mask']
    kb = int(np.argmax(fr['amp']))
    bb = fr['boxes'][kb]
    cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
    r0, r1 = max(int(cy)-40, 0), min(int(cy)+41, HEIGHT)
    c0, c1 = max(int(cx)-40, 0), min(int(cx)+41, WIDTH)
    ring = np.ones_like(pol, bool); ring[r0:r1, c0:c1] = False
    loc_bkg = float(np.median(pol[msk & ~ring][:0])) if False else float(
        np.median(pol[msk][:]))
    # local background and noise right around the bright peak, excluding the peak itself
    win = pol[max(int(cy)-60, 0):int(cy)+61, max(int(cx)-60, 0):int(cx)+61]
    wm = msk[max(int(cy)-60, 0):int(cy)+61, max(int(cx)-60, 0):int(cx)+61]
    lo = np.percentile(win[wm], 40)
    base = float(np.median(win[wm][win[wm] <= lo]))
    sm = cv2.GaussianBlur(win.astype(np.float32), (0, 0), 2.0)
    res = (win - sm)[wm]
    lnoise = float(1.4826*np.median(np.abs(res - np.median(res))))
    print(f"\nbright peak: amplitude {fr['amp'][kb]:.0f} counts at (q px {cx:.0f}, chi px {cy:.0f})")
    print(f"  sigma_q {fr['s_q'][kb]:.2f} px, sigma_chi {fr['s_c'][kb]:.1f} px"
          f"  ->  box {bb[2]-bb[0]:.1f} x {bb[3]-bb[1]:.1f} px")
    print(f"  frame I max        {pol[msk].max():.0f} counts")
    print(f"  local background   {base:.0f} counts")
    print(f"  local pixel noise  {lnoise:.0f} counts")
    print(f"  contrast           {fr['amp'][kb]/max(lnoise,1e-9):.0f} x local noise")
    print(f"  real labelled peaks reach 442 x noise at most (214 measured)")

    import h5py
    with h5py.File(a.h5, 'a') as f:
        name = a.entry or f'entry_sim{len([k for k in f if k.startswith("entry_sim")]):02d}'
        if name in f:
            del f[name]
        qmax = fr['qmax']
        q_axis = np.arange(a.n, dtype=np.float64)/(a.n-1)*(qmax/np.sqrt(2.0))
        rec = polar_to_reciprocal(pol, a.n)
        e = f.create_group(name); e.attrs['NX_class'] = 'NXentry'
        e.create_dataset('definition', data='NXgid')
        d = e.create_group('data'); d.attrs['NX_class'] = 'NXdata'
        d.attrs['signal'] = 'img_gid_q'
        d.attrs['axes'] = np.array(['frame_num', 'q_z', 'q_xy'], dtype=object)
        d.create_dataset('img_gid_q', data=rec[None].astype(np.float32),
                         compression='gzip', compression_opts=4)
        d.create_dataset('q_xy', data=q_axis)
        d.create_dataset('q_z', data=q_axis)
        d.create_dataset('frame_num', data=np.array([0], dtype=np.int64))
        d.create_dataset('filename', data=f'mosaicsim_{name[-2:]}_bright_peak')
        an = d.create_group('analysis'); an.attrs['NX_class'] = 'NXparameters'
        g = an.create_group('frame00000'); g.attrs['NX_class'] = 'NXparameters'
        peaks = boxes_to_peaks(fr['boxes'], fr['is_ring'], qmax, fr['amp'], fr['s_q'], fr['s_c'])
        g.create_dataset('fitted_peaks', data=peaks)
        g.create_dataset('detected_peaks', data=peaks)
        p = e.create_group('polar')
        p.create_dataset('image', data=pol, compression='gzip', compression_opts=4)
        p.create_dataset('mask', data=msk, compression='gzip', compression_opts=4)
        p.create_dataset('boxes', data=fr['boxes'].astype(np.float32))
        p.create_dataset('is_ring', data=fr['is_ring'])
        p.create_dataset('amplitude', data=fr['amp'])
        p.create_dataset('sigma_q', data=fr['s_q'])
        p.create_dataset('sigma_chi', data=fr['s_c'])
        p.attrs['q_max'] = qmax
        p.attrs['stratum'] = 'forced bright peak'
        p.attrs['note'] = ('xyxy polar px; x = q/q_max*1024, y = chi/90*512; '
                           'FULL box extent = coef*sigma, coef=(2.80,1.30)')
        pr = e.create_group('process'); pr.attrs['NX_class'] = 'NXprocess'
        pr.create_dataset('program', data='diagnostics/add_bright_frame.py')
        pr.create_dataset('date', data=datetime.datetime.now().isoformat())
        pr.create_dataset('NOTE', data=(
            'SIMULATED, RAW (pre-contrast). NOT A NATURAL DRAW: the brightest peak of this frame '
            'had its amplitude OVERRIDDEN to the value in settings["forced_amplitude"]. '
            'Everything else -- background, noise, widths, visibility gate, boxes -- is the '
            'ordinary pipeline.'))
        pr.create_dataset('settings', data=json.dumps(dict(
            stratum='forced bright peak', seed=a.seed, n=a.n, q_max=qmax,
            a_coef=sc.a_coef, w_coef=sc.w_coef, forced_amplitude=float(a.amp),
            forced_compact=bool(a.compact),
            forced_sigma_q=(float(a.sigma_q) if a.compact else None),
            forced_sigma_chi=(float(a.sigma_chi) if a.compact else None),
            forced_peak_index=kb, frame_I_max=float(pol[msk].max()),
            local_background=base, local_pixel_noise=lnoise,
            contrast_x_noise=float(fr['amp'][kb]/max(lnoise, 1e-9)),
            background=dict(donor_ids=[int(mb.rows[j].get('id', j)) for j in grp],
                            level_counts=float(np.median(mb.med[grp]))))))
        print(f'\nappended {name} to {a.h5}  ({os.path.getsize(a.h5)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
