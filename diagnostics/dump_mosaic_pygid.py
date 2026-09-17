"""Dump simulator frames on MOSAIC backgrounds as a pyGID/NeXus file, RAW (pre-contrast).

Same layout as the ground-truth set (`organic_labeled.h5`): `data/img_gid_q` plus
`data/analysis/frame00000/fitted_peaks` in the standard PEAK_DTYPE, so the file opens in the same
tooling and the GT boxes draw. `entry/polar/*` additionally carries the LOSSLESS polar frame the
simulator actually built, before any contrast step.

BACKGROUNDS come from `realbkg_sim.mosaic_background`, i.e. tiles of the 90 reviewed bare-silicon
Lambda frames that never contained diffraction, reassembled per frame. This replaces the old donor
bank, where every frame carried real peaks that suppression had failed to remove and that
therefore trained the detector to call peaks background.

BACKGROUND DIVERSITY IS FORCED HERE, HARDER THAN IN TRAINING. The donors are sorted by count rate
and cut into as many DISJOINT groups as there are frames, and each frame's background is
mosaicked from its own group only. No two frames in this file share a single source pixel, so
anything that looks alike across frames is the simulator, not a repeated donor. Training does the
opposite on purpose -- it wants the widest tile pool it can get -- so this file is a strict
worst-case view of how varied the backgrounds can be, not the typical one.

PEAK DIVERSITY is stratified rather than left to chance: the frames cycle through sparse, medium
and crowded spot counts, with and without powder rings, so a reviewer sees the range the simulator
can produce instead of 20 samples from the middle of the distribution.

  python diagnostics/dump_mosaic_pygid.py [--frames 24] [--out PATH]
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

# (n_oriented, p_ring, label) -- cycled over the requested frames
STRATA = [((1, 1), 0.0, 'sparse spots'),
          ((2, 2), 0.0, 'medium spots'),
          ((3, 3), 0.0, 'crowded spots'),
          ((1, 1), 1.0, 'rings + few spots'),
          ((2, 2), 1.0, 'rings + medium spots'),
          ((3, 3), 1.0, 'rings + crowded spots')]


def build_pool(n, seed, disjoint=True, donors=None):
    """Pre-generate n mosaic backgrounds with the per-donor quantities simulate_img expects.

    With `disjoint`, background i is cut only from donor group i of an exposure-sorted partition,
    so the n backgrounds have no source frame in common.
    """
    from realbkg_sim.mosaic_background import MosaicBackground
    from realbkg_simulation import RealBkgSimulation as RB
    mb = MosaicBackground(seed=seed, **({'accepted': donors} if donors else {}))
    groups = mb.exposure_partition(n) if disjoint else None
    info = []
    qsrc = None
    mm = os.path.join(os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389'),
                      'datasets/realbkg_donors_mm/qmax.npy')
    if os.path.exists(mm):
        qsrc = np.load(mm)
    bkg, msk, noi, cof, qmx = [], [], [], [], []
    for i in range(n):
        g = groups[i % len(groups)] if groups else None
        b, m = mb.background(pool=g)
        nz = RB._noise_map(b.astype(np.float64), m)
        bkg.append(b.astype(np.float32)); msk.append(m)
        noi.append(nz.astype(np.float32))
        cof.append((nz/np.sqrt(np.maximum(cv2.GaussianBlur(b, (0, 0), 16.0), 1e-6))).astype(np.float32))
        # q_max is not defined by a mosaic; take a real polar frame's so peak positions map the
        # same way they do in the evaluation data
        qmx.append(float(qsrc[i % len(qsrc)]) if qsrc is not None and len(qsrc) else 4.45)
        if groups is not None:
            info.append(dict(donor_ids=[int(mb.rows[j].get('id', j)) for j in g],
                             donor_samples=sorted({mb.rows[j].get('sample', '?') for j in g}),
                             level_counts=float(np.median(mb.med[g]))))
            print(f'  mosaic {i+1}/{n} from {len(g)} donors @ '
                  f'{info[-1]["level_counts"]:.0f} cts  ids {info[-1]["donor_ids"]}', flush=True)
        else:
            info.append(dict(donor_ids=[], donor_samples=[], level_counts=float(np.median(b[m]))))
            print(f'  mosaic {i+1}/{n}', flush=True)
    return ((np.stack(bkg), np.stack(msk), np.stack(noi), np.stack(cof),
             np.asarray(qmx, np.float32)), info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--pool', type=int, default=0, help='0 = one background per frame')
    ap.add_argument('--shared-pool', action='store_true',
                    help='let frames share backgrounds, as training does')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/datasets/mosaicsim_raw.h5')
    ap.add_argument('--donors', default=None, help='override the donor json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    import realbkg_simulation as RS
    from realbkg_simulation import RealBkgSimulation

    cfg = SLConfig.fromfile(os.path.join(_REPO, a.config))
    sc = SimulationConfig()
    sc.a_coef, sc.w_coef = (getattr(cfg, 'box_coef_override', None) or (2.80, 1.30))
    print(f'box convention: a_coef (chi) = {sc.a_coef}, w_coef (q) = {sc.w_coef}')

    npool = a.pool or a.frames
    print(f'building {npool} mosaic backgrounds '
          f'({"shared" if a.shared_pool else "one disjoint donor group each"})...', flush=True)
    pool, pinfo = build_pool(npool, a.seed, disjoint=not a.shared_pool, donors=a.donors)

    # install the mosaic pool in place of the donor bank; simulate_img is untouched
    _ld = RealBkgSimulation._load_donors

    def install(self, *args, **kw):
        self.bkg, self.mask, self.noise, self.coef, self.qmax = pool
        self.meta = [dict(source='mosaic') for _ in range(len(pool[0]))]
        print(f'[mosaic] pool of {len(self.bkg)} backgrounds installed', flush=True)
        return True
    RealBkgSimulation._load_donors = install
    sim = RealBkgSimulation(bank_path=cfg.physics_bank_path, donor_path=cfg.realbkg_donor_path,
                            stats_path=cfg.realbkg_stats_path, sim_config=sc, device='cpu')
    RealBkgSimulation._load_donors = _ld

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
        for i, b in enumerate(boxes):
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            d = np.abs((cb[:, 0]+cb[:, 2])/2 - cx)
            if not is_ring[i]:
                d = d + np.abs((cb[:, 1]+cb[:, 3])/2 - cy)
            d[used] = np.inf
            j = int(np.argmin(d)); out[i] = j; used[j] = True
        return out

    frames = []
    while len(frames) < a.frames:
        n_or, p_ring, label = STRATA[len(frames) % len(STRATA)]
        sim.n_oriented, sim.p_ring = n_or, p_ring
        k = len(frames) % npool
        if not a.shared_pool:
            # hand the simulator a pool of exactly ONE background, so frame k is guaranteed to be
            # built on donor group k rather than on whatever the RNG happens to pick
            sim.bkg, sim.mask, sim.noise, sim.coef, sim.qmax = [x[k:k+1] for x in pool]
        snap.clear()
        r = sim.simulate_img()
        if r is None:
            continue
        _img, boxes, mask, is_ring = r
        boxes = boxes.cpu().numpy(); is_ring = is_ring.cpu().numpy()
        c = snap['cand']; j = match(boxes, is_ring, c)
        frames.append(dict(pol=snap['total'].astype(np.float32), mask=snap['mask'].astype(bool),
                           boxes=boxes, is_ring=is_ring, qmax=snap['qmax'], label=label,
                           bkg=pinfo[k] if not a.shared_pool else dict(donor_ids=[]),
                           amp=c['amp'][j].astype(np.float32), s_q=c['s_q'][j].astype(np.float32),
                           s_c=c['s_c'][j].astype(np.float32)))
        f = frames[-1]
        print(f"  frame {len(frames)-1:2d} [{label:22s}] {len(boxes):3d} boxes "
              f"({int(is_ring.sum()):2d} ring)  I range "
              f"[{f['pol'][f['mask']].min():.4g}, {f['pol'][f['mask']].max():.4g}]", flush=True)

    RS.apply_contrast = _ac
    RealBkgSimulation._render = _render
    RealBkgSimulation._draw_peaks = _draw

    import h5py
    n = a.n
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with h5py.File(a.out, 'w') as f:
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
            d.create_dataset('filename', data=f'mosaicsim_{i:02d}_{fr["label"].replace(" ", "_")}')
            an = d.create_group('analysis'); an.attrs['NX_class'] = 'NXparameters'
            g = an.create_group('frame00000'); g.attrs['NX_class'] = 'NXparameters'
            peaks = boxes_to_peaks(fr['boxes'], fr['is_ring'], qmax, fr['amp'], fr['s_q'], fr['s_c'])
            g.create_dataset('fitted_peaks', data=peaks)
            g.create_dataset('detected_peaks', data=peaks)
            p = e.create_group('polar')
            p.create_dataset('image', data=fr['pol'], compression='gzip', compression_opts=4)
            p.create_dataset('mask', data=fr['mask'], compression='gzip', compression_opts=4)
            p.create_dataset('boxes', data=fr['boxes'].astype(np.float32))
            p.create_dataset('is_ring', data=fr['is_ring'])
            p.create_dataset('amplitude', data=fr['amp'])
            p.create_dataset('sigma_q', data=fr['s_q'])
            p.create_dataset('sigma_chi', data=fr['s_c'])
            p.attrs['q_max'] = qmax
            p.attrs['stratum'] = fr['label']
            p.attrs['note'] = ('xyxy polar px; x = q/q_max*1024, y = chi/90*512; '
                               'FULL box extent = coef*sigma, coef=(2.80,1.30)')
            pr = e.create_group('process'); pr.attrs['NX_class'] = 'NXprocess'
            pr.create_dataset('program', data='diagnostics/dump_mosaic_pygid.py')
            pr.create_dataset('date', data=datetime.datetime.now().isoformat())
            pr.create_dataset('NOTE', data=(
                'SIMULATED, RAW (pre-contrast). Background is a mosaic of bare-silicon Lambda '
                'modules that never contained diffraction. img_gid_q is an inverse-polar '
                'resampling and loses the high-q wedge; entry/polar/image is lossless.'))
            pr.create_dataset('settings', data=json.dumps(dict(
                stratum=fr['label'], seed=a.seed, n=n, q_max=qmax,
                a_coef=sc.a_coef, w_coef=sc.w_coef, background=fr['bkg'])))

    nb = np.array([len(x['boxes']) for x in frames])
    nr = np.array([int(x['is_ring'].sum()) for x in frames])
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"  {len(frames)} frames | boxes/frame min {nb.min()} p50 {int(np.median(nb))} "
          f"max {nb.max()} | frames with rings {int((nr > 0).sum())}")
    print("  entry_simNN/polar/image                        raw polar frame (lossless)")
    print("  entry_simNN/data/img_gid_q                     raw reciprocal frame")
    print("  entry_simNN/data/analysis/frame00000/fitted_peaks   GT boxes, pygid PEAK_DTYPE")


if __name__ == '__main__':
    main()
