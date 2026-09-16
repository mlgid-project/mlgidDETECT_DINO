"""Write N pygidSIM frames as a REAL pyGID/NeXus file: reciprocal-space `img_gid_q` + `fitted_peaks`.

`diagnostics/dump_sim_nexus.py` dumps what the simulator natively produces -- POLAR images -- which
cannot be loaded by `PyGIDDataset` because that path polar-converts `img_gid_q` itself. This script
closes that gap by inverting the polar transform, so the output file has the same layout as
`organic_labeled.h5` and routes through `detect_dataset_type` -> 'pygid'.

GEOMETRY. `_get_polar_grid` (util/exp_preprocess.py:221) with DEFAULT_BEAM_CENTER = (0, 0) maps a
reciprocal pixel (y, z) to polar (phi, r) with phi = atan2(z, y) in [0, pi/2] over 512 rows and
r = hypot(y, z) in [0, r_max] over 1024 columns, r_max = hypot(N-1, N-1). Both are linear, so the
inverse is exact and this script simply evaluates it and resamples with cv2.remap.

LOSSY BY CONSTRUCTION, and this is unavoidable: a rectangular polar image covers r up to r_max at
EVERY angle, but the reciprocal square only reaches r = (N-1)/max(|cos phi|, |sin phi|). About 18%
of the polar rectangle -- the high-q wedge near chi = 0 and chi = 90 -- has no home in the square
and is lost. Real detector frames have exactly the same property (their polar corners are invalid),
so this is a fidelity cost of the format, not a bug. The script MEASURES the round-trip error
inside the valid mask and prints it; read that number before trusting the file.

LABELS round-trip exactly. `_load_fittedpeaks` (util/pygidloader.py:145) converts
    radius_pixel = radius / q_max * 1024,  angle_pixel = angle * 512 / 90
with q_max = sqrt(2) * N / GEO_PIXELPERANGSTROEM (util/exp_preprocess.py:178, PPA = 500), so the
simulator's polar boxes are written back through the inverse of exactly that formula.

  python diagnostics/dump_sim_pygid.py [--frames 20] [--n 1641] [--config ...] [--out PATH]
"""
import os, sys, json, argparse, datetime
import numpy as np
import cv2
import torch
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

HEIGHT, WIDTH = 512, 1024          # polar shape the whole pipeline uses
PPA = 500                          # Config.GEO_PIXELPERANGSTROEM

#the dtype organic_labeled.h5 uses for data/analysis/frameNNNNN/fitted_peaks
PEAK_DTYPE = np.dtype([
    ('amplitude', '<f4'), ('angle', '<f4'), ('angle_width', '<f4'), ('radius', '<f4'),
    ('radius_width', '<f4'), ('q_xy', '<f4'), ('q_z', '<f4'), ('theta', '<f4'),
    ('score', '<f4'), ('A', '<f4'), ('B', '<f4'), ('C', '<i4'), ('is_cut_qz', '?'),
    ('is_cut_qxy', '?'), ('is_ring', '?'), ('visibility', '<i4'), ('id', '<i4')])


def polar_to_reciprocal(polar, n):
    """Exact inverse of _get_polar_grid, resampled onto an n x n reciprocal grid."""
    iy, iz = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
    r = np.hypot(iy, iz)
    phi = np.arctan2(iz, iy)
    r_max = float(np.hypot(n - 1, n - 1))
    col = (r / r_max * (WIDTH - 1)).astype(np.float32)
    row = (phi / (np.pi / 2) * (HEIGHT - 1)).astype(np.float32)
    return cv2.remap(polar.astype(np.float32), col, row, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def boxes_to_peaks(boxes, is_ring, q_max, amp=None):
    """Polar pixel boxes -> a fitted_peaks record that _load_fittedpeaks inverts exactly."""
    p = np.zeros(len(boxes), dtype=PEAK_DTYPE)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    p['radius'] = (x0 + x1) / 2 / WIDTH * q_max
    p['radius_width'] = (x1 - x0) / WIDTH * q_max
    p['angle'] = (y0 + y1) / 2 * 90.0 / HEIGHT
    p['angle_width'] = (y1 - y0) * 90.0 / HEIGHT
    p['q_xy'] = p['radius'] * np.cos(np.radians(p['angle']))
    p['q_z'] = p['radius'] * np.sin(np.radians(p['angle']))
    p['theta'] = p['angle']
    p['is_ring'] = is_ring
    p['visibility'] = 3                       # simulated peaks are ground truth -> confidence 1.0
    p['score'] = 1.0
    p['amplitude'] = 1.0 if amp is None else amp
    p['id'] = np.arange(len(boxes))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--n', type=int, default=1641, help='reciprocal grid size (real files: 1641)')
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_physics6.py')
    ap.add_argument('--stage', default='peaks', choices=['peaks', 'raw', 'final'],
                    help="peaks = img_from_labels only, the ONLY stage carrying physical peak "
                         "intensities; raw/final are min-max normalised by add_glass and "
                         "add_linear_background and carry no intensity scale at all")
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/datasets/pygidsim_raw_20_pygid.h5')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    import physics_simulation as PS
    from physics_simulation import PhysicsSimulation

    cfg = SLConfig.fromfile(os.path.join(_REPO, args.config))
    sc = SimulationConfig()
    coefs = getattr(cfg, 'box_coef_override', None) or (2.80, 1.30)
    sc.a_coef, sc.w_coef = float(coefs[0]), float(coefs[1])
    sim = PhysicsSimulation(
        cfg.physics_bank_path, sim_config=sc, device='cuda',
        unify_contrast=bool(getattr(cfg, 'unify_contrast', False)),
        n_powder=getattr(cfg, 'physics_n_powder', None),
        real_tail_only=bool(getattr(cfg, 'real_tail_only', False)),
        frame_types=getattr(cfg, 'physics_frame_types', None))

    snap = {}
    _mp, _sp = PS.mul_perlin, PS.apply_salt_pepper_noise
    PS.mul_perlin = lambda img, *a, **k: (snap.__setitem__('peaks', img.detach().clone()),
                                          _mp(img, *a, **k))[1]

    def _sp_hook(img, *a, **k):
        out = _sp(img, *a, **k); snap['raw'] = out.detach().clone(); return out
    PS.apply_salt_pepper_noise = _sp_hook

    # capture the RENDERED per-peak amplitudes. They align 1:1 with the returned boxes: after
    # img_from_labels the only thing touching boxes is flip_image, which flips coordinates but
    # drops nothing.
    from simulation import FastSimulation
    _ifl = FastSimulation.img_from_labels

    def _ifl_hook(self, boxes, intensities, is_ring):
        snap['amp'] = intensities.detach().clone()
        return _ifl(self, boxes, intensities, is_ring)
    FastSimulation.img_from_labels = _ifl_hook

    n = args.n
    # PyGIDDataset (util/pygidloader.py:187-193) OVERRIDES the geometry from the file's own axes:
    #     GEO_QMAX = hypot(q_z[-1], q_xy[-1]);  GEO_PIXELPERANGSTROEM = img.shape[0] / q_z[-1]
    # so q_max is fixed by the LAST axis value, (n-1)/PPA, not by n/PPA. Deriving it from the
    # axis array is the only way these cannot disagree -- assuming sqrt(2)*n/PPA puts every box
    # 0.6 px off.
    q_axis = np.arange(n, dtype=np.float64) / PPA
    q_max = float(np.hypot(q_axis[-1], q_axis[-1]))
    print(f'reciprocal {n}x{n}, q_max = {q_max:.4f} 1/A (PPA {PPA}), stage = {args.stage}\n')

    frames = []
    for i in range(args.frames):
        snap.clear()
        img, boxes, mask, is_ring = sim.simulate_img()
        pick = {'peaks': snap['peaks'], 'raw': snap['raw'], 'final': img}[args.stage]
        pol = pick.float().cpu().numpy()
        pol = pol[0] if pol.ndim == 3 else pol
        m = mask.cpu().numpy(); m = (m[0] if m.ndim == 3 else m).astype(bool)
        rec = polar_to_reciprocal(pol, n)
        # round-trip fidelity, inside the valid mask, vs the simulator's own polar image
        from util.exp_preprocess import _get_polar_grid
        class _C:
            PREPROCESSING_CUDA = False
        yy, zz = _get_polar_grid(_C(), (n, n), (HEIGHT, WIDTH), (0, 0))
        back = cv2.remap(rec, yy.astype(np.float32), zz.astype(np.float32), cv2.INTER_CUBIC)
        cover = (back != 0) | (pol == 0)
        rng = max(pol[m].max() - pol[m].min(), 1e-9)
        err = np.abs(back - pol)[m & cover].mean() / rng
        lost = 1.0 - (cover & m).sum() / max(m.sum(), 1)
        amp = snap['amp'].float().cpu().numpy()
        assert len(amp) == len(boxes), (len(amp), len(boxes))
        frames.append((rec, boxes.cpu().numpy(), is_ring.cpu().numpy(), amp))
        print(f'  frame {i:2d}: {len(boxes):4d} boxes   round-trip |err| {err:.2e} of range   '
              f'polar area with no reciprocal home: {lost:6.2%}', flush=True)

    PS.mul_perlin, PS.apply_salt_pepper_noise = _mp, _sp
    FastSimulation.img_from_labels = _ifl

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, 'w') as f:
        for i, (rec, boxes, is_ring, amp) in enumerate(frames):
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
            d.create_dataset('filename', data=f'pygidsim_{i:02d}')
            an = d.create_group('analysis'); an.attrs['NX_class'] = 'NXparameters'
            fr = an.create_group('frame00000'); fr.attrs['NX_class'] = 'NXparameters'
            peaks = boxes_to_peaks(boxes, is_ring, q_max, amp)
            fr.create_dataset('fitted_peaks', data=peaks)
            fr.create_dataset('detected_peaks', data=peaks)
            # `amplitude` is the RENDERED peak height: within one bank entry it is the true
            # structure-factor ratio (see PhysicsSimulation._scale -- linear, no gamma), times a
            # per-entry U(0.08, 1.0) minor/major-phase factor. There is no absolute scale
            # anywhere: _scale divides each entry by its own max.
            p = e.create_group('process'); p.attrs['NX_class'] = 'NXprocess'
            p.create_dataset('program', data='diagnostics/dump_sim_pygid.py')
            p.create_dataset('date', data=datetime.datetime.now().isoformat())
            p.create_dataset('NOTE', data=(
                'SIMULATED. img_gid_q is an inverse-polar resampling of a pygidSIM polar frame, '
                'not a measured detector image; the high-q wedge near chi=0/90 has no reciprocal '
                'home and is zero. visibility is 3 for every peak (ground truth).'))
            p.create_dataset('settings', data=json.dumps(dict(
                config=args.config, stage=args.stage, seed=args.seed, n=n, q_max=q_max,
                ppa=PPA, bank=cfg.physics_bank_path,
                unify_contrast=bool(getattr(cfg, 'unify_contrast', False)),
                real_tail_only=bool(getattr(cfg, 'real_tail_only', False)),
                frame_types=str(getattr(cfg, 'physics_frame_types', None)),
                a_coef=sc.a_coef, w_coef=sc.w_coef)))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
