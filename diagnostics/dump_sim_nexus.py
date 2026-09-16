"""Write N pygidSIM frames to a NeXus-style .h5 at three stages, BEFORE any contrast processing.

Stages, named by where they sit in `PhysicsSimulation._attempt` (physics_simulation.py:313-359):

  img_peaks   line 313 only -- `img_from_labels`, the rendered Gaussian peaks and NOTHING else.
              No perlin, no glass, no linear background, no Poisson, no dark area, no detector
              gaps, no salt-and-pepper. A noiseless render, not a detector frame.
  img_raw     through line 328 -- the full simulated DETECTOR frame: peaks + perlin + glass +
              linear background + Poisson + stretch + dark area + detector gaps + salt-and-pepper,
              but STOPPED BEFORE the contrast chain. This is the closest thing the simulator has
              to a raw measured frame, and is what "raw image, before any preprocessing" means.
  img_final   what training actually consumes: img_raw plus the contrast chain (unify_contrast /
              real_tail_only, or the legacy log->HE->clip path).

The three are snapshots of ONE call to `simulate_img()`, taken by wrapping `mul_perlin` and
`apply_salt_pepper_noise` in the physics_simulation namespace, so the chain is the production one
and cannot drift from this script.

!! COORDINATES. The simulator renders directly in POLAR (chi x q, HEIGHT x WIDTH = 512 x 1024).
It never builds a reciprocal-space map, so there is NO honest `img_gid_q` to write and the images
are stored as `img_polar` with `chi` / `q_index` axes. A real pyGID file stores reciprocal-space
frames at <group>/data/img_gid_q, which `standard_preprocessing` then polar-converts -- feeding
this file through that path would polar-convert an already-polar image. Read it directly.

  python diagnostics/dump_sim_nexus.py [--frames 20] [--config config/DINO/DINO_4scale_swin_physics6.py] [--out PATH]
"""
import os, sys, json, argparse, datetime
import numpy as np
import torch
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_physics6.py')
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/datasets/pygidsim_raw_20.h5')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(os.path.join(_REPO, args.config))
    from simulation import SimulationConfig
    import physics_simulation as PS
    from physics_simulation import PhysicsSimulation

    sc = SimulationConfig()
    coefs = getattr(cfg, 'box_coef_override', None) or (2.80, 1.30)
    sc.a_coef, sc.w_coef = float(coefs[0]), float(coefs[1])

    sim = PhysicsSimulation(
        cfg.physics_bank_path, sim_config=sc, device='cuda',
        unify_contrast=bool(getattr(cfg, 'unify_contrast', False)),
        n_powder=getattr(cfg, 'physics_n_powder', None),
        real_tail_only=bool(getattr(cfg, 'real_tail_only', False)),
        frame_types=getattr(cfg, 'physics_frame_types', None))

    # --- snapshot the production chain rather than re-implementing it ---
    snap = {}
    _mul_perlin, _salt = PS.mul_perlin, PS.apply_salt_pepper_noise

    def mul_perlin(img, *a, **k):
        snap['peaks'] = img.detach().clone()      # last write before a successful return wins
        return _mul_perlin(img, *a, **k)

    def apply_salt_pepper_noise(img, *a, **k):
        out = _salt(img, *a, **k)
        snap['raw'] = out.detach().clone()
        return out

    PS.mul_perlin, PS.apply_salt_pepper_noise = mul_perlin, apply_salt_pepper_noise

    peaks, raws, finals, masks = [], [], [], []
    boxes_all, isring_all, starts, counts = [], [], [], 0
    for i in range(args.frames):
        snap.clear()
        img, boxes, mask, is_ring = sim.simulate_img()
        assert 'peaks' in snap and 'raw' in snap, 'chain hooks did not fire -- check line numbers'
        peaks.append(snap['peaks'].float().cpu().numpy())
        raws.append(snap['raw'].float().cpu().numpy())
        finals.append(img.float().cpu().numpy())
        masks.append(mask.cpu().numpy().astype(np.uint8))
        b = boxes.cpu().numpy().astype(np.float32)
        starts.append(counts); counts += len(b)
        boxes_all.append(b); isring_all.append(is_ring.cpu().numpy().astype(bool))
        print(f'  frame {i:2d}: {len(b):4d} boxes, '
              f'peaks[{peaks[-1].min():.3g},{peaks[-1].max():.3g}] '
              f'raw[{raws[-1].min():.3g},{raws[-1].max():.3g}] '
              f'final[{finals[-1].min():.3g},{finals[-1].max():.3g}]', flush=True)

    PS.mul_perlin, PS.apply_salt_pepper_noise = _mul_perlin, _salt

    def stack(a):
        return np.stack([x[0] if x.ndim == 3 else x for x in a]).astype(np.float32)

    P, R, F = stack(peaks), stack(raws), stack(finals)
    M = np.stack([m[0] if m.ndim == 3 else m for m in masks]).astype(np.uint8)
    B = np.concatenate(boxes_all); IR = np.concatenate(isring_all)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, 'w') as f:
        e = f.create_group('entry_pygidsim'); e.attrs['NX_class'] = 'NXentry'
        e.attrs['definition'] = 'NXgid (polar, simulated)'
        d = e.create_group('data'); d.attrs['NX_class'] = 'NXdata'
        d.attrs['signal'] = 'img_raw'
        d.attrs['axes'] = np.array(['frame_num', 'chi', 'q_index'], dtype=object)
        d.attrs['WARNING'] = ('POLAR images (chi x q), NOT reciprocal space. There is no '
                              'img_gid_q because pygidSIM never builds one. Do not feed this '
                              'through standard_preprocessing -- it would polar-convert twice.')
        for name, arr, doc in (
                ('img_peaks', P, 'img_from_labels only: rendered Gaussian peaks, no background, '
                                 'no noise, no dark area, no gaps'),
                ('img_raw', R, 'full simulated detector frame BEFORE the contrast chain: peaks + '
                               'perlin + glass + linear background + Poisson + stretch + dark '
                               'area + detector gaps + salt-and-pepper'),
                ('img_final', F, 'what training consumes: img_raw + the contrast chain')):
            ds = d.create_dataset(name, data=arr, compression='gzip', compression_opts=4)
            ds.attrs['stage'] = doc
        d.create_dataset('valid_mask', data=M, compression='gzip', compression_opts=4)
        d.create_dataset('frame_num', data=np.arange(args.frames, dtype=np.int64))
        d.create_dataset('chi', data=np.arange(P.shape[1], dtype=np.float64))
        d.create_dataset('q_index', data=np.arange(P.shape[2], dtype=np.float64))

        g = e.create_group('labels'); g.attrs['NX_class'] = 'NXparameters'
        g.attrs['box_convention'] = (f'boxes are [q0, chi0, q1, chi1] in PIXELS; extent = coef * '
                                     f'sigma with a_coef={sc.a_coef} (chi), w_coef={sc.w_coef} (q)')
        g.create_dataset('boxes', data=B)
        g.create_dataset('is_ring', data=IR)
        g.create_dataset('frame_start', data=np.asarray(starts, np.int64))
        g.create_dataset('frame_count', data=np.asarray([len(b) for b in boxes_all], np.int64))

        p = e.create_group('process'); p.attrs['NX_class'] = 'NXprocess'
        p.create_dataset('program', data='diagnostics/dump_sim_nexus.py')
        p.create_dataset('date', data=datetime.datetime.now().isoformat())
        p.create_dataset('config', data=args.config)
        p.create_dataset('seed', data=args.seed)
        p.create_dataset('settings', data=json.dumps(dict(
            bank=cfg.physics_bank_path, unify_contrast=bool(getattr(cfg, 'unify_contrast', False)),
            real_tail_only=bool(getattr(cfg, 'real_tail_only', False)),
            physics_n_powder=str(getattr(cfg, 'physics_n_powder', None)),
            frame_types=str(getattr(cfg, 'physics_frame_types', None)),
            a_coef=sc.a_coef, w_coef=sc.w_coef, shape=list(P.shape))))
    print(f'\nwrote {args.out}  ({args.frames} frames, {P.shape[1]}x{P.shape[2]} polar, '
          f'{len(B)} boxes total)')


if __name__ == '__main__':
    main()
