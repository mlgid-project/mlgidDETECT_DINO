"""Phase 0 gate: does a physics peak bank place peaks more realistically than the current sim?

Everything is compared in the NORMALIZED radial coordinate the detector actually sees,
x = q / q_max in [0, 1] (util/pygidloader.py:146, radius_pixel = radius/q_max * 1024), because
q_max differs per real file and is sampled per image at render time.

Sources
  REAL organic  organic_labeled.h5, fitted_peaks.radius / q_max, q_max from that entry's
                q_xy/q_z last bin (util/pygidloader.py:187)
  REAL 41       41.h5 roi_data.radius (px) / sqrt(sum(reciprocal_shape**2))
                (util/labeleddataset.py:115)
  SIM current   simulation.py simulate_labels: pos = torch_uniform(0.048*W, 0.98*W), i.e.
                UNIFORM in the radial coordinate -- no physics at all
  BANK ...      q / q_max with q_max ~ U(2.5, 4.5), the render-time sampling used by
                physics_simulation.py

The verdict question: is a bank's KS distance to real materially SMALLER than the uniform sim's?
If not, physical peak placement buys nothing, and the phase-P negative is explained by the
mechanism it already reported rather than by the choice of CIF library.
"""
import numpy as np
import h5py

ORGANIC_H5 = '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
H5_41 = '/mnt/lustre/work/schreiber/szb389/datasets/41.h5'
BANKS = {
    'bank perovskite': '/mnt/lustre/work/schreiber/szb389/datasets/cif_library/bank/bank.npz',
    'bank organic': '/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/'
                    'bank_gate.npz',
}
RNG = np.random.default_rng(0)


def ks(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float('nan')
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, 'right') / len(a)
                               - np.searchsorted(b, grid, 'right') / len(b))))


def real_organic():
    out = []
    with h5py.File(ORGANIC_H5, 'r') as f:
        for key in f:
            d = f[key]['data']
            q_max = float(np.hypot(d['q_z'][-1], d['q_xy'][-1]))
            for fr in d['analysis']:
                p = d['analysis'][fr]['fitted_peaks'][()]
                p = p[p['visibility'] > 0]
                if len(p):
                    out.append(p['radius'].astype(float) / q_max)
    return np.concatenate(out)


def real_41():
    out = []
    with h5py.File(H5_41, 'r') as f:
        def visit(name, obj):
            if not name.endswith('roi_data') or 'radius' not in obj:
                return
            grp = f[name.rsplit('/', 1)[0]]
            shape = grp['image'].shape if 'image' in grp else (1350, 1350)
            max_radius = float(np.sqrt(np.sum(np.asarray(shape, float) ** 2)))
            out.append(obj['radius'][()].astype(float) / max_radius)
        f.visititems(visit)
    return np.concatenate(out)


def sim_current(n):
    """simulation.py:933 -- pos is uniform over the radial axis."""
    return RNG.uniform(0.048, 0.98, n)


def bank_norm(path, n, q_max):
    """Bank peaks in normalized radial coordinate for a given detector q_max.

    q_max must be the REAL one for the eval set being compared against, not the
    physics_simulation.py render-time draw U(2.5, 4.5): that range is systematically below the
    organic set's actual 4.95, which alone would push bank peaks outward and fake a mismatch.
    """
    d = np.load(path, allow_pickle=True)
    q = d['q'].astype(float)
    if len(q) > n:
        q = RNG.choice(q, n, replace=False)
    x = q / q_max
    return x[(x > 0) & (x <= 1.0)]      # peaks past the detector edge are not rendered


def main():
    org, f41 = real_organic(), real_41()
    # q_max of each eval set, in its own units, so bank peaks are placed on the right detector
    q_max_org = 4.95                       # every entry in organic_labeled.h5
    q_max_41 = np.sqrt(2 * 1350 ** 2) / 500.0   # 41 roi: shape 1350^2, 500 px/A^-1
    print(f'REAL organic {len(org):>6,} peaks   median x {np.median(org):.3f}   '
          f'q_max {q_max_org:.2f}')
    print(f'REAL 41      {len(f41):>6,} peaks   median x {np.median(f41):.3f}   '
          f'q_max {q_max_41:.2f}\n')

    uni = sim_current(400_000)
    print(f'{"source":<24} {"med x org":>10} {"KS vs org":>10} {"med x 41":>9} '
          f'{"KS vs 41":>9} {"KS sum":>8}')
    print('-' * 76)
    k_o, k_4 = ks(uni, org), ks(uni, f41)
    print(f'{"SIM current (uniform)":<24} {np.median(uni):>10.3f} {k_o:>10.3f} '
          f'{np.median(uni):>9.3f} {k_4:>9.3f} {k_o + k_4:>8.3f}')
    for name, path in BANKS.items():
        xo = bank_norm(path, 400_000, q_max_org)
        x4 = bank_norm(path, 400_000, q_max_41)
        k_o, k_4 = ks(xo, org), ks(x4, f41)
        print(f'{name:<24} {np.median(xo):>10.3f} {k_o:>10.3f} {np.median(x4):>9.3f} '
              f'{k_4:>9.3f} {k_o + k_4:>8.3f}')

    n = min(len(org), len(f41))
    print(f'\nKS noise floor at these sample sizes ~ {1.36 / np.sqrt(n):.3f} '
          f'(p=0.05, n={n}); differences below that are not meaningful.')
    print('NOTE: both banks were generated with q_xy_max = q_z_max = 3.0, so they hold no peak '
          f'beyond |q| = 4.24 -- i.e. nothing past x = {4.24 / q_max_org:.2f} on the organic '
          'detector.')


if __name__ == '__main__':
    main()
