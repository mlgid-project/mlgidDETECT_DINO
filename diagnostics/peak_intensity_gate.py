"""Phase 0 gate, intensity axis: does pygidsim give a realistic RELATIVE intensity distribution?

This is the axis the physics track is actually for. The current simulator draws peak intensities
UNIFORMLY in a bounded range (gen_intensities, simulation.py:1161: rand()*(hi-lo)+lo, with
ring_intensity_range=(2,50) and seg_intensity_range=(10,50)), boosts low-q/narrow peaks by 2x, and
rescales linearly. Real diffraction is nothing like uniform: a few strong reflections and a long
weak tail spanning orders of magnitude. pygidsim gives structure-factor intensities, which should
have the right shape by construction.

Everything is compared SCALE-FREE, as intensity normalized by the brightest peak of its own
pattern (I / I_max per frame / per bank entry), because the three sources carry different units:
  REAL 41       roi_data 'peak height' -- fitted image amplitude, directly available
  REAL organic  fitted_peaks has amplitude == 0 everywhere, so intensity is MEASURED from
                data/img_gid_q at each labeled peak: patch max minus a local background ring
  BANK          structure-factor intensity from pygidsim
  SIM current   uniform draws reproduced from gen_intensities

Reported statistics, all scale-free:
  med I/Imax     median normalized intensity -- uniform gives ~0.5, real diffraction much lower
  frac<0.1       fraction of peaks below a tenth of the pattern maximum (the weak tail)
  log10 range    log10(p95/p05) within a pattern -- the dynamic range the detector must span
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


def _norm(groups):
    """[array of intensities per pattern] -> pooled I/Imax, and per-pattern log10 dynamic range."""
    pooled, ranges = [], []
    for g in groups:
        g = np.asarray(g, float)
        g = g[np.isfinite(g) & (g > 0)]
        if len(g) < 5:
            continue
        pooled.append(g / g.max())
        p5, p95 = np.percentile(g, [5, 95])
        if p5 > 0:
            ranges.append(np.log10(p95 / p5))
    if not pooled:
        return np.array([]), np.array([])
    return np.concatenate(pooled), np.asarray(ranges)


def real_41():
    groups = []
    with h5py.File(H5_41, 'r') as f:
        def visit(name, obj):
            if name.endswith('roi_data') and 'peak height' in obj:
                groups.append(obj['peak height'][()].astype(float))
        f.visititems(visit)
    return _norm(groups)


def real_organic(half=6, bg_lo=10, bg_hi=16):
    """Measure each labeled peak's amplitude from the reciprocal-space image.

    fitted_peaks stores radius (A^-1) and angle (deg from the q_xy axis) but a zeroed amplitude,
    so the intensity has to come from the image: peak = max over a small patch, background = median
    over a surrounding annulus, amplitude = peak - background.
    """
    groups = []
    with h5py.File(ORGANIC_H5, 'r') as f:
        for key in f:
            d = f[key]['data']
            q_xy, q_z = d['q_xy'][()], d['q_z'][()]
            imgs = d['img_gid_q']
            for i, fr in enumerate(sorted(d['analysis'])):
                p = d['analysis'][fr]['fitted_peaks'][()]
                p = p[p['visibility'] > 0]
                if not len(p):
                    continue
                img = np.asarray(imgs[i], float)      # axis 0 = q_z, axis 1 = q_xy
                nz, nxy = img.shape
                rad = p['radius'].astype(float)
                ang = np.radians(p['angle'].astype(float))
                ixy = np.searchsorted(q_xy, rad * np.cos(ang)).clip(0, nxy - 1)
                iz = np.searchsorted(q_z, rad * np.sin(ang)).clip(0, nz - 1)
                amps = []
                for y, x in zip(iz, ixy):
                    y0, y1 = max(0, y - bg_hi), min(nz, y + bg_hi + 1)
                    x0, x1 = max(0, x - bg_hi), min(nxy, x + bg_hi + 1)
                    win = img[y0:y1, x0:x1]
                    if win.size < 25:
                        continue
                    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
                    r = np.hypot(yy, xx)
                    core, ring = win[r <= half], win[(r >= bg_lo) & (r <= bg_hi)]
                    if not core.size or not ring.size:
                        continue
                    amps.append(float(core.max() - np.median(ring)))
                if amps:
                    groups.append(np.asarray(amps))
    return _norm(groups)


def bank(path):
    d = np.load(path, allow_pickle=True)
    inten, start, count = d['intensity'].astype(float), d['entry_start'], d['entry_count']
    return _norm([inten[s:s + c] for s, c in zip(start, count)])


def sim_current(n_patterns=4000):
    """Reproduce gen_intensities' default path (alpha=None, i.e. raw_intensity off)."""
    groups = []
    for _ in range(n_patterns):
        # obj_num=(2,200), drawn toward the upper end (simulation.py:926)
        n = int(np.clip(RNG.normal(2 + 0.75 * 198, 198), 2, 200))
        lo, hi = (2.0, 50.0) if RNG.random() < 0.5 else (10.0, 50.0)
        v = RNG.uniform(lo, hi, n)
        v[RNG.random(n) < 0.15] *= 2.0        # the low-q / narrow-peak boost
        v = (v - v.min()) / (v.max() - v.min()) * (hi - lo) + lo
        groups.append(v)
    return _norm(groups)


def main():
    rows = [('REAL organic (measured)', real_organic()), ('REAL 41 (peak height)', real_41()),
            ('SIM current (uniform)', sim_current())]
    rows += [(k, bank(v)) for k, v in BANKS.items()]

    print(f'{"source":<26} {"peaks":>8} {"med I/Imax":>11} {"frac<0.1":>9} '
          f'{"log10 range":>12}')
    print('-' * 70)
    for name, (pooled, ranges) in rows:
        if not len(pooled):
            print(f'{name:<26} (no data)')
            continue
        print(f'{name:<26} {len(pooled):>8,} {np.median(pooled):>11.3f} '
              f'{float((pooled < 0.1).mean()):>9.3f} {np.median(ranges):>12.2f}')
    print('\nA uniform draw gives med I/Imax ~0.5, frac<0.1 ~0.1 and a narrow log10 range.')
    print('Real diffraction should sit far below on the first two and far above on the last.')


if __name__ == '__main__':
    main()
