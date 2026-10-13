"""Brightness next to invalid pixels: mean processed-image value (and local noise) as a function of the distance to the nearest invalid pixel, real 41 vs simulator pickles. CPU only.
  python heatmap/edge_profile.py name=<pkl>:<set> [...] [--max_frames 100]
Distances in px (0 = valid pixel touching an invalid one)."""
import sys, pickle, argparse
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

EDGES = [0, 1, 2, 3, 5, 8, 12, 20, 40, 1e9]


def main():
    p = argparse.ArgumentParser(); p.add_argument('src', nargs='+'); p.add_argument('--max_frames', type=int, default=100)
    a = p.parse_args()
    cache = {}
    for s in a.src:
        name, rest = s.split('=', 1); path, ds = rest.rsplit(':', 1)
        if path not in cache:
            cache[path] = pickle.load(open(path, 'rb'))
        fr = cache[path]['sets'][ds][:a.max_frames]
        M = [[] for _ in EDGES[:-1]]; N = [[] for _ in EDGES[:-1]]
        for f in fr:
            im = np.asarray(f['img'], np.float32); m = np.asarray(f['mask'], bool)
            if m.all() or not m.any():
                continue
            d = distance_transform_edt(m) - 1.0
            loc = np.abs(im - uniform_filter(im, 5))
            for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
                k = m & (d >= lo) & (d < hi)
                if k.sum() > 20:
                    M[i].append(float(im[k].mean())); N[i].append(float(loc[k].mean()))
        print(f'\n== {name}: {len(fr)} frames')
        print('   distance px    ' + ' '.join(f'{lo:>6g}' for lo in EDGES[:-1]))
        print('   mean value     ' + ' '.join(f'{np.mean(x):6.3f}' if x else '   n/a ' for x in M))
        print('   mean |local|   ' + ' '.join(f'{np.mean(x):6.3f}' if x else '   n/a ' for x in N))


if __name__ == '__main__':
    main()
