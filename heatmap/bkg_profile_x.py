"""Mean and local noise of the processed (HE) image per q-column bin: real 41 vs simulator pickles (dump_dets / sim_pickle format). CPU only.
  python heatmap/bkg_profile_x.py name=<pkl>:<set> [name=<pkl>:<set> ...] [--nbins 8]
Per bin (valid pixels only, ring/peak pixels not removed): mean image value, 'local noise' = sd of (pixel - 5x5 box mean), and the share of invalid pixels."""
import sys, pickle, argparse
import numpy as np
from scipy.ndimage import uniform_filter


def main():
    p = argparse.ArgumentParser(); p.add_argument('src', nargs='+'); p.add_argument('--nbins', type=int, default=8); p.add_argument('--max_frames', type=int, default=150)
    a = p.parse_args()
    cache = {}
    for s in a.src:
        name, rest = s.split('=', 1); path, ds = rest.rsplit(':', 1)
        if path not in cache:
            cache[path] = pickle.load(open(path, 'rb'))
        fr = cache[path]['sets'][ds][:a.max_frames]
        M, N, V = [], [], []
        edges = np.linspace(0, 1024, a.nbins + 1).astype(int)
        for f in fr:
            im = np.asarray(f['img'], np.float32); m = np.asarray(f['mask'], bool)
            loc = im - uniform_filter(im, 5)
            row_m, row_n, row_v = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = m[:, lo:hi]
                row_v.append(1.0 - k.mean())
                row_m.append(float(im[:, lo:hi][k].mean()) if k.sum() > 50 else np.nan)
                row_n.append(float(loc[:, lo:hi][k].std()) if k.sum() > 50 else np.nan)
            M.append(row_m); N.append(row_n); V.append(row_v)
        M, N, V = (np.array(x) for x in (M, N, V))
        print(f'\n== {name}: {len(fr)} frames; bins of {1024 // a.nbins} px along q')
        print('   bin start      ' + ' '.join(f'{e:6d}' for e in edges[:-1]))
        print('   mean value     ' + ' '.join(f'{np.nanmean(M[:, i]):6.3f}' for i in range(a.nbins)))
        print('   local noise sd ' + ' '.join(f'{np.nanmean(N[:, i]):6.3f}' for i in range(a.nbins)))
        print('   invalid share  ' + ' '.join(f'{V[:, i].mean():6.3f}' for i in range(a.nbins)))


if __name__ == '__main__':
    main()
