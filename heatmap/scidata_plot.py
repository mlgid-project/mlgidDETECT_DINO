"""Plot converted frames of the SciData GIWAXS benchmark with the expected CIF rings and the fitted peaks (mlgidgui env).
  python heatmap/scidata_plot.py --h5 <run.h5> --out <png> [--n 2] [--seed 0]   2 frames per family (PVK / SnO2-only / OPV)"""
import argparse, json, random
import h5py, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
p = argparse.ArgumentParser(); p.add_argument('--h5', required=True); p.add_argument('--out', required=True); p.add_argument('--n', type=int, default=2); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
idx = json.load(open(a.h5.replace('.h5', '_index.json')))
fam = lambda c: 'PVK' if 'pvk' in c.lower() else ('SnO2' if 'sno2' in c.lower() else 'OPV')
by = {}
for n, v in idx.items():
    by.setdefault(fam(v['cls']), []).append(n)
random.seed(a.seed); pick = [(f, n) for f in ('PVK', 'SnO2', 'OPV') for n in random.sample(by[f], a.n)]
PERO = [0.986, 1.395, 1.708, 1.973, 2.206, 2.417, 2.79]; SNO2 = [1.88, 2.38, 2.65]          # alpha-FAPbI3 (100)(110)(111)(200)(210)(211)(220); rutile SnO2 (110)(101)(200)
fig, axs = plt.subplots(2, len(pick) // 2, figsize=(5.2 * len(pick) // 2, 10)); axs = axs.T.ravel() if len(pick) > 2 else axs
t = np.linspace(0, np.pi / 2, 100)
with h5py.File(a.h5) as f:
    for ax, (fm, n) in zip(axs, pick):
        d = f[n]['data']; img = np.asarray(d['img_gid_q'])[0]; qxy = np.asarray(d['q_xy']); qz = np.asarray(d['q_z'])
        v = np.log10(np.where(img > 0, img, np.nan) + 1)
        ax.imshow(v, origin='lower', extent=[qxy.min(), qxy.max(), qz.min(), qz.max()], cmap='gray', vmin=np.nanpercentile(v, 10), vmax=np.nanpercentile(v, 99.8))
        for r in PERO: ax.plot(r * np.sin(t), r * np.cos(t), color='#1baf7a', lw=0.6, alpha=0.8)
        for r in SNO2: ax.plot(r * np.sin(t), r * np.cos(t), color='#2a78d6', lw=0.6, alpha=0.8, ls='--')
        fr = d['analysis']; fr = fr[list(fr.keys())[0]] if len(fr) else None
        if fr is not None and 'fitted_peaks' in fr:
            fp = fr['fitted_peaks'][()]; ax.scatter(fp['q_xy'], fp['q_z'], s=26, facecolors='none', edgecolors='#eb6834', lw=1.1)
        ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.9); ax.set_title(f'{fm}: {idx[n]["cls"]}  ai={idx[n]["ai"]}  {n}', fontsize=8)
fig.suptitle('green = alpha-FAPbI3 rings, blue dashed = SnO2 (rutile), orange = fitted peaks', fontsize=9); fig.tight_layout(); fig.savefig(a.out, dpi=62)
