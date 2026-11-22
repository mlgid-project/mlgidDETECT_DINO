"""What differs, in plain image statistics, between the real frames of each latent-map region and the simulated frames?  (CPU)

  python heatmap/sim_vs_region_stats.py --corpus_stats <stats_corpus.npz> --regions <latent_map/vith_sk2> --sim <embed_sim_*_stats.npz> [<...>] --out <dir> [--k 6]

Inputs: corpus_stats.py output (rows aligned with embed_vith_sk2.npz and regions_k<k>.csv), the sim npz written by embed_sim.py --stats (fields stats, names, feat).
For every feature of heatmap/frame_stats.py:
  - median and IQR per real region, for the labelled frames, and for each simulator                                  -> stats_table.md
  - effect = (median sim - median region) / robust sd of the whole corpus (IQR / 1.349): how many 'corpus spreads' the simulator is off  -> effect_<sim>.png (features x regions)
  - AUC (sim vs region frames): 0.5 = indistinguishable by this feature alone, 0 / 1 = fully separated; the features with the largest |AUC - 0.5| are the ones the simulator would have to change   -> auc_<sim>.png
  - distributions of the 8 most different features: box plots per group                                                  -> dist_<sim>.png
  - coverage in this interpretable space: standardised features (corpus median / robust sd), distance of each real frame to the nearest sim frame, median per region    -> in stats_table.md
Needs no GPU."""
import argparse, csv, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
p = argparse.ArgumentParser()
p.add_argument('--corpus_stats', required=True); p.add_argument('--regions', required=True); p.add_argument('--sim', nargs='+', required=True); p.add_argument('--out', required=True); p.add_argument('--k', type=int, default=6)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
zc = np.load(a.corpus_stats); names = [str(n) for n in zc['names']]; S = zc['set']; X = zc['stats']
region = np.array([int(r['region']) for r in csv.DictReader(open(f'{a.regions}/regions_k{a.k}.csv'))])
assert len(region) == len(X)
corpus = S == 'corpus'; lab = ~corpus
K = a.k
groups = {f'R{r}': X[corpus & (region == r)] for r in range(K)}
groups['labelled'] = X[lab]
sims = {}
for pth in a.sim:
    z = np.load(pth); nm = str(z['set'][0]).removeprefix('sim_'); assert [str(n) for n in z['names']] == names; sims[nm] = z['stats']
Xc = X[corpus]; med = np.nanmedian(Xc, 0); rsd = (np.nanpercentile(Xc, 75, 0) - np.nanpercentile(Xc, 25, 0)) / 1.349 + 1e-9


def auc(x, y):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 10 or len(y) < 10:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([x, y])); return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


L = ['# Frame statistics: real regions vs simulators', '', f'real frames: {int(corpus.sum())} corpus + {int(lab.sum())} labelled; ' + ', '.join(f'{n} {len(v)} frames' for n, v in sims.items()), '',
     '| feature | ' + ' | '.join(list(groups) + list(sims)) + ' |', '|---|' + '---|' * (len(groups) + len(sims))]
for j, n in enumerate(names):
    cells = []
    for g in list(groups.values()) + list(sims.values()):
        v = g[:, j]; v = v[np.isfinite(v)]
        cells.append(f'{np.median(v):.3g} ({np.percentile(v, 25):.2g}-{np.percentile(v, 75):.2g})' if len(v) else '-')
    L.append(f'| {n} | ' + ' | '.join(cells) + ' |')
L.append('')
Z = lambda M: (np.nan_to_num(M, nan=0.0) - np.nan_to_num(med)) / rsd                      # standardised features (NaN -> corpus median)
Zc = torch.tensor(Z(Xc), dtype=torch.float32)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'text.color': INK})
div = LinearSegmentedColormap.from_list('div', [BLUE, '#f4f3ef', ORANGE])
cols = list(groups)
for sn, ss in sims.items():
    Zs = torch.tensor(Z(ss), dtype=torch.float32)
    cov = []
    for r in range(K):
        Zr = torch.tensor(Z(groups[f'R{r}'][:3000]), dtype=torch.float32)
        cov.append(float(torch.cdist(Zr, Zs).min(1).values.median()))
    Zl = torch.tensor(Z(groups['labelled']), dtype=torch.float32); covl = float(torch.cdist(Zl, Zs).min(1).values.median())
    ref = []
    for r in range(K):                                                                      # reference: distance of region frames to the nearest of N real frames of the corpus (same N)
        Zr = torch.tensor(Z(groups[f'R{r}'][:3000]), dtype=torch.float32); idx = np.random.default_rng(0).choice(len(Zc), len(Zs), replace=False)
        ref.append(float(torch.cdist(Zr, Zc[idx]).min(1).values.median()))
    L += [f'## {sn}: distance to the nearest simulated frame in standardised statistics space (median), against the nearest of {len(Zs)} random real frames', '',
          '| region | ' + ' | '.join(f'R{r}' for r in range(K)) + ' | labelled |', '|---|' + '---|' * (K + 1),
          '| to nearest sim | ' + ' | '.join(f'{c:.2f}' for c in cov) + f' | {covl:.2f} |', '| to nearest real (same N) | ' + ' | '.join(f'{c:.2f}' for c in ref) + ' | |', '']
    E = np.array([[(np.nanmedian(ss[:, j]) - np.nanmedian(groups[c][:, j])) / rsd[j] for c in cols] for j in range(len(names))])
    A = np.array([[auc(ss[:, j], groups[c][:, j]) for c in cols] for j in range(len(names))])
    L += [f'## {sn}: largest differences (|AUC - 0.5|) per region', '']
    for ci, c in enumerate(cols):
        o = np.argsort(-np.abs(np.nan_to_num(A[:, ci], nan=0.5) - 0.5))[:5]
        L.append(f'- {c}: ' + ', '.join(f'{names[j]} (sim {"higher" if A[j, ci] > .5 else "lower"}, AUC {A[j, ci]:.2f}, effect {E[j, ci]:+.1f})' for j in o))
    L.append('')
    for nm, M, vmin, vmax, ttl, cb in (('effect', E, -3, 3, f'{sn}: median(sim) - median(region), in corpus spreads', 'effect (robust sd of the corpus)'), ('auc', A, 0, 1, f'{sn} vs real region: AUC of each feature (0.5 = same)', 'AUC (sim vs region frames)')):
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(M if nm == 'auc' else np.clip(M, vmin, vmax), cmap=div if nm == 'effect' else div, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(M.shape[0]):
            for jx in range(M.shape[1]):
                ax.text(jx, i, f'{M[i, jx]:+.1f}' if nm == 'effect' else f'{M[i, jx]:.2f}', ha='center', va='center', fontsize=8, color=INK)
        ax.set_title(ttl, loc='left', fontweight='bold', fontsize=12); fig.colorbar(im, ax=ax, fraction=.04, pad=.02).set_label(cb)
        fig.tight_layout(); fig.savefig(f'{a.out}/{nm}_{sn}.png', dpi=130); plt.close(fig)
    score = np.nanmax(np.abs(np.nan_to_num(A[:, [cols.index('R4'), cols.index('R5')]], nan=.5) - .5), 1); top = np.argsort(-score)[:8]
    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    for ax, j in zip(axs.ravel(), top):
        data = [g[:, j][np.isfinite(g[:, j])] for g in list(groups.values()) + [ss]]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color=INK))
        for i, bx in enumerate(bp['boxes']):
            bx.set_facecolor(ORANGE if i == len(data) - 1 else (INK2 if i == len(data) - 2 else BLUE)); bx.set_alpha(.75)
        ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(cols + [sn], rotation=60, fontsize=8); ax.set_title(names[j], loc='left', fontweight='bold'); ax.grid(axis='y', color=GRID, lw=.5)
    fig.suptitle(f'The 8 features where {sn} is most different from real R4 / R5 (blue = real region, grey = labelled, orange = {sn})', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{a.out}/dist_{sn}.png', dpi=110); plt.close(fig)
open(f'{a.out}/stats_table.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[-30:])); print('written to', a.out)
