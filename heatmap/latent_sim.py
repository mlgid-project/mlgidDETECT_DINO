"""Where do simulated training frames sit on the latent map, and which corpus regions has a simulator no counterpart for?  (CPU)

  python heatmap/latent_sim.py --emb <embed_vith_sk2.npz> --sim <embed_sim_legacy.npz> [<embed_sim_realsim_v2.npz> ...] --regions <latent_map/vith_sk2> --out <dir> [--k 6]

Same standardisation + PCA (50, fitted on the corpus only) and regions as latent_map.py / latent_visuals.py; sim frames are projected, never fitted. Per simulator (name = 'set' field of its npz):
  - region of every sim frame (nearest region centre = mean of the region's corpus frames); share of sim vs corpus per region
  - sim frame -> nearest corpus frame, against corpus frame -> nearest corpus frame of ANOTHER scan (spacing of real data)
  - per region: median distance corpus frame -> nearest sim frame (the number to compare between regions and between simulators), and the same for the labelled frames
  - gap ratio = that median / median distance to the nearest of N RANDOM real frames of other scans (N = number of sim frames); ONLY meaningful as a ranking of regions: every simulator sits far from all real frames
    (a domain offset), so the absolute ratio is large everywhere
Figures: sim_on_map.png (corpus density, every simulator, labelled frames), sim_region_gap.png (shares per region, median distance corpus -> nearest sim frame per region and simulator),
sim_gap_map_<name>.png (corpus coloured by distance to the nearest sim frame), sim_examples_<name>.png (the dumped sim frames, if <npz>_frames.npy exists). Writes sim_summary.txt."""
import argparse, csv, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
SIMCOL = [ORANGE, AQUA, '#e87ba4']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
p = argparse.ArgumentParser()
p.add_argument('--emb', required=True); p.add_argument('--sim', nargs='+', required=True); p.add_argument('--regions', required=True); p.add_argument('--out', required=True)
p.add_argument('--k', type=int, default=6); p.add_argument('--dims', type=int, default=50); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
z = np.load(a.emb)
X = torch.tensor(z['feat'].astype(np.float32)); S = z['set']; corpus = S == 'corpus'; ct = torch.tensor(corpus)
mu, sd = X[ct].mean(0), X[ct].std(0).clamp_min(1e-6)
Xs = (X - mu) / sd; c0 = Xs[ct].mean(0)
U, sv, V = torch.linalg.svd(Xs[ct] - c0, full_matrices=False)
var = (sv ** 2 / (sv ** 2).sum()).numpy()
P = (Xs - c0) @ V[:a.dims].T
Pc, Pl = P[ct], P[~ct]; K = a.k
rows = list(csv.DictReader(open(f'{a.regions}/regions_k{K}.csv')))
region = np.array([int(r['region']) for r in rows]); rc = region[corpus]
cent = torch.stack([Pc[torch.tensor(rc == r)].mean(0) for r in range(K)])
files = np.array(z['path'])[corpus]
rng = np.random.default_rng(a.seed)
cs = np.array([(rc == r).sum() for r in range(K)])
lreg = torch.cdist(Pl, cent).argmin(1).numpy()


def nn_other_scan(idx, ref_idx):
    out = np.zeros(len(idx)); R = Pc[ref_idx]
    for s in range(0, len(idx), 1024):
        d = torch.cdist(Pc[idx[s:s + 1024]], R)
        d[torch.tensor(files[idx[s:s + 1024]][:, None] == files[ref_idx][None, :])] = float('inf'); out[s:s + 1024] = d.min(1).values.numpy()
    return out


samp = rng.choice(len(Pc), 3000, replace=False)
d_cc = nn_other_scan(samp, np.arange(len(Pc)))
L = [f'{int(corpus.sum())} corpus frames, PCA {a.dims} dims ({var[:a.dims].sum():.0%} of variance), k = {K}',
     f'real spacing: corpus -> nearest corpus frame of another scan (3000 sample): median {np.median(d_cc):.1f}, 95% {np.quantile(d_cc, .95):.1f}', '']
R = {}
for path in a.sim:
    zs = np.load(path); name = str(zs['set'][0]).removeprefix('sim_'); N = len(zs['feat'])
    Q = ((torch.tensor(zs['feat'].astype(np.float32)) - mu) / sd - c0) @ V[:a.dims].T
    rs = torch.cdist(Q, cent).argmin(1).numpy()
    ref = rng.choice(len(Pc), N, replace=False)
    d_cs = torch.cat([torch.cdist(Pc[s:s + 2048], Q).min(1).values for s in range(0, len(Pc), 2048)]).numpy()
    d_cr = nn_other_scan(np.arange(len(Pc)), ref)
    d_sc = torch.cat([torch.cdist(Q[s:s + 1024], Pc).min(1).values for s in range(0, N, 1024)]).numpy()
    d_ls = torch.cdist(Pl, Q).min(1).values.numpy()
    ss = np.array([(rs == r).sum() for r in range(K)])
    L += [f'== {name}: {N} frames', f'sim -> nearest corpus frame: median {np.median(d_sc):.1f}, 90% {np.quantile(d_sc, .9):.1f}; corpus -> nearest sim frame: median {np.median(d_cs):.1f}; '
          f'labelled -> nearest sim frame: median {np.median(d_ls):.1f}',
          f'{"region":>6} {"corpus":>8} {"share":>6} {"sim":>6} {"share":>6} {"d(corpus->sim)":>15} {"d(labelled->sim)":>17} {"gap ratio":>10}']
    for r in range(K):
        m = rc == r; ml = lreg == r
        L.append(f'{r:>6} {cs[r]:>8} {cs[r] / cs.sum():>6.0%} {ss[r]:>6} {ss[r] / N:>6.0%} {np.median(d_cs[m]):>15.1f} ' + (f'{np.median(d_ls[ml]):>17.1f}' if ml.any() else f'{"-":>17}') + f' {np.median(d_cs[m]) / np.median(d_cr[m]):>10.2f}')
    L.append('')
    R[name] = dict(Q=Q, rs=rs, N=N, ss=ss, d_cs=d_cs, path=path, zs=zs, d_sc=d_sc)
open(f'{a.out}/sim_summary.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'text.color': INK})
xc, xl = Pc[:, :2].numpy(), Pl[:, :2].numpy(); cp = {r: cent[r][:2].numpy() for r in range(K)}
grey = LinearSegmentedColormap.from_list('g', ['#f4f3ef', '#c9c8c2'])


def base(ax, title, sub):
    ax.hexbin(xc[:, 0], xc[:, 1], gridsize=70, bins='log', cmap=grey, mincnt=1, linewidths=0)
    for r in range(K):
        ax.text(*cp[r], f'R{r}', ha='center', va='center', fontsize=11, fontweight='bold', zorder=3, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID, alpha=.85))
    ax.set_xlabel(f'PC1 ({var[0]:.0%} of variance)'); ax.set_ylabel(f'PC2 ({var[1]:.0%})')
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=22); ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.grid(color=GRID, lw=.4, alpha=.5); ax.set_axisbelow(True)


names = list(R)
fig, ax = plt.subplots(figsize=(11, 7.5))
base(ax, 'Simulated training frames on the latent map', 'grey: corpus density | coloured dots: simulated frames (same embedding) | triangles: labelled frames')
for i, n in enumerate(names):
    ax.scatter(R[n]['Q'][:, 0], R[n]['Q'][:, 1], s=8, color=SIMCOL[i], alpha=.5, linewidth=0, zorder=4 + i, label=f'{n} ({R[n]["N"]} frames)')
ax.scatter(xl[:, 0], xl[:, 1], s=40, marker='^', facecolor='white', edgecolor=INK, linewidth=.9, zorder=9, label='labelled frames')
ax.legend(loc='upper right', frameon=False, markerscale=3); fig.tight_layout(); fig.savefig(f'{a.out}/sim_on_map.png', dpi=140); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
xs = np.arange(K); nb = 1 + len(names); w = 0.8 / nb
a1.bar(xs - 0.4 + w / 2, cs / cs.sum() * 100, w, color=BLUE, label='corpus')
for i, n in enumerate(names):
    v = R[n]['ss'] / R[n]['N'] * 100; a1.bar(xs - 0.4 + w * (i + 1.5), v, w, color=SIMCOL[i], label=n)
    for x, y in zip(xs - 0.4 + w * (i + 1.5), v):
        a1.text(x, y + .8, f'{y:.0f}', ha='center', fontsize=8, color=INK2)
for x, y in zip(xs - 0.4 + w / 2, cs / cs.sum() * 100):
    a1.text(x, y + .8, f'{y:.0f}', ha='center', fontsize=8, color=INK2)
a1.set_xticks(xs); a1.set_xticklabels([f'R{r}' for r in range(K)]); a1.set_ylabel('% of frames'); a1.legend(frameon=False); a1.grid(axis='y', color=GRID, lw=.5); a1.set_axisbelow(True)
a1.set_title('Share of frames per region', loc='left', fontweight='bold')
for i, n in enumerate(names):
    med = [np.median(R[n]['d_cs'][rc == r]) for r in range(K)]
    a2.bar(xs - 0.4 + 0.4 / len(names) + (0.8 / len(names)) * i, med, 0.8 / len(names), color=SIMCOL[i], label=n)
    for x, y in zip(xs - 0.4 + 0.4 / len(names) + (0.8 / len(names)) * i, med):
        a2.text(x, y + .3, f'{y:.0f}', ha='center', fontsize=8, color=INK2)
a2.set_xticks(xs); a2.set_xticklabels([f'R{r}' for r in range(K)]); a2.set_ylabel('median distance, corpus frame -> nearest simulated frame'); a2.legend(frameon=False)
a2.set_title('How far each region is from the simulator (lower = closer)', loc='left', fontweight='bold'); a2.grid(axis='y', color=GRID, lw=.5); a2.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{a.out}/sim_region_gap.png', dpi=140); plt.close(fig)

seq = LinearSegmentedColormap.from_list('b', ['#e8f0fb', '#9cc2f0', BLUE, '#16417a'])
for i, n in enumerate(names):
    d_cs = R[n]['d_cs']; o = np.argsort(d_cs)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    base(ax, f'Corpus frames the {n} simulator has no counterpart for', 'colour = distance of a corpus frame to the nearest simulated frame (darker = nothing like it is simulated) | dots: simulated frames')
    sc = ax.scatter(xc[o, 0], xc[o, 1], c=d_cs[o], cmap=seq, s=3, alpha=.8, rasterized=True, vmin=np.quantile(d_cs, .02), vmax=np.quantile(d_cs, .99))
    ax.scatter(R[n]['Q'][:, 0], R[n]['Q'][:, 1], s=5, color=SIMCOL[i], alpha=.5, linewidth=0, zorder=4)
    fig.colorbar(sc, ax=ax, fraction=.035, pad=.02).set_label('distance to the nearest simulated frame')
    fig.tight_layout(); fig.savefig(f'{a.out}/sim_gap_map_{n}.png', dpi=140); plt.close(fig)
    fp = R[n]['path'].replace('.npz', '_frames.npy')
    if os.path.exists(fp):
        fr = np.load(fp); m_ = min(len(fr), 24)
        fig, axs = plt.subplots(4, 6, figsize=(18, 8))
        for j, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if j < m_:
                ax.imshow(fr[j], cmap='gray', aspect='auto'); ax.set_title(f'{n} #{j}: region R{R[n]["rs"][j]}, d {R[n]["d_sc"][j]:.0f}, {int(R[n]["zs"]["n_boxes"][j])} boxes', fontsize=8)
        fig.tight_layout(); fig.savefig(f'{a.out}/sim_examples_{n}.png', dpi=80); plt.close(fig)
print('written to', a.out)
