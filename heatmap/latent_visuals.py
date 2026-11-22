"""Latent map figures (CPU): where do the models score best, and which corpus frames are far from every labelled frame?

  HM_DATA_DIR=<dir> HM_GT_BOX_COEF=3.0,3.0 python heatmap/latent_visuals.py --emb <embed_vith_sk2.npz> --regions <latent_map/vith_sk2> --dets <sweep/dets> --out <dir> [--k 6] [--n_pick 24]

Same embedding as latent_map.py (standardise, PCA 50 fitted on the corpus, k-means regions read from regions_k<k>.csv).
Figures (PNG + csv in --out):
  1_models_on_map     corpus density in PC1-2 (grey), labelled frames coloured by the AP of the competitive models on that frame (mean over models, per-frame AP; one frame is noisy, read the trend), region labels with pooled AP
  2_coverage          corpus frames coloured by the distance to the nearest labelled frame (same 50-d space); numbered rings = the picks of 4_
  3_region_population per region: share of the corpus vs share of the labelled frames, with the pooled AP of the competitive models
  4_picks             thumbnails of the picks: the far 20 % of the corpus (distance to the nearest labelled frame) is clustered (k-means, n_pick clusters) and the frame nearest each cluster centre is shown,
                      so a pick represents a population of far frames instead of a single outlier; the cluster size is its title
picks.csv: set, file, frame, beamtime, region, distance, cluster size.
Competitive = models with a pooled AP >= --competitive on the 49 labelled frames (failing recipes would only drag the map down). Per-frame AP via the repo evaluator, +nms detections, scaled GT."""
import argparse, csv, glob, os, pickle, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from heatmap import evaluation as E
from heatmap.region_sweep import ap_of
from heatmap.latent_map import load_frame

# palette.md (documented): blue slot 1, orange slot 2, aqua slot 3; neutrals
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'

p = argparse.ArgumentParser()
p.add_argument('--emb', required=True); p.add_argument('--regions', required=True); p.add_argument('--dets', required=True); p.add_argument('--out', required=True)
p.add_argument('--k', type=int, default=6); p.add_argument('--dims', type=int, default=50); p.add_argument('--n_pick', type=int, default=24)
p.add_argument('--far_share', type=float, default=0.2); p.add_argument('--competitive', type=float, default=0.60)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---- embedding -> PCA space (as latent_map.py)
z = np.load(a.emb)
X = torch.tensor(z['feat'].astype(np.float32)); S = z['set']; corpus = S == 'corpus'; ct = torch.tensor(corpus)
mu, sd = X[ct].mean(0), X[ct].std(0).clamp_min(1e-6)
Xs = (X - mu) / sd
U, sv, V = torch.linalg.svd(Xs[ct] - Xs[ct].mean(0), full_matrices=False)
var = (sv ** 2 / (sv ** 2).sum()).numpy()
P = ((Xs - Xs[ct].mean(0)) @ V[:a.dims].T)
Pc, Pv = P[ct], P[~ct]
rows_csv = list(csv.DictReader(open(f'{a.regions}/regions_k{a.k}.csv')))      # same row order as the embedding (frame index alone is not unique across corpus scans)
assert len(rows_csv) == len(S) and all(r['set'] == s_ and int(r['frame']) == int(f_) for r, s_, f_ in zip(rows_csv, S, z['frame']))
region = np.array([int(r['region']) for r in rows_csv])
K = a.k
dval = torch.cat([torch.cdist(Pc[i:i + 4096], Pv).min(1).values for i in range(0, len(Pc), 4096)]).numpy()
print(f'{corpus.sum()} corpus, {(~corpus).sum()} labelled; distance to nearest labelled frame: median {np.median(dval):.1f}, 90% {np.quantile(dval, .9):.1f}, max {dval.max():.1f}', flush=True)

# ---- per-frame AP of the competitive models
cfg = E.make_config(E.DATASETS['organic'])
lab_keys = [(s, i) for s in ('organic', '41') for i in range(int((S == s).sum()))]
lab_idx = {(S[j], int(z['frame'][j])): j for j in np.where(~corpus)[0]}
frame_ap = []                                                    # [model][frame]
used = []
for f in sorted(glob.glob(f'{a.dets}/*.pkl')):
    d = pickle.load(open(f, 'rb'))['sets']
    dets = [(np.asarray(fr['boxes'], np.float32).reshape(-1, 4), np.asarray(fr['scores'], np.float32)) for s in ('organic', '41') for fr in d[s]]
    gts = [dict(gt=fr['gt'], gtconf=fr['gtconf'], mask=fr['mask']) for s in ('organic', '41') for fr in d[s]]
    pooled = ap_of(dets, gts, cfg)[0]
    if not pooled >= a.competitive:
        continue
    used.append(os.path.basename(f)[:-4]); frame_ap.append([ap_of([dets[i]], [gts[i]], cfg)[0] for i in range(len(dets))])
frame_ap = np.array(frame_ap)                                    # [M, 49]
fmean = np.nanmean(frame_ap, 0); fspread = np.nanstd(frame_ap, 0)
print(len(used), 'competitive models; per-frame AP mean over models: min %.2f median %.2f max %.2f' % (np.nanmin(fmean), np.nanmedian(fmean), np.nanmax(fmean)), flush=True)
lab_j = np.array([lab_idx[k] for k in lab_keys])
reg_ap = {}                                                      # region -> mean over labelled frames of fmean
for r in range(K):
    m = [i for i, j in enumerate(lab_j) if region[j] == r and not np.isnan(fmean[i])]
    reg_ap[r] = (float(np.mean(fmean[m])), len(m)) if m else (np.nan, 0)
cent = {r: Pc[torch.tensor(region[corpus] == r)].mean(0).numpy() for r in range(K)}
size = {r: int((region[corpus] == r).sum()) for r in range(K)}

# ---- picks: far 20 % of the corpus, clustered
far = np.where(dval >= np.quantile(dval, 1 - a.far_share))[0]
Pf = Pc[far]
g = torch.Generator().manual_seed(0)
c = Pf[torch.randint(len(Pf), (1,), generator=g)]
for _ in range(a.n_pick - 1):
    dd = torch.cdist(Pf, c).min(1).values ** 2
    c = torch.cat([c, Pf[torch.multinomial(dd / dd.sum(), 1, generator=g)]])
for _ in range(100):
    lb = torch.cdist(Pf, c).argmin(1)
    new = torch.stack([Pf[lb == j].mean(0) if (lb == j).any() else c[j] for j in range(a.n_pick)])
    if torch.allclose(new, c, atol=1e-5):
        break
    c = new
lb = torch.cdist(Pf, c).argmin(1)
corp_idx = np.where(corpus)[0]
picks = []
for j in range(a.n_pick):
    mem = torch.where(lb == j)[0]
    if len(mem) == 0:
        continue
    medoid = mem[torch.cdist(Pf[mem], c[j:j + 1])[:, 0].argmin()].item()
    picks.append((len(mem), far[medoid]))
picks.sort(key=lambda t: -t[0])                                  # biggest far population first
rows = []
for n, (cnt, ci) in enumerate(picks, 1):
    j = corp_idx[ci]
    rows.append(dict(n=n, j=j, ci=ci, cluster=cnt, dist=float(dval[ci]), region=int(region[j]), file=str(z['path'][j]), frame=int(z['frame'][j]), beamtime=str(z['beamtime'][j])))
with open(f'{a.out}/picks.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['n', 'set', 'file', 'frame', 'beamtime', 'region', 'dist_to_nearest_labelled', 'far_cluster_size'])
    for r in rows:
        w.writerow([r['n'], 'corpus', r['file'], r['frame'], r['beamtime'], r['region'], f"{r['dist']:.2f}", r['cluster']])
cov_by_region = {r: float(np.median(dval[region[corpus] == r])) for r in range(K)}
far_by_region = collections.Counter(region[corp_idx[far]].tolist())
bt_far = collections.Counter(z['beamtime'][corp_idx[far]].tolist()).most_common(6)
print('far 20% of the corpus by region:', dict(far_by_region)); print('far 20% by beamtime:', bt_far)
print('median distance to nearest labelled frame per region:', {r: round(v, 1) for r, v in cov_by_region.items()})

# ---- figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.facecolor': 'white', 'figure.facecolor': 'white', 'text.color': INK})
seq = LinearSegmentedColormap.from_list('blue', ['#e8f0fb', '#9cc2f0', BLUE, '#16417a'])           # one hue, light -> dark
seq_o = LinearSegmentedColormap.from_list('orange', ['#fdeee7', '#f7b497', ORANGE, '#8a2f0a'])
xy = P[:, :2].numpy(); xc = xy[corpus]


def base(ax, title, sub):
    hb = ax.hexbin(xc[:, 0], xc[:, 1], gridsize=70, bins='log', cmap=LinearSegmentedColormap.from_list('g', ['#f4f3ef', '#c9c8c2']), mincnt=1, linewidths=0)
    for r in range(K):
        ax.text(cent[r][0], cent[r][1], f'R{r}', ha='center', va='center', fontsize=11, fontweight='bold', color=INK, zorder=3,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GRID, alpha=.85))
    ax.set_xlabel(f'PC1 ({var[0]:.0%} of variance)'); ax.set_ylabel(f'PC2 ({var[1]:.0%})')
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold'); ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=9, color=INK2, va='bottom')
    ax.grid(color=GRID, lw=.4, alpha=.5); ax.set_axisbelow(True)


# 1: models on the map
fig, ax = plt.subplots(figsize=(11, 7.5))
base(ax, 'Where the models score: AP per labelled frame, on the latent map', '')
ax.set_title('Where the models score best on the latent map', loc='left', fontsize=13, fontweight='bold', pad=22)
ax.text(0, 1.02, f'grey: {int(corpus.sum()):,} corpus frames (darker = denser) | dots: the 49 labelled frames, colour = mean per-frame AP of {len(used)} competitive models | R = k-means region (k={K})', transform=ax.transAxes, fontsize=8.5, color=INK2)
for s, mk in (('organic', 'o'), ('41', '^')):
    idx = [i for i, k in enumerate(lab_keys) if k[0] == s and not np.isnan(fmean[i])]
    sc = ax.scatter(xy[lab_j[idx], 0], xy[lab_j[idx], 1], c=fmean[idx], cmap=seq, vmin=0, vmax=1, s=90, marker=mk, edgecolor=INK, linewidth=0.9, zorder=5, label=f'{s} ({len(idx)} frames with GT)')
nogt = [i for i in range(len(lab_keys)) if np.isnan(fmean[i])]
if nogt:
    ax.scatter(xy[lab_j[nogt], 0], xy[lab_j[nogt], 1], s=25, marker='x', color=INK2, zorder=5, label='no GT in frame')
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.set_label('mean per-frame AP over the competitive models')
ax.legend(loc='upper right', fontsize=9, frameon=False)
# region table inside the figure (bottom right, direct text)
txt = 'region: labelled frames, mean per-frame AP\n' + '\n'.join(f'R{r}: {reg_ap[r][1]:>2} frames, AP ' + (f'{reg_ap[r][0]:.2f}' if reg_ap[r][1] else '-') + f'   ({size[r]:,} corpus)' for r in range(K))
ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, family='monospace', color=INK2, bbox=dict(fc='white', ec=GRID, boxstyle='round,pad=0.4'))
fig.tight_layout(); fig.savefig(f'{a.out}/1_models_on_map.png', dpi=140); plt.close(fig)

# 2: coverage
fig, ax = plt.subplots(figsize=(11, 7.5))
base(ax, '', '')
ax.set_title('Which corpus frames the validation sets do not cover', loc='left', fontsize=13, fontweight='bold', pad=22)
ax.text(0, 1.02, 'colour = distance of a corpus frame to the nearest labelled frame (50-d feature space; darker = further from anything we validate on) | rings: the picks of figure 4', transform=ax.transAxes, fontsize=8.5, color=INK2)
o = np.argsort(dval)
sc = ax.scatter(xc[o, 0], xc[o, 1], c=dval[o], cmap=seq_o, s=3, alpha=.8, rasterized=True, vmin=np.quantile(dval, .02), vmax=np.quantile(dval, .99))
ax.scatter(xy[lab_j, 0], xy[lab_j, 1], s=50, marker='^', facecolor='white', edgecolor=INK, linewidth=1.1, zorder=6, label='labelled frames (organic + 41)')
for r in rows:
    ax.scatter(*xc[r['ci']], s=210, facecolor='none', edgecolor=INK, linewidth=1.4, zorder=7)
    ax.text(*xc[r['ci']], str(r['n']), ha='center', va='center', fontsize=7, fontweight='bold', zorder=8)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.set_label('distance to the nearest labelled frame')
ax.legend(loc='upper right', fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(f'{a.out}/2_coverage.png', dpi=140); plt.close(fig)

# 3: region population
fig, ax = plt.subplots(figsize=(10, 5.2))
xs = np.arange(K); w = 0.38
cs = np.array([size[r] for r in range(K)], float); ls = np.array([reg_ap[r][1] for r in range(K)], float)
b1 = ax.bar(xs - w / 2 - 0.01, cs / cs.sum() * 100, w, color=BLUE, label='share of the corpus (%)')
b2 = ax.bar(xs + w / 2 + 0.01, ls / ls.sum() * 100, w, color=ORANGE, label=f'share of the {int(ls.sum())} labelled frames (%)')
for x, v in zip(xs - w / 2, cs / cs.sum() * 100):
    ax.text(x, v + 0.8, f'{v:.0f}', ha='center', fontsize=9, color=INK2)
for x, v, n in zip(xs + w / 2, ls / ls.sum() * 100, ls):
    ax.text(x, v + 0.8, f'{v:.0f} ({int(n)})', ha='center', fontsize=9, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels([f'R{r}\nAP ' + (f'{reg_ap[r][0]:.2f}' if reg_ap[r][1] else '-') + f'\n{cov_by_region[r]:.0f} to nearest' for r in range(K)], fontsize=9)
ax.set_ylabel('% of frames'); ax.set_title('Corpus versus validation population per region', loc='left', fontsize=13, fontweight='bold', pad=22)
ax.text(0, 1.02, 'under each region: mean per-frame AP of the competitive models, median distance of its corpus frames to the nearest labelled frame', transform=ax.transAxes, fontsize=8.5, color=INK2)
ax.legend(frameon=False, loc='upper right'); ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{a.out}/3_region_population.png', dpi=140); plt.close(fig)

# 4: picks as thumbnails (+ two labelled frames per region for reference would be too many; the picks only)
nc = 6; nr = int(np.ceil(len(rows) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr))
for ax in axs.ravel():
    ax.axis('off')
for r, ax in zip(rows, axs.ravel()):
    try:
        ax.imshow(load_frame('corpus', r['file'], r['frame']), cmap='gray', aspect='auto')
    except Exception as e:
        ax.text(.5, .5, f'could not load\n{e}'[:80], ha='center', va='center', fontsize=7, transform=ax.transAxes)
    ax.set_title(f"{r['n']}  R{r['region']}  {r['beamtime'][:22]}\n{os.path.basename(r['file'])[:26]} #{r['frame']}\nd={r['dist']:.0f}, represents {r['cluster']:,} far frames", fontsize=7, color=INK)
fig.suptitle(f'Picks: one frame per cluster of the far {a.far_share:.0%} of the corpus (biggest cluster first); q across, chi down', fontsize=12, fontweight='bold', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(f'{a.out}/4_picks.png', dpi=90); plt.close(fig)
print('written to', a.out)

# 5: map with example corpus frames and arrows to where they lie
from matplotlib.patches import ConnectionPatch
cidx = np.where(corpus)[0]
OVERRIDE = {0: 12208}                                              # hand-picked frame (global embedding index) near the R0 centre: ivan_pb300_00669_pe.h5 #11, clear peaks; the nearest frame is a blurry blob
sel = []                                                          # (corpus row, label)
for r in range(K):
    d = torch.cdist(Pc, torch.tensor(cent[r])[None])[:, 0].numpy(); d[region[cidx] != r] = np.inf
    sel.append((int(d.argmin()) if r not in OVERRIDE else int(np.searchsorted(cidx, OVERRIDE[r])), f'R{r} centre'))
for lbl, arr in (('R3 far end', xc[:, 0]), ('R1 lower tip', -xc[:, 1])):
    thr = np.quantile(arr, 0.995); cand = np.where(arr >= thr)[0]
    sel.append((int(cand[np.argsort(arr[cand])[len(cand) // 2]]), lbl))
sel.sort(key=lambda t: xc[t[0], 0]); half = len(sel) // 2
cols = [sorted(sel[:half], key=lambda t: -xc[t[0], 1]), sorted(sel[half:], key=lambda t: -xc[t[0], 1])]
fig = plt.figure(figsize=(17, 9.5))
axm = fig.add_axes([0.25, 0.07, 0.52, 0.86])
base(axm, '', '')
axm.set_title('Latent map with example corpus frames', loc='left', fontsize=13, fontweight='bold', pad=22)
axm.text(0, 1.02, 'each thumbnail is a real corpus frame; the arrow ends on its position in PC1-2 | triangles: labelled frames', transform=axm.transAxes, fontsize=8.5, color=INK2)
axm.scatter(xy[lab_j, 0], xy[lab_j, 1], s=35, marker='^', facecolor='white', edgecolor=INK, linewidth=.9, zorder=4)
for side, items in enumerate(cols):
    n = len(items); h = 0.86 / n
    for i, (ci, lbl) in enumerate(items):
        x0 = 0.01 if side == 0 else 0.80
        ax = fig.add_axes([x0, 0.93 - (i + 1) * h + 0.035, 0.19, h - 0.06])
        j = cidx[ci]
        try:
            ax.imshow(load_frame('corpus', str(z['path'][j]), int(z['frame'][j])), cmap='gray', aspect='auto')
        except Exception as e:
            ax.text(.5, .5, 'could not load', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK2); sp.set_linewidth(1.2)
        ax.set_title(f'{lbl}: {z["beamtime"][j][:24]}', fontsize=8.5, color=INK, pad=3)
        axm.scatter(*xc[ci], s=60, facecolor=ORANGE, edgecolor=INK, linewidth=1, zorder=6)
        fig.add_artist(ConnectionPatch(xyA=(1.0 if side == 0 else 0.0, 0.5), coordsA=ax.transAxes, xyB=tuple(xc[ci]), coordsB=axm.transData, arrowstyle='-|>', color=INK2, lw=1.2, mutation_scale=12, shrinkB=4, zorder=5))
fig.savefig(f'{a.out}/5_map_with_examples.png', dpi=130); plt.close(fig)
print('figure 5 written')
