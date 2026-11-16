"""Latent map step 2 (CPU): regions of the corpus in feature space, and where the labelled validation frames sit in it.

  python heatmap/latent_map.py --emb <embed_corpus.npz> --out <dir> [--k 6 8 10] [--dims 50]

How it works:
  1. standardise each feature channel with the corpus mean / std (no channel dominates just by scale)
  2. PCA fitted on the corpus frames only, keep --dims components (most of the variance, much less noise and faster distances)
  3. k-means on the corpus frames (k-means++ start, 5 restarts, best inertia) for every k in --k -> k regions
  4. validation frames are projected with the same standardisation + PCA and given the nearest region centre (they never move a centre)
  5. coverage: for every corpus frame the distance to the nearest validation frame, compared with the typical distance between
     neighbouring corpus frames; a region whose frames are all far from any validation frame is one we cannot score at all
Outputs in <dir>: regions_k<k>.csv (one row per frame: set, file, frame, beamtime, region, distance to the nearest validation frame),
summary_k<k>.txt (per region: corpus frames, scans, beamtimes, organic / 41 frames, coverage), map_k<k>.png (PCA 1-2 scatter),
examples_k<k>.png (per region the 6 corpus frames closest to its centre, each from a different scan, plus up to 3 validation frames of the region).
"""
import argparse, csv, os, sys
from collections import Counter
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def kmeans(x, k, restarts=5, iters=100, seed=0):
    g = torch.Generator().manual_seed(seed); best = None
    for _ in range(restarts):
        c = x[torch.randint(len(x), (1,), generator=g)]
        for _ in range(k - 1):                                    # k-means++ start
            d = torch.cdist(x, c).min(1).values ** 2
            c = torch.cat([c, x[torch.multinomial(d / d.sum(), 1, generator=g)]])
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)
            new = torch.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if torch.allclose(new, c, atol=1e-5):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum().item()
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    # order regions by size (region 0 = largest), so the numbering is stable to read
    _, c, lab = best
    order = torch.bincount(lab, minlength=k).argsort(descending=True)
    c = c[order]
    return c, torch.cdist(x, c).argmin(1)


def load_frame(s, f, i):
    """the HE image as the backbone saw it (corpus: from the h5; validation: through evaluation.iter_frames)"""
    if s == 'corpus':
        import h5py
        from heatmap.embed_corpus import decode, DEPLOYED
        from util.exp_preprocess import apply_contrast
        with h5py.File(f, 'r') as h:
            raw, ok = decode(h['images'][int(i)], float(h.attrs['log_max']))
        return apply_contrast(raw, ok, DEPLOYED)
    from heatmap import evaluation as E
    for n, (cfg, ic) in enumerate(E.iter_frames(f)):
        if n == int(i):
            return E.frame_inputs(ic, 'cpu')[0, 0].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb', required=True); p.add_argument('--out', required=True)
    p.add_argument('--k', type=int, nargs='+', default=[6, 8, 10]); p.add_argument('--dims', type=int, default=50)
    p.add_argument('--no_examples', action='store_true')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.emb)
    X = torch.tensor(z['feat'].astype(np.float32)); S = z['set']; corpus = torch.tensor(S == 'corpus')
    mu, sd = X[corpus].mean(0), X[corpus].std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    U, sv, V = torch.linalg.svd(Xs[corpus] - Xs[corpus].mean(0), full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    P = (Xs - Xs[corpus].mean(0)) @ V[:a.dims].T                       # [N, dims]
    print(f'{int(corpus.sum())} corpus frames, {int((~corpus).sum())} validation frames; PCA {a.dims} dims keep {var[:a.dims].sum():.1%} of the variance (PC1 {var[0]:.1%}, PC2 {var[1]:.1%})')
    Pc, Pv = P[corpus], P[~corpus]
    # coverage: distance to the nearest validation frame, against the typical nearest-neighbour distance inside the corpus (500-frame sample)
    dval = torch.cat([torch.cdist(Pc[i:i + 4096], Pv).min(1).values for i in range(0, len(Pc), 4096)])
    samp = Pc[torch.randperm(len(Pc), generator=torch.Generator().manual_seed(0))[:500]]
    nn_corpus = torch.cdist(samp, Pc).topk(2, largest=False).values[:, 1].median().item()
    dist_all = torch.zeros(len(P)); dist_all[corpus] = dval
    print(f'typical corpus nearest-neighbour distance {nn_corpus:.2f}; corpus to nearest validation frame: median {dval.median():.2f}, 90 % {dval.quantile(0.9):.2f}')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pal = plt.get_cmap('tab10').colors
    for k in a.k:
        C, lab_c = kmeans(Pc, k)
        lab = torch.cdist(P, C).argmin(1)
        with open(f'{a.out}/regions_k{k}.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['set', 'file', 'frame', 'beamtime', 'held_out', 'region', 'dist_to_nearest_val'])
            for j in range(len(P)):
                w.writerow([S[j], z['path'][j], int(z['frame'][j]), z['beamtime'][j], bool(z['held_out'][j]), int(lab[j]), f'{dist_all[j]:.3f}' if S[j] == 'corpus' else ''])
        lines = [f'k = {k}; coverage = share of the region\'s corpus frames within 2x the typical corpus neighbour distance ({2 * nn_corpus:.2f}) of a validation frame',
                 f'{"region":>6} {"corpus":>7} {"scans":>6} {"beamtimes":>9} {"organic":>7} {"41":>4} {"coverage":>8}  top beamtimes']
        for r in range(k):
            mc = (lab == r) & corpus
            files = set(z['path'][mc.numpy()]); bts = Counter(z['beamtime'][mc.numpy()])
            cov = (dist_all[mc] < 2 * nn_corpus).float().mean().item() if mc.any() else float('nan')
            lines.append(f'{r:>6} {int(mc.sum()):>7} {len(files):>6} {len(bts):>9} {int(((lab == r) & torch.tensor(S == "organic")).sum()):>7} '
                         f'{int(((lab == r) & torch.tensor(S == "41")).sum()):>4} {cov:>8.0%}  ' + ', '.join(f'{b} {n}' for b, n in bts.most_common(3)))
        open(f'{a.out}/summary_k{k}.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
        # map: corpus coloured by region (small, light), validation frames on top
        fig, ax = plt.subplots(figsize=(9, 7))
        for r in range(k):
            m = ((lab == r) & corpus).numpy()
            ax.scatter(P[m, 0], P[m, 1], s=2, alpha=0.25, color=pal[r % 10], label=f'region {r}', rasterized=True)
        for s, mk in (('organic', 'o'), ('41', '^')):
            m = S == s
            ax.scatter(P[m, 0], P[m, 1], s=22, marker=mk, facecolor='none', edgecolor='black', linewidth=0.8, label=f'{s} (labelled)')
        ax.set_xlabel(f'PC1 ({var[0]:.0%} of variance)'); ax.set_ylabel(f'PC2 ({var[1]:.0%})')
        ax.set_title(f'Corpus frames in feature space, {k} regions (k-means on {a.dims} PCs; this view shows only PC1-2)')
        ax.legend(markerscale=4, fontsize=8, loc='best'); fig.tight_layout(); fig.savefig(f'{a.out}/map_k{k}.png', dpi=130); plt.close(fig)
        if a.no_examples:
            continue
        # examples: the 6 corpus frames nearest each centre (one per scan) and up to 3 validation frames of the region
        fig, axs = plt.subplots(k, 9, figsize=(27, 2.2 * k))
        for r in range(k):
            d = torch.cdist(P, C[r:r + 1])[:, 0]; seen, picks = set(), []
            for j in d.argsort().tolist():
                if S[j] == 'corpus' and lab[j] == r and z['path'][j] not in seen:
                    seen.add(z['path'][j]); picks.append(j)
                if len(picks) == 6:
                    break
            vals = [j for j in d.argsort().tolist() if S[j] != 'corpus' and lab[j] == r][:3]
            for c, j in enumerate(picks + [None] * (6 - len(picks)) + vals + [None] * (3 - len(vals))):
                ax = axs[r, c]; ax.axis('off')
                if j is None:
                    continue
                ax.imshow(load_frame(S[j], z['path'][j], z['frame'][j]), cmap='gray', aspect='auto')
                ax.set_title(f'{"R" + str(r) + " " if c == 0 else ""}{S[j]} {os.path.basename(str(z["path"][j]))[:22]} #{int(z["frame"][j])}', fontsize=7,
                             color='black' if S[j] == 'corpus' else 'darkred')
        fig.suptitle(f'k = {k}: per region (rows) the 6 corpus frames closest to its centre (one per scan) | up to 3 labelled frames of the region (red titles); q across, chi down')
        fig.tight_layout(); fig.savefig(f'{a.out}/examples_k{k}.png', dpi=80); plt.close(fig)
    print('written to', a.out)


if __name__ == '__main__':
    main()
