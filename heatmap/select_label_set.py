"""Pick frames of one latent-map region to label (CPU): k-means over the region's frames in the 50-d PCA space, the frame nearest each centre (one per scan), mapped back to its raw scan and frame.

  python heatmap/select_label_set.py --emb <embed_vith_sk2.npz> --regions <latent_map/vith_sk2> --region 3 --beamtime 2024_07_ESRF_ID10_Dima --n 15 --out <selection.json>

selection.json: one entry per frame (biggest cluster first): corpus h5 + frame, the raw frame name ('eiger4m_0000.h5#i'), the raw scan folder (on the group share) and the geometry the corpus conversion used.
Convert them for mlgidGUI with heatmap/convert_label_set.py."""
import argparse, csv, json, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, h5py
p = argparse.ArgumentParser()
p.add_argument('--emb', required=True); p.add_argument('--regions', required=True); p.add_argument('--k', type=int, default=6); p.add_argument('--region', type=int, required=True)
p.add_argument('--beamtime', default=None); p.add_argument('--n', type=int, default=15); p.add_argument('--out', required=True); p.add_argument('--dims', type=int, default=50)
a = p.parse_args()
z = np.load(a.emb); S = z['set']; ct = torch.tensor(S == 'corpus')
X = torch.tensor(z['feat'].astype(np.float32)); mu, sd = X[ct].mean(0), X[ct].std(0).clamp_min(1e-6); Xs = (X - mu) / sd; c0 = Xs[ct].mean(0)
U, s, V = torch.linalg.svd(Xs[ct] - c0, full_matrices=False); P = (Xs - c0) @ V[:a.dims].T
reg = np.array([int(r['region']) for r in csv.DictReader(open(f'{a.regions}/regions_k{a.k}.csv'))])
m = (reg == a.region) & (S == 'corpus')
if a.beamtime:
    m &= z['beamtime'] == a.beamtime
idx = np.where(m)[0]; Pr = P[idx]; K = a.n
print('frames', len(idx), 'scans', len(set(z['path'][idx])))
g = torch.Generator().manual_seed(0)
c = Pr[torch.randint(len(Pr), (1,), generator=g)]
for _ in range(K - 1):
    d = torch.cdist(Pr, c).min(1).values ** 2; c = torch.cat([c, Pr[torch.multinomial(d / d.sum(), 1, generator=g)]])
for _ in range(100):
    lb = torch.cdist(Pr, c).argmin(1); new = torch.stack([Pr[lb == j].mean(0) if (lb == j).any() else c[j] for j in range(K)])
    if torch.allclose(new, c, atol=1e-5):
        break
    c = new
lb = torch.cdist(Pr, c).argmin(1); used, sel = set(), []
for j in range(K):
    mem = torch.where(lb == j)[0]; order = mem[torch.cdist(Pr[mem], c[j:j + 1])[:, 0].argsort()]
    for q in order.tolist():
        pth = str(z['path'][idx[q]])
        if pth not in used:
            used.add(pth); sel.append((len(mem), int(idx[q]))); break
sel.sort(key=lambda t: -t[0]); out = []
for n, (cnt, j) in enumerate(sel):
    pth = str(z['path'][j]); i = int(z['frame'][j])
    with h5py.File(pth, 'r') as h:
        out.append(dict(n=n + 1, cluster=cnt, region=a.region, corpus_h5=pth, corpus_idx=i, raw_name=h['frame'][i].decode(), folder=str(h.attrs['folder']), beamtime=str(h.attrs['beamtime']),
                        center_x=float(h.attrs['center_x']), center_y=float(h.attrs['center_y']), distance_mm=float(h.attrs['distance_mm']), energy_eV=float(h.attrs['energy_eV']), pixel_um=float(h.attrs['pixel_um'])))
    print(out[-1]['n'], cnt, out[-1]['raw_name'], out[-1]['folder'].split('/')[-4:])
json.dump(out, open(a.out, 'w'), indent=1)
