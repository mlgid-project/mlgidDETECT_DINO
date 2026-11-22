"""frame_stats of the real frames of the latent-map embedding (corpus 5 frames per scan + organic + 41), same order as embed_vith_sk2.npz  (CPU, multiprocessing).

  python heatmap/corpus_stats.py --emb <embed_vith_sk2.npz> --out <stats_corpus.npz> [--workers 4]"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, h5py
from multiprocessing import Pool
from heatmap.frame_stats import frame_stats, NAMES


def work(job):
    path, idxs, kind = job
    from heatmap.embed_corpus import decode, DEPLOYED
    from util.exp_preprocess import apply_contrast
    out = []
    if kind == 'corpus':
        with h5py.File(path, 'r') as h:
            lm = float(h.attrs['log_max'])
            for i in idxs:
                raw, ok = decode(h['images'][int(i)], lm); out.append(frame_stats(apply_contrast(raw, ok, DEPLOYED), ok))
    else:
        os.environ.setdefault('HM_DATA_DIR', os.path.dirname(path))
        from heatmap import evaluation as E
        want = set(int(i) for i in idxs)
        for n, (cfg, ic) in enumerate(E.iter_frames(path)):
            if n in want:
                out.append(frame_stats(E.frame_inputs(ic, 'cpu')[0, 0].numpy(), E.frame_mask(ic, 'cpu')[0].numpy()))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--emb', required=True); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    z = np.load(a.emb); S, path, fr = z['set'], z['path'], z['frame']
    jobs, rows = [], []
    for pth in dict.fromkeys(path.tolist()):                                  # one job per file, in embedding order of first appearance
        ix = np.where(path == pth)[0]
        jobs.append((pth, fr[ix].tolist(), 'corpus' if S[ix[0]] == 'corpus' else 'val')); rows.append(ix)
    print(len(jobs), 'files', flush=True)
    res = np.full((len(S), len(NAMES)), np.nan, np.float32)
    vi = [i for i, j in enumerate(jobs) if j[2] != 'corpus']                    # the labelled sets are read in this process (their reader starts its own pool)
    for i in vi:
        for r, o in zip(rows[i], work(jobs[i])):
            res[r] = [o[n] for n in NAMES]
    ci = [i for i, j in enumerate(jobs) if j[2] == 'corpus']
    with Pool(a.workers) as pool:
        for k, (ix, out) in enumerate(zip([rows[i] for i in ci], pool.imap(work, [jobs[i] for i in ci], chunksize=8))):
            for r, o in zip(ix, out):
                res[r] = [o[n] for n in NAMES]
            if k % 500 == 0:
                print(f'{k}/{len(jobs)} files', flush=True)
    np.savez(a.out, stats=res, names=np.array(NAMES), set=S, path=path, frame=fr)
    print('saved', a.out, res.shape, 'nan rows', int(np.isnan(res).any(1).sum()))
