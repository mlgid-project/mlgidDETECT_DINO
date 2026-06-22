"""
Convention-invariant eval-leak detection via q-calibrated azimuthal profile I(q).
Averaging over chi removes the caking-convention dependence that broke the 2-D matcher;
calibrating q removes scale differences. Same physical frame -> matching I(q).

Validation first (must pass or the negative result is meaningless):
  V1 within-shard: I(q) correlation of same-scan frames (should be ~1) vs different scans.
  V2 cross-rep   : 41.h5 I(q) from `image` (per-pixel q) vs from `polar_image` (col-profile)
                   -- same frame, two representations -> should correlate highly.
Then: each eval frame's I(q) vs every shard I(q); report top matches + render panels.
"""
import os, sys, glob
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
QMAX, B = 6.0, 300
GRID = np.linspace(0, QMAX, B)


def zstd(v):
    v = np.asarray(v, np.float64)
    m = v != 0
    if m.sum() < 8:
        return None
    v = v.copy(); mu = v[m].mean(); sd = v[m].std()
    if sd < 1e-9:
        return None
    out = np.zeros_like(v); out[m] = (v[m] - mu) / sd
    return out, m


def corr(a, b):
    """Pearson over the shared support (both nonzero)."""
    ra, rb = zstd(a), zstd(b)
    if ra is None or rb is None:
        return 0.0
    (za, ma), (zb, mb) = ra, rb
    m = ma & mb
    if m.sum() < 8:
        return 0.0
    x, y = za[m], zb[m]
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def shard_iq_from_col(col, q_max):
    """col (1024,) I(q) over q in [0,q_max] -> resample to common GRID."""
    col = np.asarray(col, np.float64)
    if not np.isfinite(q_max) or q_max <= 0:
        q_max = 5.4                                  # P08 fallback for existing-* w/o q_max
    x = np.linspace(0, q_max, len(col))
    return np.interp(GRID, x, col, left=0, right=0)


def eval_iq_from_recip(img, qz, qx):
    """img (H,W) reciprocal; qz (H,), qx (W,) physical q axes -> I(q) on common GRID."""
    a = np.log1p(np.clip(np.nan_to_num(img), 0, None).astype(np.float64))
    m = (img != 0) & np.isfinite(img)
    Q = np.sqrt(qz[:, None] ** 2 + qx[None, :] ** 2)
    bins = np.clip((Q / QMAX * B).astype(int), 0, B - 1)
    num = np.bincount(bins[m].ravel(), weights=a[m].ravel(), minlength=B)
    cnt = np.bincount(bins[m].ravel(), minlength=B)
    return np.divide(num, cnt, out=np.zeros(B), where=cnt > 0)


def parse4(x):
    """Extract up to 4 floats from an array / bytes / '(0, 3.2, 0, 3.2)' string."""
    import re
    if isinstance(x, np.ndarray) and x.dtype.kind in 'fi' and x.size == 4:
        return x.astype(float).ravel()
    if isinstance(x, np.ndarray):
        x = x.reshape(-1)[0] if x.size else b''
    s = x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    return np.array([float(n) for n in nums[:4]]) if len(nums) >= 4 else None


def axes_for_eval(g, kind, img_shape):
    """Return (qz, qx) physical axes for an eval frame."""
    H, W = img_shape
    if kind == "organic":
        qx = np.asarray(g['data/q_xy']).astype(float)
        qz = np.asarray(g['data/q_z']).astype(float)
        if qx.size != W: qx = np.linspace(qx.min(), qx.max(), W)
        if qz.size != H: qz = np.linspace(qz.min(), qz.max(), H)
        return qz, qx
    # 41.h5: qz_qxy_range_[A-1] = [qz_min,qz_max,qxy_min,qxy_max] if present & nonzero
    rng = None
    if 'metadata' in g and 'qz_qxy_range_[A-1]' in g['metadata']:
        rng = parse4(g['metadata']['qz_qxy_range_[A-1]'][()])
    if rng is not None and rng.size == 4 and np.any(rng != 0):
        qz = np.linspace(rng[0], rng[1], H); qx = np.linspace(rng[2], rng[3], W)
    else:                                            # missing -> assume P08 axis max ~3.8 (diag~5.4)
        qz = np.linspace(0, 3.8, H); qx = np.linspace(0, 3.8, W)
    return qz, qx


def load_eval_iq():
    out = []
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/41.h5", "r") as f:
        for fac in f.keys():
            for samp in f[fac].keys():
                g = f[fac][samp]
                if 'image' not in g or g['image'].ndim != 2:
                    continue
                img = np.asarray(g['image']); qz, qx = axes_for_eval(g, "41", img.shape)
                out.append((f"41/{fac}/{samp}", eval_iq_from_recip(img, qz, qx)))
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5", "r") as f:
        for ent in f.keys():
            g = f[ent]
            if 'data/img_gid_q' not in g:
                continue
            st = g['data/img_gid_q']
            for i in range(st.shape[0]):
                img = np.asarray(st[i]); qz, qx = axes_for_eval(g, "organic", img.shape)
                out.append((f"organic/{ent}/{i}", eval_iq_from_recip(img, qz, qx)))
    return out


def load_shard_profiles():
    parts = sorted(glob.glob(os.path.join(CUR, "profile_parts", "*.npz")))
    keys, scans, frames, qmax, iqs = [], [], [], [], []
    for p in parts:
        d = np.load(p, allow_pickle=True)
        keys.append(d['key']); scans.append(d['scan']); frames.append(d['frame_idx'])
        qmax.append(d['q_max'])
        col = d['col'].astype(np.float32)
        iq = np.stack([shard_iq_from_col(col[i], float(d['q_max'][i])) for i in range(len(col))])
        iqs.append(iq.astype(np.float32))
    return (np.concatenate(keys), np.concatenate(scans), np.concatenate(frames),
            np.concatenate(qmax), np.concatenate(iqs))


def main():
    print("loading shard profiles...", flush=True)
    keys, scans, frames, qmax, S = load_shard_profiles()
    N = len(keys); print(f"  {N} shard I(q)", flush=True)

    # ---- V1 within-shard ----
    print("\n[V1] within-shard I(q) correlation:", flush=True)
    from collections import Counter
    cc = Counter(scans.tolist())
    # pick a CONTIGUOUS (non-decimated) scan: max-min+1 == count
    big = []
    for s, c in cc.items():
        if c >= 30:
            fi = frames[scans == s]
            if int(fi.max() - fi.min() + 1) == c:
                big = [s]; break
    if big:
        idx = np.where(scans == big[0])[0]; idx = idx[np.argsort(frames[idx])]
        print(f"   (contiguous scan: {big[0]}, n={len(idx)})", flush=True)
        same = np.median([corr(S[idx[k]], S[idx[k+1]]) for k in range(min(30, len(idx)-1))])
        rnd = np.random.RandomState(0).randint(0, N, 30)
        diff = np.median([corr(S[idx[0]], S[j]) for j in rnd])
        print(f"   same-scan median {same:.3f}  vs  random-pair median {diff:.3f}", flush=True)

    # ---- V2 cross-rep ----
    print("\n[V2] cross-rep (41.h5 image vs polar_image) I(q):", flush=True)
    cs = []
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/41.h5", "r") as f:
        for fac in f.keys():
            for samp in f[fac].keys():
                g = f[fac][samp]
                if 'image' not in g or 'polar_image' not in g:
                    continue
                if not isinstance(g['polar_image'], h5py.Dataset) or g['polar_image'].ndim != 2:
                    continue
                img = np.asarray(g['image']); qz, qx = axes_for_eval(g, "41", img.shape)
                iq_img = eval_iq_from_recip(img, qz, qx)
                pol = np.asarray(g['polar_image'])           # polar: assume cols=q over [0,q_max]
                # col-average over nonzero -> I(q_frac); compare on fractional grid
                colp = np.divide((np.log1p(np.clip(pol,0,None))*(pol!=0)).sum(0), (pol!=0).sum(0),
                                 out=np.zeros(pol.shape[1]), where=(pol!=0).sum(0)>0)
                iq_pol = shard_iq_from_col(colp, np.nan)      # fractional->[0,5.4] assumed
                cs.append(corr(iq_img, iq_pol))
    if cs:
        cs = np.array(cs)
        print(f"   N={len(cs)} median {np.median(cs):.3f} (>0.7 => I(q) cross-rep OK)", flush=True)

    # ---- eval -> shard search ----
    ev = load_eval_iq()
    print(f"\n[SEARCH] {len(ev)} eval frames vs {N} shard I(q):", flush=True)
    Sz = np.stack([ (lambda r: r[0] if r else np.zeros(B))(zstd(S[i])) for i in range(N)])
    Sn = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
    rows = []
    for eid, iq in ev:
        r = zstd(iq)
        if r is None:
            rows.append((eid, 0.0, "no-data", -1, [])); continue
        z = r[0]; z = z / (np.linalg.norm(z) + 1e-9)
        sim = Sn @ z
        cand = np.argsort(-sim)[:10]                  # fast top-10, then exact intersection-Pearson
        exact = [(int(j), corr(S[j], iq)) for j in cand]
        exact.sort(key=lambda t: -t[1])
        j, c = exact[0]
        top3 = [(str(scans[jj]), round(cc_, 3)) for jj, cc_ in exact[:3]]
        rows.append((eid, float(c), str(scans[j]), int(frames[j]), top3))
    rows.sort(key=lambda x: -x[1])
    print(f"\n{'corr':>6} {'eval id':<50} {'best shard scan':<44} frame", flush=True)
    for eid, c, sc, fr, top3 in rows:
        print(f"{c:6.3f} {eid[:50]:<50} {sc[:44]:<44} {fr}", flush=True)
    with open(os.path.join(CUR, "iq_leak.tsv"), "w") as fo:
        fo.write("corr\teid\tbest_scan\tframe\ttop3\n")
        for eid, c, sc, fr, top3 in rows:
            fo.write(f"{c:.3f}\t{eid}\t{sc}\t{fr}\t{top3}\n")


if __name__ == "__main__":
    main()
