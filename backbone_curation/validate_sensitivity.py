"""
Sanity / sensitivity controls for the leak matcher. If the descriptor cannot match a
frame to a *different rendering of itself*, a low eval->shard score is meaningless.

(A) Positive control, cross-pipeline: for each 41.h5 frame that stores BOTH `image`
    (reciprocal) and `polar_image` (their own polar), compare MY cake(image) descriptor
    to THEIR polar_image descriptor. Same physical frame, two pipelines -> must score high.
(B) Positive control, near-dup: within shards, cosine of .raw.png vs .png of the same
    frame, and of consecutive in-situ frames -> must be high.
(C) Provenance cross-check: does each eval frame's source scan-token (from filename/
    folder) appear among shard scan_ids / sources? Catches same-provenance under any name.
"""
import os, sys, glob, re
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_common as fp
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
DCHI, DQ = 32, 64


def cos_flip(a, b):
    """flip-invariant cosine between two 2048-d z-scored descriptors."""
    g = b.reshape(DCHI, DQ)
    variants = [g, g[::-1], g[:, ::-1], g[::-1, ::-1]]
    a = a / (np.linalg.norm(a) + 1e-9)
    best = -1
    for v in variants:
        v = v.reshape(-1); v = v / (np.linalg.norm(v) + 1e-9)
        best = max(best, float(a @ v))
    return best


def control_A():
    print("=== (A) cross-pipeline positive control: my cake(image) vs their polar_image ===")
    path = "/mnt/lustre/work/schreiber/szb389/datasets/41.h5"
    scores = []
    with h5py.File(path, "r") as f:
        for fac in f.keys():
            for samp in f[fac].keys():
                g = f[fac][samp]
                if "image" not in g or "polar_image" not in g:
                    continue
                if not isinstance(g["polar_image"], h5py.Dataset) or g["polar_image"].ndim != 2:
                    continue
                recip = np.asarray(g["image"]); pol = np.asarray(g["polar_image"])
                d_cake = fp.descriptor(fp.cake_reciprocal(recip))
                d_their = fp.descriptor(pol)
                c = cos_flip(d_cake, d_their)
                scores.append(c)
                if len(scores) <= 8:
                    print(f"   {fac}/{samp[:34]:34s} polar={pol.shape}  self-cos={c:.3f}")
    scores = np.array(scores)
    print(f"   N={len(scores)}  self-cos: min {scores.min():.3f}  median {np.median(scores):.3f}  max {scores.max():.3f}")
    print(f"   --> descriptor {'IS' if np.median(scores)>0.8 else 'IS NOT'} robust to pipeline differences")
    return scores


def control_B():
    print("\n=== (B) near-dup positive control inside shards ===")
    parts = sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz")))
    # load a couple of useful parts
    def load(name):
        d = np.load([p for p in parts if name in p][0], allow_pickle=True)
        return d
    # raw vs non-raw same frame (2019_11 lab6)
    d = load("raw_2019_11")
    keys = d["key"]; fr = d["frame_idx"]; israw = d["is_raw"]; desc = d["desc"].astype(np.float32)
    pairs = []
    for i in range(len(keys)):
        if israw[i]:
            # find non-raw with same frame_idx
            for j in range(len(keys)):
                if not israw[j] and fr[j] == fr[i]:
                    pairs.append((i, j)); break
    cs = [cos_flip(desc[i], desc[j]) for i, j in pairs[:20]]
    if cs:
        print(f"   .raw.png vs .png same frame: N={len(cs)} cos median {np.median(cs):.3f} min {min(cs):.3f}")
    # consecutive in-situ frames (existing-000000, a big scan)
    d = load("existing-000000")
    scan = d["scan"]; fr = d["frame_idx"]; desc = d["desc"].astype(np.float32)
    # pick the most frequent scan
    from collections import Counter
    top_scan = Counter(scan.tolist()).most_common(1)[0][0]
    idx = np.where(scan == top_scan)[0]
    idx = idx[np.argsort(fr[idx])]
    cons = [cos_flip(desc[idx[k]], desc[idx[k+1]]) for k in range(min(20, len(idx)-1))]
    print(f"   consecutive frames of '{top_scan[:40]}' (n={len(idx)}): cos median {np.median(cons):.3f}")
    # far-apart frames in same scan
    if len(idx) > 40:
        far = [cos_flip(desc[idx[k]], desc[idx[k+30]]) for k in range(min(20, len(idx)-30))]
        print(f"   frames 30 apart same scan: cos median {np.median(far):.3f}")


def control_C():
    print("\n=== (C) provenance cross-check: eval source tokens vs shard scans/sources ===")
    # gather shard scan ids + sources
    parts = sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz")))
    shard_scans = set(); shard_src_blob = []
    for p in parts:
        d = np.load(p, allow_pickle=True)
        shard_scans.update(s.lower() for s in d["scan"].tolist())
        shard_src_blob.append("\n".join(s.lower() for s in d["source"].tolist()))
    shard_src_blob = "\n".join(shard_src_blob)
    # eval tokens
    ev = np.load(os.path.join(CUR, "eval_fp.npz"), allow_pickle=True)
    tok_re = re.compile(r'([a-z]+[a-z0-9_]*?_\d{4,6})', re.I)         # e.g. timo2_mapbi3..., ekaterina_June_2020_00510
    hits = []
    for i in range(len(ev["eid"])):
        meta = str(ev["meta"][i]).lower()
        # extract candidate scan tokens from filename/folder
        cands = set()
        for m in re.findall(r'([a-z0-9][a-z0-9_]+?_\d{3,6})', meta):
            cands.add(m)
        # also raw filename stems like '..._00510-00740'
        for m in re.findall(r'([a-z0-9_]+?_\d{4,6})-\d+', meta):
            cands.add(m)
        matched = []
        for c in cands:
            if len(c) < 6:
                continue
            if any(c in s for s in shard_scans) or c in shard_src_blob:
                matched.append(c)
        if matched:
            hits.append((str(ev["eid"][i]), matched))
    if hits:
        print("   PROVENANCE OVERLAPS FOUND:")
        for eid, m in hits:
            print(f"     {eid[:55]:55s} <- tokens {m}")
    else:
        print("   no eval source token appears among shard scan-ids/sources")
    # also list eval tokens for manual eyeballing
    print("\n   (eval source tokens extracted, for reference:)")
    for i in range(len(ev["eid"])):
        meta = str(ev["meta"][i]).lower()
        cands = set(re.findall(r'([a-z0-9_]+?_\d{4,6})(?:-\d+)?', meta))
        cands = {c for c in cands if len(c) >= 6}
        if cands:
            print(f"     {str(ev['eid'][i])[:50]:50s} {sorted(cands)}")


if __name__ == "__main__":
    control_A()
    control_B()
    control_C()
