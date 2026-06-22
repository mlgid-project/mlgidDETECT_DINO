"""
Evaluate the keep-target: how much *new* information vs redundancy you get at
different N_TARGET values. Reuses finalize_manifest's exact selection policy
(per-scan even-spacing cap C + phash<=1 static trim) and adds redundancy metrics:

 - C and actual kept count per target
 - scans represented; how many scans are SATURATED (all frames kept) vs capped
 - within-scan redundancy: phash-Hamming between *consecutive kept* frames.
   low distance = near-duplicate. We report the fraction of kept frames whose
   adjacent kept neighbor is within Hamming T (T=1,2,4,8).
 - MARGINAL analysis: the frames added going from a lower target to a higher one,
   and how redundant THOSE specific frames are (the ones whose value is in question).
"""
import os, glob
import numpy as np
from collections import defaultdict, Counter

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"


def load():
    keys, scans, frames = [], [], []
    for p in sorted(glob.glob(os.path.join(CUR, "profile_parts", "*.npz"))):
        d = np.load(p, allow_pickle=True)
        keys.append(d['key']); scans.append(d['scan']); frames.append(d['frame_idx'])
    keys = np.concatenate(keys); scans = np.concatenate(scans); frames = np.concatenate(frames)
    md5_by, blank_by, ph_by = {}, {}, {}
    for p in sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz"))):
        d = np.load(p, allow_pickle=True)
        for k, m, b, h in zip(d['key'], d['md5'], d['blank'], d['phash']):
            md5_by[str(k)] = str(m); blank_by[str(k)] = bool(b); ph_by[str(k)] = int(h)
    return keys, scans, frames, md5_by, blank_by, ph_by


keys, scans, frames, md5_by, blank_by, ph_by = load()
N = len(keys)
phash = np.array([ph_by.get(str(keys[i]), 0) for i in range(N)], dtype=np.uint64)

# mandatory excludes (same as finalize_manifest)
seen = set(); exactdup = np.zeros(N, bool)
for i in range(N):
    m = md5_by.get(str(keys[i]))
    if m in seen:
        exactdup[i] = True
    elif m is not None:
        seen.add(m)
blank = np.array([blank_by.get(str(keys[i]), False) for i in range(N)])
lab6 = np.array(['lab6' in scans[i].lower() for i in range(N)])
avail = ~(exactdup | blank | lab6)
print(f"corpus {N}; mandatory excludes -> available {int(avail.sum())} "
      f"(exact_dup {int(exactdup.sum())}, blank {int(blank.sum())}, lab6 {int(lab6.sum())})")

by = defaultdict(list)
for i in range(N):
    if avail[i]:
        by[scans[i]].append(i)
for s in by:
    by[s].sort(key=lambda i: frames[i])
scan_sizes = {s: len(idx) for s, idx in by.items()}


def ham(a, b):
    return int(a ^ b).bit_count()


def select(C):
    keep = np.zeros(N, bool)
    for s, idx in by.items():
        cand = idx if len(idx) <= C else [idx[j] for j in np.linspace(0, len(idx) - 1, C).astype(int)]
        last = None
        for i in cand:
            if last is None or ham(phash[i], phash[last]) > 1:
                keep[i] = True; last = i
    return keep


def C_for_target(target):
    lo, hi, best = 1, 4000, None
    for _ in range(20):
        C = (lo + hi) // 2
        n = int(select(C).sum())
        if best is None or abs(n - target) < best[1]:
            best = (C, abs(n - target), n)
        if n < target: lo = C + 1
        else: hi = C - 1
        if lo > hi: break
    return best[0]


def redundancy(keep):
    """fraction of kept frames whose adjacent kept neighbor (same scan, by frame order)
    is within Hamming T. Also median consecutive-kept distance."""
    dists = []
    for s, idx in by.items():
        kept = [i for i in idx if keep[i]]
        for a, b in zip(kept, kept[1:]):
            dists.append(ham(phash[a], phash[b]))
    dists = np.array(dists) if dists else np.array([99])
    fr = {T: float((dists <= T).mean()) for T in (1, 2, 4, 8)}
    return dists, fr


# absolute ceiling: keep everything available (minus <=1 trim)
ceil_keep = select(10 ** 9)
print(f"absolute pool (all available, <=1 static-trim only): {int(ceil_keep.sum())}\n")

targets = [10000, 13000, 16000, 20000, 25000, int(ceil_keep.sum())]
print(f"{'target':>7} {'C':>5} {'kept':>6} {'scans':>6} {'satur':>6} "
      f"{'medD':>5} {'<=1':>6} {'<=2':>6} {'<=4':>6} {'<=8':>6}")
results = {}
for t in targets:
    C = C_for_target(t)
    keep = select(C)
    results[t] = keep
    n = int(keep.sum())
    sc = set(scans[i] for i in range(N) if keep[i])
    satur = sum(1 for s in sc if int(keep[[i for i in by[s]]].sum()) == scan_sizes[s])
    dists, fr = redundancy(keep)
    print(f"{t:>7} {C:>5} {n:>6} {len(sc):>6} {satur:>6} "
          f"{int(np.median(dists)):>5} {fr[1]:>6.2f} {fr[2]:>6.2f} {fr[4]:>6.2f} {fr[8]:>6.2f}")

# marginal: frames added 13k -> 20k, how redundant are THEY
k13, k20 = results[13000], results[20000]
added = k20 & ~k13
print(f"\n--- marginal frames added 13k -> 20k: {int(added.sum())} ---")
# redundancy of added frames vs their nearest KEPT-at-20k same-scan neighbor by frame order
adddist = []
for s, idx in by.items():
    kept20 = [i for i in idx if k20[i]]
    pos = {i: j for j, i in enumerate(kept20)}
    for i in kept20:
        if added[i]:
            j = pos[i]
            nb = []
            if j > 0: nb.append(ham(phash[i], phash[kept20[j - 1]]))
            if j < len(kept20) - 1: nb.append(ham(phash[i], phash[kept20[j + 1]]))
            if nb: adddist.append(min(nb))
adddist = np.array(adddist) if adddist else np.array([99])
print(f"added-frame nearest-kept-neighbor phash dist: "
      f"median {int(np.median(adddist))}, "
      f"<=1 {float((adddist<=1).mean()):.2f}  <=2 {float((adddist<=2).mean()):.2f}  "
      f"<=4 {float((adddist<=4).mean()):.2f}  <=8 {float((adddist<=8).mean()):.2f}")

# which scans absorb the 13k->20k growth
growth = Counter()
for i in range(N):
    if added[i]: growth[scans[i]] += 1
print("\ntop scans absorbing the +7k (added | their total available):")
for s, g in growth.most_common(12):
    print(f"   +{g:5d}   (scan has {scan_sizes[s]:5d} avail)   {s}")
