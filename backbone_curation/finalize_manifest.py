"""
Final keep-list targeting ~N_TARGET frames while PRESERVING within-scan evolution.

Policy (per scan, ordered by frame_idx):
  - small scans (<= C frames): keep all -> full evolution of short/angle scans
  - big scans (> C): keep C EVENLY-SPACED frames -> samples the start->end trajectory
  - then a light exact-dup trim: drop a selected frame if phash-Hamming <=1 to the
    previously kept one (collapses truly-static stretches without touching evolving ones)
The per-scan cap C is tuned by bisection so the net total ~= N_TARGET.
Mandatory excludes first: exact md5 dups, blank frames, LaB6 calibrant.
NON-destructive: writes manifest.npz / manifest.tsv.
"""
import os, sys, glob
import numpy as np
from collections import defaultdict

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
N_TARGET = 13000


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


def main():
    keys, scans, frames, md5_by, blank_by, ph_by = load()
    N = len(keys)
    phash = np.array([ph_by.get(str(keys[i]), 0) for i in range(N)], dtype=object)

    # mandatory excludes
    seen = set(); exactdup = np.zeros(N, bool)
    for i in range(N):
        m = md5_by.get(str(keys[i]))
        if m in seen:
            exactdup[i] = True
        elif m is not None:
            seen.add(m)
    blank = np.array([blank_by.get(str(keys[i]), False) for i in range(N)])
    lab6 = np.array(['lab6' in scans[i].lower() for i in range(N)])
    mand = exactdup | blank | lab6
    avail = ~mand
    print(f"corpus {N}; mandatory excludes: exact_dup {int(exactdup.sum())}, "
          f"blank {int(blank.sum())}, lab6 {int(lab6.sum())} -> available {int(avail.sum())}")

    by = defaultdict(list)
    for i in range(N):
        if avail[i]:
            by[scans[i]].append(i)
    for s in by:
        by[s].sort(key=lambda i: frames[i])

    def select(C):
        keep = np.zeros(N, bool)
        for s, idx in by.items():
            cand = idx if len(idx) <= C else [idx[j] for j in np.linspace(0, len(idx) - 1, C).astype(int)]
            last = None
            for i in cand:
                if last is None or (int(phash[i]) ^ int(phash[last])).bit_count() > 1:
                    keep[i] = True; last = i
        return keep

    # bisection on C to hit N_TARGET
    lo, hi = 1, 2000
    best = None
    for _ in range(18):
        C = (lo + hi) // 2
        k = select(C); n = int(k.sum())
        if abs(n - N_TARGET) < (best[1] if best else 1e9):
            best = (C, abs(n - N_TARGET), n, k)
        if n < N_TARGET:
            lo = C + 1
        else:
            hi = C - 1
        if lo > hi:
            break
    C, _, n, keep = best
    print(f"\nchosen per-scan cap C={C} -> {n} frames kept ({100*n/N:.1f}% of corpus)")

    # report
    from collections import Counter
    cnt = Counter(scans.tolist())
    print("\nbig-scan reduction (raw -> kept):")
    for s, c in cnt.most_common(14):
        kk = int(keep[[i for i in range(N) if scans[i] == s]].sum())
        print(f"   {c:6d} -> {kk:4d}   {s}")
    kept_scans = len(set(scans[i] for i in range(N) if keep[i]))
    persc = Counter(scans[i] for i in range(N) if keep[i])
    ks = np.array(sorted(persc.values()))
    print(f"\nscans represented: {kept_scans}/{len(cnt)};  kept-per-scan: "
          f"min {ks.min()} median {int(np.median(ks))} max {ks.max()}")

    # manifest
    reason = np.array(['keep'] * N, dtype=object)
    reason[~keep & ~mand] = 'within_scan_subsampled'
    reason[exactdup] = 'exact_dup'; reason[blank & ~exactdup] = 'blank'
    reason[lab6 & ~exactdup & ~blank] = 'calibrant_lab6'
    np.savez_compressed(os.path.join(CUR, "manifest.npz"),
                        key=keys, scan=scans, frame_idx=frames, keep=keep, reason=reason)
    with open(os.path.join(CUR, "manifest.tsv"), "w") as f:
        f.write("keep\treason\tscan\tframe\tkey\n")
        for i in range(N):
            f.write(f"{int(keep[i])}\t{reason[i]}\t{scans[i]}\t{frames[i]}\t{keys[i]}\n")
    with open(os.path.join(CUR, "keep_keys.txt"), "w") as f:
        for i in range(N):
            if keep[i]:
                f.write(str(keys[i]) + "\n")
    print(f"\nwrote manifest.npz, manifest.tsv, keep_keys.txt  (FINAL KEEP = {int(keep.sum())})")


if __name__ == "__main__":
    main()
