"""
Redundancy analysis + within-scan change-based decimation for the backbone corpus.

Similarity = min(cos[I(q)], cos[I(chi)]) on the per-frame profiles (same convention within
a scan, so no flips needed). Within each scan, frames are ordered by frame_idx and decimated
greedily: keep a frame only if it differs enough from the last KEPT frame (sim < threshold).
This collapses near-stationary stretches of in-situ scans while keeping the transitions.

Also: exact-duplicate detection across the whole corpus via md5, quality excludes (blank),
and a LaB6-calibrant flag. Writes a keep/discard manifest + a report. NON-destructive.
"""
import os, sys, glob
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iq_match as IQ
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"


def znorm(v):
    v = np.asarray(v, np.float32)
    m = v != 0
    out = np.zeros_like(v)
    if m.sum() >= 8:
        mu, sd = v[m].mean(), v[m].std()
        if sd > 1e-9:
            out[m] = (v[m] - mu) / sd
    n = np.linalg.norm(out)
    return out / n if n > 1e-9 else out


def load_all():
    # profiles -> I(q), I(chi)
    pp = sorted(glob.glob(os.path.join(CUR, "profile_parts", "*.npz")))
    keys, scans, frames, IQs, IC = [], [], [], [], []
    for p in pp:
        d = np.load(p, allow_pickle=True)
        keys.append(d['key']); scans.append(d['scan']); frames.append(d['frame_idx'])
        col = d['col'].astype(np.float32)
        IQs.append(np.stack([IQ.shard_iq_from_col(col[i], float(d['q_max'][i])) for i in range(len(col))]))
        IC.append(d['row'].astype(np.float32))
    keys = np.concatenate(keys); scans = np.concatenate(scans); frames = np.concatenate(frames)
    IQs = np.concatenate(IQs); IC = np.concatenate(IC)
    # shard_parts -> md5, blank, is_raw, phash (join by key)
    md5_by, blank_by, raw_by, ph_by = {}, {}, {}, {}
    for p in sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz"))):
        d = np.load(p, allow_pickle=True)
        for k, m, b, r, h in zip(d['key'], d['md5'], d['blank'], d['is_raw'], d['phash']):
            md5_by[str(k)] = str(m); blank_by[str(k)] = bool(b)
            raw_by[str(k)] = bool(r); ph_by[str(k)] = int(h)
    return keys, scans, frames, IQs, IC, md5_by, blank_by, raw_by, ph_by


def main():
    keys, scans, frames, IQs, IC, md5_by, blank_by, raw_by, ph_by = load_all()
    N = len(keys)
    print(f"corpus: {N} frames")
    Zq = np.stack([znorm(IQs[i]) for i in range(N)])
    Zc = np.stack([znorm(IC[i]) for i in range(N)])
    phash = np.array([ph_by.get(str(keys[i]), 0) for i in range(N)], dtype=np.uint64)

    def popcount(x):
        x = np.asarray(x, np.uint64); c = np.zeros(x.shape, np.int32)
        for i in range(64):
            c += ((x >> np.uint64(i)) & np.uint64(1)).astype(np.int32)
        return c

    # ---- scan-size distribution ----
    cnt = Counter(scans.tolist())
    sizes = np.array(sorted(cnt.values(), reverse=True))
    print(f"\nscans: {len(cnt)}  | frames per scan: max {sizes.max()} median {int(np.median(sizes))}")
    print("  size buckets:", {f"{lo}-{hi}": int(((sizes >= lo) & (sizes <= hi)).sum())
                               for lo, hi in [(1,1),(2,9),(10,49),(50,199),(200,999),(1000,99999)]})
    print("  biggest scans:")
    for s, c in cnt.most_common(10):
        print(f"     {c:6d}  {s}")

    # group indices by scan, ordered by frame
    by_scan = defaultdict(list)
    for i in range(N):
        by_scan[scans[i]].append(i)
    for s in by_scan:
        by_scan[s] = sorted(by_scan[s], key=lambda i: frames[i])

    def decimate(H):
        """greedy within-scan: keep frame if phash-Hamming to last KEPT frame > H
        (i.e. perceptually changed). Drops near-identical stationary runs."""
        keep = np.zeros(N, bool)
        for s, idx in by_scan.items():
            last = None
            for i in idx:
                if last is None:
                    keep[i] = True; last = i; continue
                if popcount(np.uint64(int(phash[i]) ^ int(phash[last]))) > H:
                    keep[i] = True; last = i
        return keep

    print("\n[Hamming sweep] within-scan change-based decimation (phash):")
    print(f"  {'Ham>':>5} {'kept':>7} {'dropped':>8} {'kept%':>6}")
    for H in [1, 2, 4, 6, 8]:
        k = decimate(H)
        print(f"  {H:5d} {int(k.sum()):7d} {int((~k).sum()):8d} {100*k.mean():6.1f}")

    # ---- choose operating threshold ----
    TH = 4
    keep = decimate(TH)

    # ---- exact-duplicate (md5) across corpus: keep first occurrence ----
    seen = set(); exactdup = np.zeros(N, bool)
    for i in range(N):
        m = md5_by.get(str(keys[i]))
        if m is None:
            continue
        if m in seen:
            exactdup[i] = True
        else:
            seen.add(m)
    # ---- quality / calibrant flags ----
    blank = np.array([blank_by.get(str(keys[i]), False) for i in range(N)])
    lab6 = np.array(['lab6' in scans[i].lower() or 'lab6' in str(keys[i]).lower() for i in range(N)])

    final_keep = keep & ~exactdup & ~blank & ~lab6
    print(f"\n[manifest @ thresh={TH}]")
    print(f"  within-scan kept     : {int(keep.sum())}")
    print(f"  minus exact md5 dups : -{int((keep & exactdup).sum())}")
    print(f"  minus blank frames   : -{int((keep & ~exactdup & blank).sum())}")
    print(f"  minus LaB6 calibrant : -{int((keep & ~exactdup & ~blank & lab6).sum())}")
    print(f"  ==> FINAL KEEP       : {int(final_keep.sum())}  ({100*final_keep.mean():.1f}% of {N})")

    # per-scan kept summary for the big scans
    print("\n  big-scan reduction (frames -> kept):")
    for s, c in cnt.most_common(12):
        idx = by_scan[s]
        kk = int(final_keep[idx].sum())
        print(f"     {c:6d} -> {kk:4d}   {s}")

    # write manifest
    reason = np.array(['keep'] * N, dtype=object)
    reason[~keep] = 'within_scan_dup'
    reason[keep & exactdup] = 'exact_dup'
    reason[keep & ~exactdup & blank] = 'blank'
    reason[keep & ~exactdup & ~blank & lab6] = 'calibrant_lab6'
    np.savez_compressed(os.path.join(CUR, "manifest.npz"),
                        key=keys, scan=scans, frame_idx=frames,
                        keep=final_keep, reason=reason)
    with open(os.path.join(CUR, "manifest.tsv"), "w") as f:
        f.write("keep\treason\tscan\tframe\tkey\n")
        for i in range(N):
            f.write(f"{int(final_keep[i])}\t{reason[i]}\t{scans[i]}\t{frames[i]}\t{keys[i]}\n")
    print(f"\nwrote manifest.npz + manifest.tsv ({N} rows)")


if __name__ == "__main__":
    main()
