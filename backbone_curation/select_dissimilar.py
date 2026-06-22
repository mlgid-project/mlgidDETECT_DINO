"""
Content-driven (redundancy-aware) frame selection, as an alternative to the
count-driven even-spacing in finalize_manifest.py.

Idea: for each scan, walk frames in time order and keep a frame ONLY if it has
changed enough from the LAST KEPT frame: phash-Hamming(frame, last_kept) > T.
=> static scans contribute few frames; evolving scans contribute many; the kept
set is non-redundant by construction (every consecutive kept pair differs by > T).
Comparing to last-KEPT (not last-seen) lets slow drift accumulate until it matters.

Usage:
  python select_dissimilar.py                 # sweep T, print count<->redundancy tradeoff
  python select_dissimilar.py --T 2 --write    # write a manifest for threshold T (does NOT build a corpus)

--write emits, to $CUR, with DISTINCT names so the live 13k manifest is never touched:
    manifest_dissim_T{T}.npz   (same schema as manifest.npz: key/scan/frame_idx/keep/reason)
    manifest_dissim_T{T}.tsv
    keep_keys_dissim_T{T}.txt
To build a corpus from it LATER, point corpus_builder.py at the new .npz (it currently
hardcodes manifest.npz) — not done here on purpose; we keep using the current 13k corpus.
"""
import os, glob, argparse
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

seen = set(); exactdup = np.zeros(N, bool)
for i in range(N):
    m = md5_by.get(str(keys[i]))
    if m in seen: exactdup[i] = True
    elif m is not None: seen.add(m)
blank = np.array([blank_by.get(str(keys[i]), False) for i in range(N)])
lab6 = np.array(['lab6' in scans[i].lower() for i in range(N)])
avail = ~(exactdup | blank | lab6)

by = defaultdict(list)
for i in range(N):
    if avail[i]: by[scans[i]].append(i)
for s in by: by[s].sort(key=lambda i: frames[i])
scan_sizes = {s: len(idx) for s, idx in by.items()}
print(f"available {int(avail.sum())} frames across {len(by)} scans\n")


def ham(a, b): return int(a ^ b).bit_count()


def select_T(T):
    """keep frame if phash-Hamming(frame, last_kept) > T, per scan in time order."""
    keep = np.zeros(N, bool)
    for s, idx in by.items():
        last = None
        for i in idx:
            if last is None or ham(phash[i], phash[last]) > T:
                keep[i] = True; last = i
    return keep


def redundancy(keep):
    d = []
    for s, idx in by.items():
        kept = [i for i in idx if keep[i]]
        for a, b in zip(kept, kept[1:]):
            d.append(ham(phash[a], phash[b]))
    d = np.array(d) if d else np.array([99])
    return d


def write_manifest(T):
    """Write a content-driven manifest for threshold T (distinct filenames; never
    overwrites manifest.npz/keep_keys.txt that the live 13k corpus uses)."""
    keep = select_T(T)
    # reason per frame, parallel to the available/excluded split
    reason = np.empty(N, dtype=object)
    for i in range(N):
        if exactdup[i]:        reason[i] = 'exact_dup'
        elif blank[i]:         reason[i] = 'blank'
        elif lab6[i]:          reason[i] = 'calibrant_lab6'
        elif keep[i]:          reason[i] = 'keep'
        else:                  reason[i] = f'redundant_within_T{T}'
    npz = os.path.join(CUR, f"manifest_dissim_T{T}.npz")
    tsv = os.path.join(CUR, f"manifest_dissim_T{T}.tsv")
    kk  = os.path.join(CUR, f"keep_keys_dissim_T{T}.txt")
    np.savez_compressed(npz, key=keys, scan=scans, frame_idx=frames, keep=keep, reason=reason)
    with open(tsv, "w") as f:
        f.write("keep\treason\tscan\tframe\tkey\n")
        for i in range(N):
            f.write(f"{int(keep[i])}\t{reason[i]}\t{scans[i]}\t{frames[i]}\t{keys[i]}\n")
    with open(kk, "w") as f:
        for i in range(N):
            if keep[i]:
                f.write(str(keys[i]) + "\n")
    d = redundancy(keep)
    sc = len(set(scans[i] for i in range(N) if keep[i]))
    print(f"\nthreshold T>{T}: kept {int(keep.sum())} frames across {sc} scans "
          f"(median consec dist {int(np.median(d))}, all pairs > {T} by construction)")
    print(f"wrote:\n  {npz}\n  {tsv}\n  {kk}")
    print("NOTE: no corpus built. To build later, point corpus_builder.py at "
          f"manifest_dissim_T{T}.npz (it currently hardcodes manifest.npz).")


def sweep():
    print(f"{'T(>)':>4} {'kept':>6} {'scans':>6} {'medD':>5} {'<=2':>6} {'<=4':>6} "
          f"{'p10D':>5}   note")
    for T in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        keep = select_T(T)
        n = int(keep.sum())
        sc = len(set(scans[i] for i in range(N) if keep[i]))
        d = redundancy(keep)
        note = "<- near 20k" if abs(n - 20000) < 1500 else ("<- near 13k" if abs(n-13000)<1200 else "")
        print(f"{T:>4} {n:>6} {sc:>6} {int(np.median(d)):>5} {float((d<=2).mean()):>6.2f} "
              f"{float((d<=4).mean()):>6.2f} {int(np.percentile(d,10)):>5}   {note}")
    print("\nfor reference, redundancy of the even-spacing sets (from finalize policy):")
    print("   13k even-spacing: ~0.83 within Ham<=2  |  20k even-spacing: ~0.86 within Ham<=2")
    print("   (content-driven sets above are bounded: every kept pair differs by > T)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=None,
                    help="keep frame if phash-Hamming to last-kept > T; omit to run the sweep")
    ap.add_argument("--write", action="store_true",
                    help="write manifest_dissim_T{T}.{npz,tsv} + keep_keys_dissim_T{T}.txt")
    a = ap.parse_args()
    if a.T is None:
        sweep()
    elif a.write:
        write_manifest(a.T)
    else:
        keep = select_T(a.T); d = redundancy(keep)
        sc = len(set(scans[i] for i in range(N) if keep[i]))
        print(f"T>{a.T}: kept {int(keep.sum())} across {sc} scans "
              f"(median consec dist {int(np.median(d))}). Add --write to save the manifest.")
