"""
Build the SSL backbone corpus HDF5 from the keep-list, storing images UNPROCESSED.

We store the moneta 8-bit polar frame exactly as it sits in the tar (512x1024 uint8,
0 = no-data) — NO flip, NO contrast, NO normalization baked in — so the preprocessing
can be changed later. The detector-matching transform is applied in the dataloader
(see backbone_transform.py), not here.

Output: one chunked, gzip-compressed HDF5 (Lustre-friendly), frames ordered by
(scan, frame_idx) so each scan is contiguous, with side datasets for scan-aware splits.
"""
import os, sys, glob, tarfile
import numpy as np
import cv2
import h5py
from collections import defaultdict

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
TARDIR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE"
OUT = os.path.join(CUR, "backbone_ssl_corpus.h5")
H, W = 512, 1024


def main():
    # key -> tar
    key2tar = {}
    for p in sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz"))):
        d = np.load(p, allow_pickle=True)
        for k, t in zip(d['key'], d['tar']):
            key2tar[str(k)] = str(t)
    # kept rows from manifest, ordered by (scan, frame_idx)
    m = np.load(os.path.join(CUR, "manifest.npz"), allow_pickle=True)
    m_scan, m_fr, m_key, m_keep = m['scan'], m['frame_idx'], m['key'], m['keep']  # materialize once
    rows = [(str(m_scan[i]), int(m_fr[i]), str(m_key[i]))
            for i in range(len(m_key)) if m_keep[i]]
    rows.sort(key=lambda r: (r[0], r[1]))
    N = len(rows)
    print(f"building corpus: {N} frames -> {OUT}")

    # group output indices by tar for single-pass extraction
    by_tar = defaultdict(list)
    for out_idx, (scan, fr, key) in enumerate(rows):
        by_tar[key2tar[key]].append((out_idx, key))

    str_dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(OUT, "w") as h:
        dimg = h.create_dataset("images", shape=(N, H, W), dtype=np.uint8,
                                chunks=(1, H, W), compression="gzip", compression_opts=4)
        dscan = h.create_dataset("scan_id", shape=(N,), dtype=str_dt)
        dbeam = h.create_dataset("beamtime", shape=(N,), dtype=str_dt)
        dfr = h.create_dataset("frame_idx", shape=(N,), dtype=np.int32)
        dkey = h.create_dataset("key", shape=(N,), dtype=str_dt)
        # metadata
        h.attrs["description"] = ("Unlabeled GIWAXS polar frames for SSL backbone pretraining. "
                                  "Stored UNPROCESSED (moneta 8-bit polar as in source tars, "
                                  "0=no-data). Apply detector-matching transform in the dataloader.")
        h.attrs["representation"] = "moneta polar (q x chi), 512(chi) x 1024(q), uint8, 0=no-data"
        h.attrs["n_frames"] = N
        h.attrs["source"] = "DINO_BACKBONE tar shards; keep_keys.txt manifest"

        scans = np.array([r[0] for r in rows])
        frames = np.array([r[1] for r in rows], np.int32)
        keys = np.array([r[2] for r in rows])
        dscan[:] = scans
        dbeam[:] = np.array([s.split('/')[0] for s in scans])
        dfr[:] = frames
        dkey[:] = keys

        done = 0
        for tarb, items in sorted(by_tar.items()):
            with tarfile.open(os.path.join(TARDIR, tarb + ".tar")) as tar:
                mem = {x.name: x for x in tar.getmembers()}
                for out_idx, key in items:
                    b = tar.extractfile(mem[key]).read()
                    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
                    if im.ndim == 3:
                        im = im[..., 0]
                    if im.shape != (H, W):
                        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                    dimg[out_idx] = im
                    done += 1
            print(f"  {done:6d}/{N}  ({tarb})", flush=True)
    sz = os.path.getsize(OUT) / 1e9
    print(f"\nDONE: {OUT}  ({sz:.2f} GB)")
    # quick verify
    with h5py.File(OUT, "r") as h:
        print(f"  images {h['images'].shape} {h['images'].dtype}; "
              f"scans={len(set(h['scan_id'][:].astype(str)))}; "
              f"sample nonzero frac={float((h['images'][0]!=0).mean()):.3f}")


if __name__ == "__main__":
    main()
