"""
Fingerprint every eval frame in 41.h5 (roi_data schema) and organic_labeled.h5 (pyGID
schema). Each frame's reciprocal q-map is caked to the shard polar convention, then
fingerprinted with the SAME canonical functions as the shards. Saves eval_fp.npz.

We also keep the caked 512x1024 polar (downsampled) for each eval frame so detect_leaks.py
can render side-by-side panels for visual confirmation.
"""
import os, sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_common as fp

EVAL = {"41": "/mnt/lustre/work/schreiber/szb389/datasets/41.h5",
        "organic": "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"}
OUT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"


def dec(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.ndarray):
        return dec(x.reshape(-1)[0]) if x.size else ""
    return str(x)


def iter_41(f):
    """roi_data: <facility>/<sample>/image (2D reciprocal)."""
    for fac in f.keys():
        for samp in f[fac].keys():
            g = f[fac][samp]
            if "image" in g and isinstance(g["image"], h5py.Dataset) and g["image"].ndim == 2:
                meta = {}
                if "metadata" in g:
                    for k in ["folder", "filename", "year", "energy", "angle_of_incidence", "facility"]:
                        if k in g["metadata"]:
                            try: meta[k] = dec(g["metadata"][k][()])
                            except Exception: pass
                yield f"41/{fac}/{samp}", np.asarray(g["image"]), meta


def iter_organic(f):
    """pyGID: entry_X/data/img_gid_q (N,H,W) stack."""
    for ent in f.keys():
        g = f[ent]
        if "data/img_gid_q" not in g:
            continue
        stack = g["data/img_gid_q"]
        meta0 = {}
        for path, key in [("instrument/angle_of_incidence", "angle_of_incidence"),
                          ("data/filename", "filename"), ("sample/name", "sample")]:
            if path in g:
                try: meta0[key] = dec(g[path][()])
                except Exception: pass
        n = stack.shape[0]
        for i in range(n):
            yield f"organic/{ent}/{i}", np.asarray(stack[i]), dict(meta0)


def main():
    ids, files, metas = [], [], []
    descs, phashes, thumbs = [], [], []
    for tag, path in EVAL.items():
        with h5py.File(path, "r") as f:
            it = iter_41(f) if tag == "41" else iter_organic(f)
            for eid, recip, meta in it:
                polar = fp.cake_reciprocal(recip)            # -> 512x1024 polar
                ids.append(eid); files.append(tag); metas.append(str(meta))
                descs.append(fp.descriptor(polar).astype(np.float16))
                phashes.append(np.uint64(fp.phash64(polar)))
                # small thumbnail (64x128 u8) for visual panels
                import cv2
                thumbs.append(cv2.resize(fp.to_u8(polar), (128, 64), interpolation=cv2.INTER_AREA))
                print(f"  {eid:55s} recip={recip.shape} meta={ {k:v[:40] for k,v in meta.items()} }", flush=True)
    np.savez_compressed(os.path.join(OUT, "eval_fp.npz"),
                        eid=np.array(ids), evalset=np.array(files), meta=np.array(metas),
                        desc=np.stack(descs), phash=np.array(phashes, np.uint64),
                        thumb=np.stack(thumbs))
    print(f"WROTE {len(ids)} eval fingerprints -> eval_fp.npz", flush=True)


if __name__ == "__main__":
    main()
