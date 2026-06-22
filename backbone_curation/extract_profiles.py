"""
Second pass: proper full-resolution profiles per shard frame, for convention-invariant
leak matching and better dedup.
  col_prof (1024): chi-averaged I(q) over nonzero pixels per q-column  -> I(q), q in [0,q_max]
  row_prof (512) : q-averaged   I(chi) over nonzero pixels per chi-row -> I(chi)
Stored with key/scan/frame_idx/q_max so it joins to shard_parts by key.
"""
import os, sys, io, json, re, time, tarfile
import numpy as np
import cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_shards import parse_scan, TARDIR

OUT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/profile_parts"
os.makedirs(OUT, exist_ok=True)


def col_row_profiles(img):
    a = img.astype(np.float32)
    a = np.log1p(a)                                  # log domain (shard PNG already compressed, mild)
    m = (img != 0)
    csum = a.sum(axis=0); ccnt = m.sum(axis=0)
    rsum = a.sum(axis=1); rcnt = m.sum(axis=1)
    col = np.divide(csum, ccnt, out=np.zeros_like(csum), where=ccnt > 0)
    row = np.divide(rsum, rcnt, out=np.zeros_like(rsum), where=rcnt > 0)
    return col.astype(np.float16), row.astype(np.float16)


def process_tar(tarpath):
    t0 = time.time(); tarname = os.path.basename(tarpath)[:-4]
    keys, scans, fidxs, qmaxs, cols, rows = [], [], [], [], [], []
    try:
        with tarfile.open(tarpath, "r") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            for n in [x for x in members if x.endswith('.png')]:
                try:
                    b = tar.extractfile(members[n]).read()
                    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = img[..., 0]
                    jn = n[:-4] + '.json'; source = ''; qmax = np.nan
                    if jn in members:
                        try:
                            j = json.loads(tar.extractfile(members[jn]).read())
                            source = j.get('source', ''); qmax = float(j.get('q_max', np.nan))
                        except Exception:
                            pass
                    scan, fidx = parse_scan(source, n)
                    c, r = col_row_profiles(img)
                    keys.append(n); scans.append(scan); fidxs.append(fidx); qmaxs.append(qmax)
                    cols.append(c); rows.append(r)
                except Exception:
                    continue
    except Exception as e:
        return (tarname, 0, f"ERR {e}", time.time() - t0)
    if keys:
        np.savez_compressed(os.path.join(OUT, tarname + ".npz"),
                            key=np.array(keys), scan=np.array(scans),
                            frame_idx=np.array(fidxs, np.int32), q_max=np.array(qmaxs, np.float32),
                            col=np.stack(cols), row=np.stack(rows))
    return (tarname, len(keys), "ok", time.time() - t0)


if __name__ == "__main__":
    tars = sorted(os.path.join(TARDIR, f) for f in os.listdir(TARDIR) if f.endswith('.tar'))
    print(f"profiles: {len(tars)} tars / {min(len(tars),12)} workers", flush=True)
    with Pool(min(len(tars), 12)) as p:
        for name, n, st, dt in p.imap_unordered(process_tar, tars):
            print(f"  [{st:>8}] {n:6d}  {dt:6.1f}s  {name}", flush=True)
    print("DONE", flush=True)
