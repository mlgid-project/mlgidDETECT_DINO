"""
Stream every .png member of every tar in DINO_BACKBONE, fingerprint it, and write
one npz part per tar. Parts are combined later. Read-only on the tars.

Per frame we store: key, tarname, scan_id, frame_idx, source, beamtime, q_max,
descriptor(2048 float16), phash(uint64), md5(hex), and quality stats.
"""
import os, sys, io, json, re, hashlib, time, tarfile
import numpy as np
import cv2
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_common as fp

TARDIR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE"
OUT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/shard_parts"
os.makedirs(OUT, exist_ok=True)

RAW_RE = re.compile(r'beamtimes/([^/]+)/(?:raw|Converted|converted)/(.+?)/pe/.*?-(\d+)(?:\.raw)?\.tif', re.I)
EXIST_RE = re.compile(r'tensor/([^/]+)/(.+)/[^/]+?_f?(\d+)\.(?:pt|png|tif)', re.I)
TRAILNUM = re.compile(r'(\d+)\D*$')


def parse_scan(source, member):
    """Return (scan_id, frame_idx). scan_id = '<beamtime>/<sample>'."""
    s = source or member
    m = RAW_RE.search(s)
    if m:
        return f"{m.group(1)}/{m.group(2)}", int(m.group(3))
    m = EXIST_RE.search(s)
    if m:
        return f"{m.group(1)}/{m.group(2)}", int(m.group(3))
    # fallback: use member-name tokens
    base = member.split('/')[-1]
    toks = base.replace('.raw', '').rsplit('.', 1)[0].split('__')
    fr = TRAILNUM.search(base)
    fidx = int(fr.group(1)) if fr else -1
    scan = '/'.join(toks[-3:-1]) if len(toks) >= 3 else base
    return scan, fidx


def process_tar(tarpath):
    t0 = time.time()
    tarname = os.path.basename(tarpath)[:-4]
    keys, scans, fidxs, srcs, beamts, qmaxs = [], [], [], [], [], []
    descs, phashes, md5s = [], [], []
    fvalid, sat, blankf, isder = [], [], [], []
    try:
        with tarfile.open(tarpath, "r") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            pngs = [n for n in members if n.endswith('.png')]
            for n in pngs:
                try:
                    b = tar.extractfile(members[n]).read()
                    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    if img.ndim == 3:
                        img = img[..., 0]
                    # sibling json
                    jn = n[:-4] + '.json'
                    source = ''; beamt = ''; qmax = np.nan
                    if jn in members:
                        try:
                            j = json.loads(tar.extractfile(members[jn]).read())
                            source = j.get('source', '')
                            beamt = j.get('beamtime', '')
                            qmax = float(j.get('q_max', np.nan))
                        except Exception:
                            pass
                    scan, fidx = parse_scan(source, n)
                    q = fp.quality_stats(img)
                    keys.append(n); scans.append(scan); fidxs.append(fidx)
                    srcs.append(source); beamts.append(beamt); qmaxs.append(qmax)
                    descs.append(fp.descriptor(img).astype(np.float16))
                    phashes.append(np.uint64(fp.phash64(img)))
                    md5s.append(hashlib.md5(b).hexdigest())
                    fvalid.append(q['frac_valid']); sat.append(q['saturation'])
                    blankf.append(q['blank']); isder.append(n.endswith('.raw.png'))
                except Exception as e:
                    continue
    except Exception as e:
        return (tarname, 0, f"TAR ERROR: {e}", time.time() - t0)
    if not keys:
        return (tarname, 0, "no frames", time.time() - t0)
    np.savez_compressed(
        os.path.join(OUT, tarname + ".npz"),
        key=np.array(keys), tar=np.array([tarname] * len(keys)),
        scan=np.array(scans), frame_idx=np.array(fidxs, np.int32),
        source=np.array(srcs), beamtime=np.array(beamts),
        q_max=np.array(qmaxs, np.float32),
        desc=np.stack(descs), phash=np.array(phashes, np.uint64),
        md5=np.array(md5s), frac_valid=np.array(fvalid, np.float32),
        saturation=np.array(sat, np.float32), blank=np.array(blankf),
        is_raw=np.array(isder))
    return (tarname, len(keys), "ok", time.time() - t0)


if __name__ == "__main__":
    tars = sorted(os.path.join(TARDIR, f) for f in os.listdir(TARDIR) if f.endswith('.tar'))
    only = [a for a in sys.argv[1:]]
    if only:
        tars = [t for t in tars if any(o in t for o in only)]
    print(f"processing {len(tars)} tars with {min(len(tars),12)} workers", flush=True)
    with Pool(min(len(tars), 12)) as p:
        for name, n, status, dt in p.imap_unordered(process_tar, tars):
            print(f"  [{status:>10}] {n:6d} frames  {dt:6.1f}s  {name}", flush=True)
    print("DONE", flush=True)
