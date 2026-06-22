"""
Eval-leak detection: for every eval frame (41.h5 + organic), find its nearest shard
frames by flip-invariant descriptor cosine and perceptual-hash Hamming. Render a
side-by-side panel (eval | top-K shard matches) for visual confirmation, and write a
ranked table. The threshold for "this is a leak" is set AFTER looking at the panels.

Flip-invariance: eval reciprocal->polar caking may differ from the shard pipeline in
chi-direction / origin corner, so we compare each eval descriptor against the shards
under 4 flips (id, vflip, hflip, both) and keep the max cosine.
"""
import os, sys, glob, io, tarfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
TARDIR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)
K = 6
DCHI, DQ = 32, 64


def load_shards():
    parts = sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz")))
    cols = {k: [] for k in ["key", "tar", "scan", "frame_idx", "source", "desc", "phash",
                            "frac_valid", "saturation", "blank", "is_raw", "md5", "beamtime"]}
    for p in parts:
        d = np.load(p, allow_pickle=True)
        for k in cols:
            cols[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    return out, len(parts)


def flips(mat):
    """mat: (N,2048) -> list of 4 flipped variants (N,2048)."""
    g = mat.reshape(-1, DCHI, DQ)
    return [g.reshape(len(g), -1),
            g[:, ::-1, :].reshape(len(g), -1),
            g[:, :, ::-1].reshape(len(g), -1),
            g[:, ::-1, ::-1].reshape(len(g), -1)]


def l2norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return m / n


def hamming(a, b):
    """a:(N,) uint64, b: scalar uint64 -> popcount XOR."""
    x = np.bitwise_xor(a, np.uint64(b))
    c = np.zeros(len(x), np.int32)
    for i in range(64):
        c += ((x >> np.uint64(i)) & np.uint64(1)).astype(np.int32)
    return c


def main():
    ev = np.load(os.path.join(CUR, "eval_fp.npz"), allow_pickle=True)
    sh, nparts = load_shards()
    N = len(sh["key"])
    print(f"shards: {N} frames from {nparts} parts;  eval: {len(ev['eid'])} frames", flush=True)

    S = l2norm(sh["desc"].astype(np.float32))            # (N,2048) normed
    E = ev["desc"].astype(np.float32)                    # (M,2048)
    M = len(E)
    # flip-invariant cosine: max over 4 eval flips
    best = np.zeros((M, N), np.float32)
    for var in flips(E):
        sim = l2norm(var) @ S.T                          # (M,N)
        best = np.maximum(best, sim)
    # phash hamming (min over shards handled per-eval below)
    rows = []
    topk = {}
    for i in range(M):
        order = np.argsort(-best[i])[:K]
        ham = hamming(sh["phash"], ev["phash"][i])
        ham_best = int(ham.min()); ham_arg = int(ham.argmin())
        topk[i] = order
        rows.append(dict(
            eid=str(ev["eid"][i]), evalset=str(ev["evalset"][i]),
            best_cos=float(best[i, order[0]]),
            top_scan=str(sh["scan"][order[0]]), top_key=str(sh["key"][order[0]]),
            top_frame=int(sh["frame_idx"][order[0]]),
            cos_list=[round(float(best[i, j]), 3) for j in order],
            scan_list=[str(sh["scan"][j]) for j in order],
            min_hamming=ham_best, ham_scan=str(sh["scan"][ham_arg]),
            meta=str(ev["meta"][i])[:120]))
    rows.sort(key=lambda r: -r["best_cos"])

    # write table
    with open(os.path.join(CUR, "leak_candidates.tsv"), "w") as f:
        f.write("best_cos\tmin_hamming\teid\tevalset\ttop_scan\ttop_frame\tcos_top6\tmeta\n")
        for r in rows:
            f.write(f"{r['best_cos']:.3f}\t{r['min_hamming']}\t{r['eid']}\t{r['evalset']}\t"
                    f"{r['top_scan']}\t{r['top_frame']}\t{r['cos_list']}\t{r['meta']}\n")
    print("\nTOP eval->shard matches (sorted by cosine):", flush=True)
    print(f"{'cos':>5} {'ham':>4} {'eval id':<48} {'best shard scan':<45} {'cos top-6'}", flush=True)
    for r in rows:
        print(f"{r['best_cos']:5.3f} {r['min_hamming']:4d} {r['eid'][:48]:<48} "
              f"{r['top_scan'][:45]:<45} {r['cos_list']}", flush=True)

    # ---- render panels for the strongest candidates (cos>=0.5) ----
    render = [i for i in range(M) if best[i, topk[i][0]] >= 0.5]
    # gather needed shard thumbs grouped by tar
    need = {}
    for i in render:
        for j in topk[i]:
            need.setdefault(str(sh["tar"][j]) + ".tar", []).append((str(sh["key"][j]), j))
    thumb_by_j = {}
    for tarbase, items in need.items():
        try:
            with tarfile.open(os.path.join(TARDIR, tarbase)) as tar:
                mem = {m.name: m for m in tar.getmembers()}
                for key, j in items:
                    if key in mem:
                        b = tar.extractfile(mem[key]).read()
                        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
                        if im is not None:
                            if im.ndim == 3: im = im[..., 0]
                            thumb_by_j[j] = cv2.resize(im, (256, 128), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print("panel extract err", tarbase, e)

    def lab(img, txt):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, txt, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)
        return img
    panels = []
    for i in render:
        evt = cv2.resize(ev["thumb"][i], (256, 128), interpolation=cv2.INTER_AREA)
        rowimg = [lab(evt, "EVAL " + str(ev["eid"][i]).split('/')[-1][:22])]
        for rank, j in enumerate(topk[i]):
            t = thumb_by_j.get(j, np.zeros((128, 256), np.uint8))
            rowimg.append(lab(t, f"{best[i,j]:.2f} {str(sh['scan'][j]).split('/')[-1][:16]}"))
        panels.append(np.hstack(rowimg))
    if panels:
        W = max(p.shape[1] for p in panels)
        panels = [np.pad(p, ((0, 0), (0, W - p.shape[1]), (0, 0))) for p in panels]
        sheet = np.vstack(panels)
        cv2.imwrite(os.path.join(FIG, "leak_panels.png"), sheet)
        print(f"\nwrote {len(panels)} panels (cos>=0.5) -> figures/leak_panels.png", flush=True)
    else:
        print("\nno candidates with cos>=0.5 to render", flush=True)


if __name__ == "__main__":
    main()
