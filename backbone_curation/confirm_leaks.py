"""
Confirm/deny leaks for the top I(q) candidates:
  (1) same-material baseline: I(q) corr between DIFFERENT perovskite scans, to contextualize
      whether the eval top-corr (~0.96) is 'same frame' or just 'same material'.
  (2) visual panels: eval caked polar | its top-K shard I(q) matches (re-read polar PNGs),
      so same-frame (identical rings AND azimuthal peaks) vs same-material (rings match,
      texture differs) can be told apart by eye.
"""
import os, sys, glob, tarfile
import numpy as np
import cv2, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_common as fp
import iq_match as IQ
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
TARDIR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    keys, scans, frames, qmax, S = IQ.load_shard_profiles()
    N = len(keys)
    # key -> tar (from shard_parts)
    key2tar = {}
    for p in sorted(glob.glob(os.path.join(CUR, "shard_parts", "*.npz"))):
        d = np.load(p, allow_pickle=True)
        for k, t in zip(d['key'], d['tar']):
            key2tar[str(k)] = str(t)

    # ---- (1) same-material baseline ----
    print("=== same-material baseline: I(q) corr between DIFFERENT scans of same material ===")
    def scan_first_idx(substr, maxn=10):
        out = {}
        for i in range(N):
            s = str(scans[i])
            if substr in s.lower() and s not in out:
                out[s] = i
            if len(out) >= maxn:
                break
        return list(out.values())
    for mat in ["ekaterina", "drs", "mapi", "fapi"]:
        idx = scan_first_idx(mat)
        if len(idx) >= 3:
            cs = []
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    cs.append(IQ.corr(S[idx[a]], S[idx[b]]))
            cs = np.array(cs)
            print(f"   {mat:10s} ({len(idx)} distinct scans): I(q) corr median {np.median(cs):.3f} "
                  f"p90 {np.percentile(cs,90):.3f} max {cs.max():.3f}")

    # ---- (2) panels for top eval candidates ----
    ev = IQ.load_eval_iq()
    eval_imgs = {}
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/41.h5", "r") as f:
        for fac in f.keys():
            for samp in f[fac].keys():
                g = f[fac][samp]
                if 'image' in g and g['image'].ndim == 2:
                    eval_imgs[f"41/{fac}/{samp}"] = np.asarray(g['image'])
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5", "r") as f:
        for ent in f.keys():
            if 'data/img_gid_q' in f[ent]:
                eval_imgs[f"organic/{ent}/0"] = np.asarray(f[ent]['data/img_gid_q'][0])

    # rank eval by best I(q) corr
    Sz = []
    for i in range(N):
        r = IQ.zstd(S[i]); Sz.append(r[0] if r else np.zeros(IQ.B))
    Sz = np.stack(Sz); Sn = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
    ranked = []
    for eid, iq in ev:
        r = IQ.zstd(iq)
        if r is None:
            continue
        z = r[0] / (np.linalg.norm(r[0]) + 1e-9)
        sim = Sn @ z
        cand = np.argsort(-sim)[:12]
        exact = sorted([(int(j), IQ.corr(S[j], iq)) for j in cand], key=lambda t: -t[1])[:4]
        ranked.append((eid, exact))
    ranked.sort(key=lambda t: -t[1][0][1])

    def thumb_shard(j):
        key = str(keys[j]); tarb = key2tar.get(key)
        if not tarb:
            return np.zeros((128, 256), np.uint8)
        with tarfile.open(os.path.join(TARDIR, tarb + ".tar")) as tar:
            m = {x.name: x for x in tar.getmembers()}
            if key not in m:
                return np.zeros((128, 256), np.uint8)
            b = tar.extractfile(m[key]).read()
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
            if im is None:
                return np.zeros((128, 256), np.uint8)
            if im.ndim == 3: im = im[..., 0]
            return cv2.resize(im, (256, 128), interpolation=cv2.INTER_AREA)

    def lab(img, txt, color=(0, 255, 0)):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, txt, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        return img

    rowimgs = []
    TOPE = 22
    for eid, exact in ranked[:TOPE]:
        ec = fp.cake_reciprocal(eval_imgs[eid])
        et = cv2.resize(fp.to_u8(ec), (256, 128), interpolation=cv2.INTER_AREA)
        row = [lab(et, "EVAL " + eid.split('/')[-1][:24], (0, 200, 255))]
        for j, c in exact:
            color = (0, 0, 255) if c >= 0.985 else (0, 255, 0)
            row.append(lab(thumb_shard(j), f"{c:.3f} {str(scans[j]).split('/')[-1][:16]}", color))
        rowimgs.append(np.hstack(row))
    W = max(r.shape[1] for r in rowimgs)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rowimgs])
    cv2.imwrite(os.path.join(FIG, "iq_confirm_panels.png"), sheet)
    print(f"\nwrote panels for top {TOPE} eval candidates -> figures/iq_confirm_panels.png")
    print("(red label = corr>=0.985 'same-frame level'; green = lower)")


if __name__ == "__main__":
    main()
