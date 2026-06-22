"""
Clinching discriminator: combine I(q) (radial, q-calibrated) with I(chi) (azimuthal,
scale-invariant). A genuine duplicate matches in BOTH; same-material-different-sample
matches in I(q) only. leak_score = min(corr_Iq, corr_Ichi_flipinv).

Calibration: within-shard same-scan adjacent frames give the 'same-frame' level for
leak_score (expect ~0.98). If no eval frame approaches that, there is no leak.
"""
import os, sys, glob
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iq_match as IQ
CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
NCHI = 512


def eval_ichi(img, qz, qx):
    a = np.log1p(np.clip(np.nan_to_num(img), 0, None).astype(np.float64))
    m = (img != 0) & np.isfinite(img)
    chi = np.degrees(np.arctan2(np.abs(qz)[:, None], np.abs(qx)[None, :] + 1e-9))  # [0,90]
    b = np.clip((chi / 90.0 * NCHI).astype(int), 0, NCHI - 1)
    num = np.bincount(b[m].ravel(), weights=a[m].ravel(), minlength=NCHI)
    cnt = np.bincount(b[m].ravel(), minlength=NCHI)
    return np.divide(num, cnt, out=np.zeros(NCHI), where=cnt > 0)


def corr_flip(a, b):
    return max(IQ.corr(a, b), IQ.corr(a, b[::-1]))


def load_shard_all():
    parts = sorted(glob.glob(os.path.join(CUR, "profile_parts", "*.npz")))
    keys, scans, frames, qmax, iqs, rows = [], [], [], [], [], []
    for p in parts:
        d = np.load(p, allow_pickle=True)
        keys.append(d['key']); scans.append(d['scan']); frames.append(d['frame_idx'])
        qmax.append(d['q_max'])
        col = d['col'].astype(np.float32)
        iqs.append(np.stack([IQ.shard_iq_from_col(col[i], float(d['q_max'][i])) for i in range(len(col))]).astype(np.float32))
        rows.append(d['row'].astype(np.float32))
    return (np.concatenate(keys), np.concatenate(scans), np.concatenate(frames),
            np.concatenate(iqs), np.concatenate(rows))


def main():
    keys, scans, frames, S_iq, S_chi = load_shard_all()
    N = len(keys)

    # calibration: same-scan adjacent -> same-frame leak_score level
    from collections import Counter
    cc = Counter(scans.tolist())
    for s, c in cc.items():
        if c >= 50:
            fi = frames[scans == s]
            if int(fi.max() - fi.min() + 1) == c:
                idx = np.where(scans == s)[0]; idx = idx[np.argsort(frames[idx])]
                sc = [min(IQ.corr(S_iq[idx[k]], S_iq[idx[k+1]]),
                          corr_flip(S_chi[idx[k]], S_chi[idx[k+1]])) for k in range(30)]
                print(f"[calib] same-scan adjacent leak_score (min of Iq,Ichi): median {np.median(sc):.3f} "
                      f"min {np.min(sc):.3f}  (this is the 'same-frame' level)")
                break

    # eval
    ev_iq = IQ.load_eval_iq()
    ev_chi = {}
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/41.h5", "r") as f:
        for fac in f.keys():
            for samp in f[fac].keys():
                g = f[fac][samp]
                if 'image' in g and g['image'].ndim == 2:
                    img = np.asarray(g['image']); qz, qx = IQ.axes_for_eval(g, "41", img.shape)
                    ev_chi[f"41/{fac}/{samp}"] = eval_ichi(img, qz, qx)
    with h5py.File("/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5", "r") as f:
        for ent in f.keys():
            if 'data/img_gid_q' in f[ent]:
                g = f[ent]; img = np.asarray(g['data/img_gid_q'][0])
                qz, qx = IQ.axes_for_eval(g, "organic", img.shape)
                ev_chi[f"organic/{ent}/0"] = eval_ichi(img, qz, qx)

    # for each eval: top-12 by I(q), then combined min(Iq, Ichi_flip)
    Sz = np.stack([(lambda r: r[0] if r else np.zeros(IQ.B))(IQ.zstd(S_iq[i])) for i in range(N)])
    Sn = Sz / (np.linalg.norm(Sz, axis=1, keepdims=True) + 1e-9)
    out = []
    for eid, iq in ev_iq:
        r = IQ.zstd(iq)
        if r is None:
            out.append((eid, 0.0, 0.0, 0.0, "no-data", -1)); continue
        z = r[0] / (np.linalg.norm(r[0]) + 1e-9)
        cand = np.argsort(-(Sn @ z))[:12]
        best = None
        for j in cand:
            ciq = IQ.corr(S_iq[j], iq)
            cchi = corr_flip(S_chi[j], ev_chi[eid])
            comb = min(ciq, cchi)
            if best is None or comb > best[0]:
                best = (comb, ciq, cchi, str(scans[j]), int(frames[j]))
        out.append((eid, *best))
    out.sort(key=lambda t: -t[1])
    print(f"\n{'combo':>6} {'Iq':>5} {'Ichi':>5} {'eval id':<48} {'best shard scan':<40} frm")
    for eid, comb, ciq, cchi, sc, fr in out:
        flag = "  <-- SAME-FRAME?" if comb >= 0.95 else ""
        print(f"{comb:6.3f} {ciq:5.3f} {cchi:5.3f} {eid[:48]:<48} {sc[:40]:<40} {fr}{flag}")
    mx = max(o[1] for o in out)
    print(f"\nMAX combined leak_score across all 49 eval frames: {mx:.3f}")
    with open(os.path.join(CUR, "combined_leak.tsv"), "w") as fo:
        fo.write("combo\tIq\tIchi\teid\tbest_scan\tframe\n")
        for eid, comb, ciq, cchi, sc, fr in out:
            fo.write(f"{comb:.3f}\t{ciq:.3f}\t{cchi:.3f}\t{eid}\t{sc}\t{fr}\n")


if __name__ == "__main__":
    main()
