"""Phase AA.5 — recall strata with RINGS EXCLUDED, both gates. AA.3's 41 strata were contaminated.

AA.3 stratified recall by χ-gap over ALL labelled objects. AA.4 then established (after the user
caught the ring count) that 41.h5 stores its rings as ordinary peaks — no ROI is marked `type==1`, so
the flag reads 0 — and that **16.9 of its 41 objects per frame are near-full-span rings**. Rings have
enormous χ extents, so they fall almost entirely into the widest gap bucket: AA.3's 41 "33 px+"
bucket held n=651 against ~692 rings in the set. Its 0.802 recall was therefore mostly RING recall,
and AA.3 used it as the well-separated-peak baseline for sizing the close-pair prize on 41. That
number is not trustworthy and is retracted.

Organic is barely affected (~28 rings in 817 objects, 3%), so its strata stand — but it is recomputed
here on the same footing so the two gates are comparable.

Rings are identified by the criterion the user supplied and AA.4 validated: a box covering >= 70% of
the valid (non-NaN) χ extent measured from the image at that box's radius. It needs no flag, which
matters because 41's flag is empty and the simulator's disagrees with its own geometry (14.8 flagged
vs 7.9 spanning).

REPORTED per gate: overall / segments-only / rings-only recall; precision; recall stratified by
χ-gap to the nearest same-radius SEGMENT (rings excluded from both the gap statistic and the strata);
and the false positives measured against segments only.

SINGLE MODEL ssl1, score>0.3, deployed postprocessing. GPU, ~10 min.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG, ST, HI, \
    ONRING_PX, HIQ_PCT
from diagnostics.ring_count_fix import classify, RING_FRAC

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
QTOL = 8.0
BINS = [(0, 5), (5, 10), (10, 20), (20, 33), (33, 1e9)]


def seg_gap(b):
    """χ-distance from each SEGMENT to its nearest same-radius SEGMENT (inf if none)."""
    n = len(b)
    out = np.full(n, np.inf)
    if n < 2:
        return out
    q = (b[:, 0] + b[:, 2]) / 2
    c = (b[:, 1] + b[:, 3]) / 2
    for i in range(n):
        m = np.abs(q - q[i]) < QTOL
        m[i] = False
        if m.any():
            out[i] = np.min(np.abs(c[m] - c[i]))
    return out


def run(name, path, model, a, dev):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                               buffer_size=5))
    matcher = get_matcher('q', min_iou=0.1)
    n_fr = 0
    n_seg = n_ring = seg_hit = ring_hit = 0
    tp = fp_on = fp_off = 0
    bn = {b: 0 for b in BINS}; bh = {b: 0 for b in BINS}
    fp_chi, fp_chi_on, fp_chi_hi, fp_q = [], [], [], []

    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]
            valid = img_np > 1e-6
            den = valid.sum(0); den[den == 0] = 1
            Iq = (img_np * valid).sum(0) / den
            Iq_pct = np.argsort(np.argsort(Iq)) / len(Iq)

            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            if not len(b):
                continue
            _h, _sp, is_ring = classify(img_np, b)      # the SPAN criterion, no flag involved
            n_fr += 1
            n_seg += int((~is_ring).sum()); n_ring += int(is_ring.sum())

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, a.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(),
                   out['pred_boxes'].detach().cpu().numpy()]
            gc2 = filter_boxes(cfg, onnx_to_xyxy(cfg, gc, raw))
            pred, sc = gc2.boxes, gc2.scores
            keep = sc > ST; pred, sc = pred[keep], sc[keep]

            gt_t = torch.tensor(b, dtype=torch.float32)
            row = np.array([], int); col = np.array([], int)
            if len(pred):
                try:
                    _, row, col = matcher(gt_t, pred)
                except IndexError:
                    pass
            hit = set(row.tolist()); cset = set(col.tolist())
            tp += len(cset)
            for i in range(len(b)):
                if is_ring[i]:
                    ring_hit += int(i in hit)
                else:
                    seg_hit += int(i in hit)

            segs = b[~is_ring]
            if len(segs):
                g = seg_gap(segs)
                idx_seg = np.where(~is_ring)[0]
                for k in range(len(segs)):
                    gi = g[k] if np.isfinite(g[k]) else 1e9
                    for bk in BINS:
                        if bk[0] <= gi < bk[1]:
                            bn[bk] += 1
                            bh[bk] += int(idx_seg[k] in hit)
                            break
                sq = (segs[:, 0] + segs[:, 2]) / 2
                scc = (segs[:, 1] + segs[:, 3]) / 2
                for j in range(len(pred)):
                    if j in cset:
                        continue
                    q = float((pred[j, 0] + pred[j, 2]) / 2)
                    c = float((pred[j, 1] + pred[j, 3]) / 2)
                    qd = float(np.min(np.abs(sq - q)))
                    m = np.abs(sq - q) < QTOL
                    cd = float(np.min(np.abs(scc[m] - c))) if m.any() else np.inf
                    fp_q.append(qd); fp_chi.append(cd)
                    onring = (qd < ONRING_PX) and (Iq_pct[int(np.clip(q, 0, 1023))] > HIQ_PCT)
                    if onring:
                        fp_on += 1; fp_chi_on.append(cd)
                    else:
                        fp_off += 1
                    if float(sc[j]) > HI:
                        fp_chi_hi.append(cd)
    if hasattr(ds, 'close'):
        ds.close()

    def med(v):
        v = np.asarray([x for x in v if np.isfinite(x)])
        return (float(np.median(v)), float((v < 10).mean()), len(v)) if len(v) else (float('nan'), float('nan'), 0)

    fp = fp_on + fp_off
    return dict(name=name, frames=n_fr, seg=n_seg, ring=n_ring,
                seg_recall=seg_hit / max(n_seg, 1), ring_recall=ring_hit / max(n_ring, 1),
                all_recall=(seg_hit + ring_hit) / max(n_seg + n_ring, 1),
                tp=tp, fp=fp, precision=tp / max(tp + fp, 1),
                fp_onring_frac=fp_on / max(fp, 1),
                buckets={f"{x[0]}-{'inf' if x[1] > 1e8 else x[1]}":
                         dict(n=bn[x], recall=bh[x] / max(bn[x], 1)) for x in BINS},
                fp_q_med=med(fp_q)[0], fp_chi=med(fp_chi), fp_chi_on=med(fp_chi_on),
                fp_chi_hi=med(fp_chi_hi))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1  RING_FRAC={RING_FRAC}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    rows = [run(n, p, model, a, dev) for n, p in DSETS]

    print("\n" + "=" * 104)
    print("  RECALL with rings separated (rings = box covering >=70% of the valid χ span at its radius)")
    print("=" * 104)
    print(f"  {'gate':<18s}{'frames':>7s}{'segs':>7s}{'rings':>7s}"
          f"{'seg recall':>12s}{'ring recall':>13s}{'all recall':>12s}{'precision':>11s}")
    for r in rows:
        print(f"  {r['name']:<18s}{r['frames']:7d}{r['seg']:7d}{r['ring']:7d}"
              f"{r['seg_recall']:12.3f}{r['ring_recall']:13.3f}{r['all_recall']:12.3f}"
              f"{r['precision']:11.3f}")

    print("\n" + "=" * 104)
    print("  SEGMENT recall by χ-gap to nearest same-radius SEGMENT (rings excluded from both)")
    print("=" * 104)
    keys = list(rows[0]['buckets'].keys())
    print(f"  {'gate':<18s}" + "".join(f"{k:>17s}" for k in keys))
    for r in rows:
        print(f"  {r['name']:<18s}" +
              "".join(f"{r['buckets'][k]['recall']:9.3f} (n{r['buckets'][k]['n']:>4d})"
                      for k in keys))

    print("\n" + "=" * 104)
    print("  FALSE POSITIVES measured against SEGMENTS only")
    print("=" * 104)
    print(f"  {'gate':<18s}{'FP':>6s}{'on-ring':>9s}{'q med':>8s}{'χ med':>8s}"
          f"{'χ<10px':>9s}{'on-ring χ':>11s}{'hi-conf χ':>11s}")
    for r in rows:
        print(f"  {r['name']:<18s}{r['fp']:6d}{r['fp_onring_frac']:9.3f}{r['fp_q_med']:8.1f}"
              f"{r['fp_chi'][0]:8.1f}{r['fp_chi'][1]:9.3f}{r['fp_chi_on'][0]:11.1f}"
              f"{r['fp_chi_hi'][0]:11.1f}")

    json.dump(rows, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/strata_segonly.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
