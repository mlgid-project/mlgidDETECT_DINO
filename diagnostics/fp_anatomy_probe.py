"""Phase AB — WHAT are the false positives? 41 emits 507 of them and nobody has looked.

AA.5: organic makes 74 FPs sitting a median 21 px in χ from the nearest real segment, with the
CONFIDENT ones nearest (9.4 px). 41 makes **507** — one unmatched detection for every two real
segments — sitting a median **74 px** away, with confidence making no difference. Two different
failure modes, and the 41 one has never been characterised.

**A CORRECTION THIS PROBE EXISTS TO FIX.** AA.5 reported 41's FPs as only 23.5% "on-ring". That number
is not trustworthy: "on-ring" there meant *q-distance to the nearest SEGMENT < 8 px with high
integrated intensity*, because that probe had excluded rings from its reference set. On a gate where
**41% of labelled objects ARE rings** (AA.4: 16.9 near-full-span rings per frame), a detection sitting
squarely on a ring shows a large distance to the nearest segment and is scored as OFF-ring. The
measurement under-counts ring-associated errors by construction. Here rings are the reference.

FOUR CANDIDATE ANATOMIES, each with a measurement that separates it:

  ON RINGS      — the model finds extra peaks along a ring. Measured: share of FPs whose centre falls
                  inside a labelled RING's box, and share at the same radius (<8 px in q) as a ring.
  RING FRAGMENTS— the model chops a ring into pieces, each an unmatched detection. Measured: FPs split
                  by PREDICTED CLASS, with box-height percentiles against real ring and segment
                  heights. A fragmented ring shows as ring-radius detections carrying segment-sized
                  boxes.
  DUPLICATES    — the model double-reports things it already found. Measured: χ-distance from each FP
                  to the nearest MATCHED detection. Small distances mean duplication, not discovery.
  LOCALISED     — low-q noise or detector-edge artefacts. Measured: score and q-position percentiles
                  for FPs against the same for true positives.

Organic runs as the control throughout, so the two gates' failure modes are compared rather than
described separately.

SINGLE MODEL ssl1, score>0.3, deployed postprocessing. Rings identified by the AA.4 geometric
criterion (>=70% of the valid χ span), since 41's `is_ring` flag is empty — see
`diagnostics/ring_count_fix.py` and the memory note on that trap. GPU, ~10 min.
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
from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG, ST, HI
from diagnostics.ring_count_fix import classify

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
QTOL = 8.0
PCTS = [10, 25, 50, 75, 90]


def pcts(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return {p: float(np.percentile(v, p)) for p in PCTS} if len(v) else {p: float('nan') for p in PCTS}


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
    R = dict(fp_h=[], fp_score=[], fp_q=[], fp_cls=[], tp_h=[], tp_score=[], tp_q=[],
             fp_in_ring=[], fp_ringq=[], fp_segq=[], fp_to_tp=[], fp_to_anygt=[],
             gt_ring_h=[], gt_seg_h=[])
    n_fr = n_fp = n_tp = 0
    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]
            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            if not len(b):
                continue
            h_gt, _sp, is_ring = classify(img_np, b)
            R['gt_ring_h'] += list(h_gt[is_ring]); R['gt_seg_h'] += list(h_gt[~is_ring])
            n_fr += 1

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, a.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(),
                   out['pred_boxes'].detach().cpu().numpy()]
            gc2 = filter_boxes(cfg, onnx_to_xyxy(cfg, gc, raw))
            pred, sc = gc2.boxes, gc2.scores
            lab = getattr(gc2, 'pred_labels', None)
            lab = (lab.cpu().numpy() if hasattr(lab, 'cpu') else np.asarray(lab)) \
                if lab is not None else np.full(len(pred), -1)
            keep = (sc > ST).cpu().numpy() if hasattr(sc, 'cpu') else (sc > ST)
            pred = pred[keep]; sc = sc[keep]; lab = lab[keep] if len(lab) == len(keep) else lab

            gt_t = torch.tensor(b, dtype=torch.float32)
            col = np.array([], int)
            if len(pred):
                try:
                    _, _row, col = matcher(gt_t, pred)
                except IndexError:
                    pass
            cset = set(col.tolist())
            P = pred.numpy() if hasattr(pred, 'numpy') else np.asarray(pred)
            S = sc.numpy() if hasattr(sc, 'numpy') else np.asarray(sc)
            pq = (P[:, 0] + P[:, 2]) / 2; pc = (P[:, 1] + P[:, 3]) / 2
            ph = np.abs(P[:, 3] - P[:, 1])
            tp_idx = np.array(sorted(cset), dtype=int)
            gq = (b[:, 0] + b[:, 2]) / 2; gcc = (b[:, 1] + b[:, 3]) / 2
            rb = b[is_ring]; rq = gq[is_ring]; sq = gq[~is_ring]

            for j in range(len(P)):
                if j in cset:
                    n_tp += 1
                    R['tp_h'].append(ph[j]); R['tp_score'].append(float(S[j])); R['tp_q'].append(pq[j])
                    continue
                n_fp += 1
                R['fp_h'].append(ph[j]); R['fp_score'].append(float(S[j])); R['fp_q'].append(pq[j])
                R['fp_cls'].append(int(lab[j]) if j < len(lab) else -1)
                inr = bool(np.any((pq[j] >= rb[:, 0]) & (pq[j] <= rb[:, 2]) &
                                  (pc[j] >= rb[:, 1]) & (pc[j] <= rb[:, 3]))) if len(rb) else False
                R['fp_in_ring'].append(inr)
                R['fp_ringq'].append(bool(np.any(np.abs(rq - pq[j]) < QTOL)) if len(rq) else False)
                R['fp_segq'].append(bool(np.any(np.abs(sq - pq[j]) < QTOL)) if len(sq) else False)
                if len(tp_idx):
                    d = np.sqrt((pq[tp_idx] - pq[j]) ** 2 + (pc[tp_idx] - pc[j]) ** 2)
                    R['fp_to_tp'].append(float(d.min()))
                dg = np.sqrt((gq - pq[j]) ** 2 + (gcc - pc[j]) ** 2)
                R['fp_to_anygt'].append(float(dg.min()) if len(dg) else np.inf)
    if hasattr(ds, 'close'):
        ds.close()
    cls = np.asarray(R['fp_cls'])
    return dict(name=name, frames=n_fr, tp=n_tp, fp=n_fp,
                in_ring=float(np.mean(R['fp_in_ring'])) if R['fp_in_ring'] else float('nan'),
                ring_radius=float(np.mean(R['fp_ringq'])) if R['fp_ringq'] else float('nan'),
                seg_radius=float(np.mean(R['fp_segq'])) if R['fp_segq'] else float('nan'),
                cls_seg=float(np.mean(cls == 0)) if len(cls) else float('nan'),
                cls_ring=float(np.mean(cls == 1)) if len(cls) else float('nan'),
                fp_h=pcts(R['fp_h']), tp_h=pcts(R['tp_h']),
                gt_ring_h=pcts(R['gt_ring_h']), gt_seg_h=pcts(R['gt_seg_h']),
                fp_score=pcts(R['fp_score']), tp_score=pcts(R['tp_score']),
                fp_q=pcts(R['fp_q']), tp_q=pcts(R['tp_q']),
                fp_to_tp=pcts(R['fp_to_tp']), fp_to_anygt=pcts(R['fp_to_anygt']),
                dup_frac=float(np.mean(np.asarray(R['fp_to_tp']) < 20)) if R['fp_to_tp'] else float('nan'))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    rows = [run(n, p, model, a, dev) for n, p in DSETS]

    print("\n" + "=" * 100)
    print("  ARE THEY ON RINGS?  (rings by the AA.4 geometric criterion — 41's is_ring flag is empty)")
    print("=" * 100)
    print(f"  {'gate':<18s}{'FP':>6s}{'inside a ring box':>19s}{'at a ring radius':>18s}"
          f"{'at a segment radius':>21s}")
    for r in rows:
        print(f"  {r['name']:<18s}{r['fp']:6d}{r['in_ring']:19.3f}{r['ring_radius']:18.3f}"
              f"{r['seg_radius']:21.3f}")

    print("\n" + "=" * 100)
    print("  ARE THEY RING FRAGMENTS?  predicted class, and box height vs the real classes")
    print("=" * 100)
    print(f"  {'gate':<18s}{'pred segment':>14s}{'pred ring':>11s}")
    for r in rows:
        print(f"  {r['name']:<18s}{r['cls_seg']:14.3f}{r['cls_ring']:11.3f}")
    print(f"\n  box χ-height percentiles (px)")
    print(f"  {'gate / set':<28s}" + "".join(f"{('p' + str(p)):>10s}" for p in PCTS))
    for r in rows:
        for k, lbl in (('fp_h', 'FP boxes'), ('tp_h', 'TP boxes'),
                       ('gt_seg_h', 'GT segments'), ('gt_ring_h', 'GT rings')):
            print(f"  {(r['name'] + ' — ' + lbl):<28s}" +
                  "".join(f"{r[k][p]:10.1f}" for p in PCTS))

    print("\n" + "=" * 100)
    print("  ARE THEY DUPLICATES?  distance to the nearest MATCHED detection, and to any GT")
    print("=" * 100)
    print(f"  {'gate':<18s}" + "".join(f"{('p' + str(p)):>10s}" for p in PCTS) + f"{'<20px':>9s}")
    for r in rows:
        print(f"  {(r['name'] + ' → TP'):<18s}" +
              "".join(f"{r['fp_to_tp'][p]:10.1f}" for p in PCTS) + f"{r['dup_frac']:9.3f}")
    for r in rows:
        print(f"  {(r['name'] + ' → any GT'):<18s}" +
              "".join(f"{r['fp_to_anygt'][p]:10.1f}" for p in PCTS) + f"{'-':>9s}")

    print("\n" + "=" * 100)
    print("  ARE THEY LOCALISED?  score and q-position, FP vs TP")
    print("=" * 100)
    for k, lbl in (('fp_score', 'FP score'), ('tp_score', 'TP score'),
                   ('fp_q', 'FP q-position'), ('tp_q', 'TP q-position')):
        print(f"  {lbl:<16s}" + "".join(f"{('p' + str(p)):>10s}" for p in PCTS))
        for r in rows:
            print(f"    {r['name']:<14s}" + "".join(f"{r[k][p]:10.2f}" for p in PCTS))

    json.dump(rows, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/fp_anatomy.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
