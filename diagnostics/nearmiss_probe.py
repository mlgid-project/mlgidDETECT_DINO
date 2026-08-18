"""
Near-miss probe — WHERE in the pipeline does a missed peak get lost?

Follows the prominence probe (diagnostics/prominence_probe.py, 2026-08-18), which showed the
misses are NOT contrast-limited (separation AUC 0.489 organic / 0.597 on 41) and instead track the
annotator's confidence. "There is headroom" is not yet actionable: a missed peak can be lost at
four different places, each needing a completely different fix. This probe assigns every miss to
one of them by replaying the deployed ensemble's own stages:

    900 queries x 2 classes  --top-225-->  pooled 450  --NMS-->  survivors  --score>0.3-->  output

For each missed GT peak we record the best score of a COMPATIBLE box (the q-matcher's own
criterion: IoU > 0.1 and |dq_centre| < 10 px) at each stage, then bucket it:

  ASSIGNMENT   a qualifying box exists in the final output, but Hungarian assignment gave it to a
               different GT peak      -> duplicate/competition in crowded frames
  BELOW_THRESH box survives NMS but scores under 0.30
               -> ranking/calibration; the operating point is the fix (and the phase-Q label-
                  completeness finding means the measured precision cost of lowering it is
                  OVERSTATED)
  NMS_KILLED   box was in the top-225 but NMS removed it
               -> NMS IoU thresholds are eating true adjacent peaks (cf. the negative recall-vs-
                  crowding correlation, -0.39 on 41)
  RANK_CUT     a query responded but did not survive top-225 selection
               -> output cap binds (num_select=225, num_queries=900)
  NO_RESPONSE  no query anywhere produces a compatible box at any score
               -> genuine representational blindness; only this case justifies a new head

The bucket x annotator-confidence cross-tab is the payload: if the conf=0.1 misses are mostly
NO_RESPONSE, model and annotator agree nothing is there (evidence those labels are unreliable);
if they carry real sub-threshold scores, they are real peaks the model under-ranks.

Also sweeps the score threshold so the recall available from re-tuning the operating point is
quantified rather than guessed.

GPU, ~10 min. See tmp_diag/run_nearmiss_probe.sbatch.
"""
import os, sys, json
import numpy as np
import torch
from torchvision.ops import box_iou
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes, box_cxcywh_to_xyxy
from util.matchers import get_matcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CKPT_B, CONFIG, DSETS, OUT

ST = 0.30           # deployed operating point
NMS_BASE = 0.001    # score floor BEFORE the ST cut, so NMS is isolated from thresholding
MIN_IOU = 0.1       # q-matcher compatibility (util/matchers.calc_box_dq_mtx)
QTHRESH = 10.0
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
BUCKETS = ['ASSIGNMENT', 'BELOW_THRESH', 'NMS_KILLED', 'RANK_CUT', 'NO_RESPONSE']


def raw_candidates(config, out):
    """All 900 query boxes with their best-class score — the model's response BEFORE top-k."""
    logits = out['pred_logits'].detach().cpu()          # (1, 900, ncls)
    bbox = out['pred_boxes'].detach().cpu()             # (1, 900, 4)
    boxes = box_cxcywh_to_xyxy(config, bbox)            # (900, 4) polar pixels
    scores = logits.sigmoid()[0].max(dim=-1).values     # (900,)
    return boxes, scores


def best_compatible(gt, boxes, scores):
    """For each GT box, the best score among boxes compatible under the q-matcher's criterion."""
    n = len(gt)
    if n == 0:
        return np.zeros(0)
    if len(boxes) == 0:
        return np.zeros(n)
    iou = box_iou(gt, boxes).numpy()
    qt = ((gt[:, 0] + gt[:, 2]) / 2).numpy()
    qp = ((boxes[:, 0] + boxes[:, 2]) / 2).numpy()
    ok = (iou > MIN_IOU) & (np.abs(qt[:, None] - qp[None, :]) < QTHRESH)
    sc = scores.numpy()[None, :] * ok
    return sc.max(axis=1)


def run_dataset(tag, path, modelA, modelB, a, device):
    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = NMS_BASE
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = path
    if detect_dataset_type(path) == 'pygid':
        ds = PyGIDDataset(config, path=path, preprocess_func=standard_preprocessing,
                          buffer_size=5, load_labels=True)
    else:
        ds = H5GIWAXSDataset(config, path=path, preprocess_func=standard_preprocessing,
                             buffer_size=5)
    matcher = get_matcher('q', min_iou=MIN_IOU)

    R = dict(det=[], bucket=[], s_raw=[], s_topk=[], s_nms=[], s_op=[], conf=[], qn=[], frame=[])
    sweep_hits = {t: [0, 0] for t in SWEEP}          # thr -> [matched GT, detections]
    nf = 0
    with torch.no_grad():
        for gc in ds.iter_images():
            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device)
            img = img.repeat(1, a.num_channels, 1, 1)
            outA, outB = modelA(img), modelB(img)

            # stage 1: raw query responses, both models pooled
            brA, srA = raw_candidates(config, outA)
            brB, srB = raw_candidates(config, outB)
            b_raw, s_raw = torch.cat([brA, brB]), torch.cat([srA, srB])

            # stage 2: each model's top-225, pooled (exactly the deployed ensemble input)
            def topk(model_out):
                raw = [model_out['pred_logits'].detach().cpu().numpy(),
                       model_out['pred_boxes'].detach().cpu().numpy()]
                onnx_to_xyxy(config, gc, raw)
                return gc.boxes.clone(), gc.scores.clone(), gc.pred_labels.clone()
            bA, sA, lA = topk(outA)
            bB, sB, lB = topk(outB)
            b_topk, s_topk = torch.cat([bA, bB]), torch.cat([sA, sB])

            # stage 3: production class-aware NMS (score floor only, no operating-point cut yet)
            gc.boxes, gc.scores = b_topk.clone(), s_topk.clone()
            gc.pred_labels = torch.cat([lA, lB])
            filter_boxes(config, gc)
            b_nms, s_nms = gc.boxes.clone(), gc.scores.clone()

            # stage 4: the operating point
            keep = s_nms > ST
            b_op, s_op = b_nms[keep], s_nms[keep]

            L = gc.polar_labels
            gt = (torch.tensor(np.array(L.boxes), dtype=torch.float32)
                  if len(L.boxes) else torch.zeros((0, 4)))
            conf = np.asarray(L.confidences if len(L.confidences) else [np.nan] * len(gt), float)
            if len(gt) == 0:
                nf += 1
                continue

            row = np.array([], int)
            if len(b_op):
                try:
                    _, row, _ = matcher(gt, b_op)
                except IndexError:
                    pass
            mset = set(row.tolist())

            v_raw = best_compatible(gt, b_raw, s_raw)
            v_topk = best_compatible(gt, b_topk, s_topk)
            v_nms = best_compatible(gt, b_nms, s_nms)
            v_op = best_compatible(gt, b_op, s_op)

            for i in range(len(gt)):
                det = i in mset
                if det:
                    bucket = 'DETECTED'
                elif v_op[i] > 0:
                    bucket = 'ASSIGNMENT'
                elif v_nms[i] > 0:
                    bucket = 'BELOW_THRESH'
                elif v_topk[i] > 0:
                    bucket = 'NMS_KILLED'
                elif v_raw[i] > 0:
                    bucket = 'RANK_CUT'
                else:
                    bucket = 'NO_RESPONSE'
                R['det'].append(det); R['bucket'].append(bucket)
                R['s_raw'].append(float(v_raw[i])); R['s_topk'].append(float(v_topk[i]))
                R['s_nms'].append(float(v_nms[i])); R['s_op'].append(float(v_op[i]))
                R['conf'].append(float(conf[i]) if i < len(conf) else np.nan)
                R['qn'].append(float((gt[i, 0] + gt[i, 2]) / 2) / 1024.0)
                R['frame'].append(nf)

            for t in SWEEP:                       # threshold sweep on the SAME cached NMS output
                k = s_nms > t
                bt = b_nms[k]
                rr = np.array([], int)
                if len(bt):
                    try:
                        _, rr, _ = matcher(gt, bt)
                    except IndexError:
                        pass
                sweep_hits[t][0] += len(set(rr.tolist()))
                sweep_hits[t][1] += len(bt)
            nf += 1
            print(f"  [{tag}] frame {nf}: GT={len(gt)} op={len(b_op)} nms={len(b_nms)}", flush=True)

    if hasattr(ds, 'close'):
        ds.close()
    R = {k: np.array(v) for k, v in R.items()}
    return R, sweep_hits, nf


def report(tag, R, sweep_hits, nf):
    det = R['det'].astype(bool)
    bucket = R['bucket']
    N = len(det)
    print("\n" + "=" * 96)
    print(f"{tag.upper()}   {nf} frames, {N} labeled peaks, recall={det.mean():.3f} @score>{ST}")
    print("=" * 96)
    miss = ~det
    nm = int(miss.sum())
    print(f"\n  WHERE THE {nm} MISSES ARE LOST")
    print(f"    {'bucket':14s} {'n':>5s} {'% of misses':>12s} {'% of all GT':>12s}  "
          f"{'median best score':>18s}")
    for b in BUCKETS:
        m = miss & (bucket == b)
        if m.sum() == 0:
            print(f"    {b:14s} {0:5d} {0.0:11.1f}% {0.0:11.1f}%")
            continue
        key = dict(ASSIGNMENT='s_op', BELOW_THRESH='s_nms', NMS_KILLED='s_topk',
                   RANK_CUT='s_raw', NO_RESPONSE='s_raw')[b]
        print(f"    {b:14s} {int(m.sum()):5d} {100*m.sum()/nm:11.1f}% {100*m.sum()/N:11.1f}%  "
              f"{np.median(R[key][m]):18.3f}")

    print(f"\n  MODEL RESPONSE AT THE PEAK (best raw query score, before ANY postprocessing)")
    print(f"    detected peaks : median={np.median(R['s_raw'][det]):.3f}  "
          f"q25={np.quantile(R['s_raw'][det],.25):.3f}  q75={np.quantile(R['s_raw'][det],.75):.3f}")
    if nm:
        print(f"    MISSED peaks   : median={np.median(R['s_raw'][miss]):.3f}  "
              f"q25={np.quantile(R['s_raw'][miss],.25):.3f}  "
              f"q75={np.quantile(R['s_raw'][miss],.75):.3f}")
        z = int((R['s_raw'][miss] == 0).sum())
        print(f"    missed peaks with NO response at all: {z} ({100*z/nm:.1f}% of misses)")

    cf = R['conf']
    if np.isfinite(cf).any():
        print(f"\n  BUCKET x ANNOTATOR CONFIDENCE  (row % of that confidence tier's MISSES)")
        print(f"    {'conf':>6s} {'misses':>7s} " + "".join(b[:11].rjust(13) for b in BUCKETS))
        for c in np.unique(np.round(cf[np.isfinite(cf)], 3)):
            m = miss & np.isclose(cf, c, atol=1e-3)
            if m.sum() < 5:
                continue
            row = f"    {c:6.1f} {int(m.sum()):7d} "
            for b in BUCKETS:
                row += f"{100*np.sum(m & (bucket == b))/m.sum():12.1f}%"
            print(row)
        print(f"\n    (same rows, median raw response score at the missed peak)")
        for c in np.unique(np.round(cf[np.isfinite(cf)], 3)):
            m = miss & np.isclose(cf, c, atol=1e-3)
            if m.sum() < 5:
                continue
            print(f"    conf={c:<4} n={int(m.sum()):4d}  median s_raw={np.median(R['s_raw'][m]):.3f}"
                  f"   frac with zero response={np.mean(R['s_raw'][m] == 0):.2f}")

    print(f"\n  OPERATING-POINT SWEEP  (recall available from re-thresholding alone)")
    print(f"    {'thr':>6s} {'recall':>8s} {'d recall':>9s} {'detections':>11s} {'det/frame':>10s}")
    base = sweep_hits[ST][0] / N
    for t in SWEEP:
        hit, ndet = sweep_hits[t]
        print(f"    {t:6.2f} {hit/N:8.3f} {hit/N-base:+9.3f} {ndet:11d} {ndet/max(nf,1):10.1f}")

    return dict(n=N, recall=float(det.mean()), n_miss=nm,
                buckets={b: int(np.sum(miss & (bucket == b))) for b in BUCKETS},
                s_raw_det=float(np.median(R['s_raw'][det])),
                s_raw_miss=float(np.median(R['s_raw'][miss])) if nm else None,
                zero_response_frac=float(np.mean(R['s_raw'][miss] == 0)) if nm else None,
                sweep={str(t): dict(recall=sweep_hits[t][0] / N, dets=sweep_hits[t][1])
                       for t in SWEEP})


def merge_prominence(tag, R):
    """Cross-tab bucket against the prominence measured by the previous probe (same GT order)."""
    f = os.path.join(OUT, f'prominence_{tag}.npz')
    if not os.path.exists(f):
        return
    z = np.load(f)
    if len(z['gt_prom']) != len(R['det']):
        print(f"\n  (prominence merge skipped: {len(z['gt_prom'])} vs {len(R['det'])} rows)")
        return
    if not np.allclose(np.nan_to_num(z['gt_qn']), np.nan_to_num(R['qn']), atol=1e-6):
        print("\n  (prominence merge skipped: GT order mismatch)")
        return
    prom = z['gt_prom']; miss = ~R['det'].astype(bool)
    print(f"\n  BUCKET x PROMINENCE  (is the model blind to faint peaks, or to prominent ones?)")
    print(f"    {'bucket':14s} {'n':>5s} {'median prominence':>18s}")
    print(f"    {'DETECTED':14s} {int((~miss).sum()):5d} {np.median(prom[~miss]):18.4f}")
    for b in BUCKETS:
        m = miss & (R['bucket'] == b)
        if m.sum() >= 5:
            print(f"    {b:14s} {int(m.sum()):5d} {np.median(prom[m]):18.4f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)
    print("building model A (ssl1)");     modelA, a = build_model_from_ckpt(CONFIG, CKPT_A, device)
    print("building model B (baseline)"); modelB, _ = build_model_from_ckpt(CONFIG, CKPT_B, device)

    summary = {}
    for tag, path in DSETS:
        print(f"\n########## {tag} ##########", flush=True)
        R, sw, nf = run_dataset(tag, path, modelA, modelB, a, device)
        summary[tag] = report(tag, R, sw, nf)
        merge_prominence(tag, R)
        np.savez(os.path.join(OUT, f'nearmiss_{tag}.npz'), **R)
    json.dump(summary, open(os.path.join(OUT, 'nearmiss_probe.json'), 'w'), indent=2, default=str)
    print("\nwrote nearmiss_probe.json")
    print("PROBE DONE")


if __name__ == '__main__':
    main()
