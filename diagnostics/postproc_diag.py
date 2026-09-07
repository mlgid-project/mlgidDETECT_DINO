"""Is the rings+segments gap a POSTPROCESSING problem or a training problem?

Observation to explain: on frames carrying many rings AND many segments, the old faster_rcnn
path is more reliable than DINO. Everything downstream of the network is shared logic, so
before spending another 3-day training run we check whether the loss happens after the
forward pass.

Runs each ONNX model over both labeled gates ONCE, caches the raw (pred_logits, pred_boxes),
and then replays many postprocessing configurations off that cache for free.

Sections
  A  premise      per-frame ap_total vs crowding (GT count, ring:segment mix). Writes a
                  per-frame CSV so it can be joined against faster_rcnn in compare_predictions.ipynb.
  B  cap          does the top-225 selection in onnx_to_xyxy truncate real candidates?
  C  duplicates   topk runs over the FLATTENED (query x class) grid, so one query can be
                  selected as both ring and segment. Class-aware NMS then sorts those two
                  identical boxes into DIFFERENT pools, where neither can suppress the other.
  D  extent       predicted vs GT angular extent, and how often the head's class disagrees
                  with the >=35% chi-span geometry the old util/nms.py heuristic used.
  E  sweep        ap_total over num_select x NMSIOU_RING x NMSIOU_SEG, plus class-agnostic
                  NMS and a cross-class dedupe variant.

CPU only (onnxruntime):  python diagnostics/postproc_diag.py [model.onnx ...]
"""
import os, sys, csv, itertools
import numpy as np
import torch
from torchvision.ops import nms

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes, box_cxcywh_to_xyxy
from util.evaluation import Evaluator, get_full_conf_results

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DATASETS = {'41': f'{CUR}/41.h5', 'organic': f'{CUR}/organic_labeled.h5'}
MODELS = sys.argv[1:] or [f'{CUR}/DINO_BACKBONE_curation/onnx/dino_ssl1.onnx']
POLAR = (512, 1024)
#WIRING CHECK ONLY. This takes the FIRST N frames, not a sample of the file, and the head of
#41.h5 is ring-heavy and segment-free -- on 2 frames it reports the top-k cap as never binding
#(really 7/41) and the predicted angular extent as too LONG (really 12% too short). Never quote
#a number from a capped run.
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '0')) or None
if MAX_FRAMES:
    print(f'!! MAX_FRAMES={MAX_FRAMES}: first {MAX_FRAMES} frames per set, NOT a sample. '
          f'Wiring check only -- the numbers below are not measurements.', flush=True)
OUT = os.path.join(_REPO, 'diagnostics')

#deployed operating point (main.py evaluate_giwaxs_ap / mlgiddetect defaults)
DEF_SELECT, DEF_RING, DEF_SEG, DEF_SCORE = 225, 0.1, 0.4, 0.1


class _P:  # onnx_to_xyxy / filter_boxes only touch .boxes/.scores/.pred_labels
    pass


# ----------------------------------------------------------------------------- postprocessing
def postprocess(logits, boxes_cxcywh, num_select=DEF_SELECT, iou_ring=DEF_RING,
                iou_seg=DEF_SEG, score=DEF_SCORE, classaware=True, dedupe=False):
    """Replay of onnx_to_xyxy + filter_boxes with every knob exposed.

    Returns (boxes xyxy, scores, labels, query_idx). `dedupe` keeps only the better-scoring
    class per query BEFORE NMS, which is the fix for the section-C duplicate mechanism.
    """
    prob = torch.from_numpy(logits).sigmoid()           # (1, Q, C)
    Q, C = prob.shape[1], prob.shape[2]
    k = min(num_select, Q * C)
    topk_values, topk_idx = torch.topk(prob.view(1, -1), k, dim=1)
    scores_ = topk_values[0]
    qidx = topk_idx[0] // C
    labels = topk_idx[0] % C

    boxes = box_cxcywh_to_xyxy(_CFG, torch.from_numpy(boxes_cxcywh))[qidx]

    if dedupe:                                          # one entry per query, best class wins
        seen, keep = {}, []
        for i, q in enumerate(qidx.tolist()):           # topk is score-sorted, first wins
            if q not in seen:
                seen[q] = i
                keep.append(i)
        keep = torch.tensor(keep, dtype=torch.long)
        boxes, scores_, labels, qidx = boxes[keep], scores_[keep], labels[keep], qidx[keep]

    if classaware:
        parts = []
        for cls, thr in ((1, iou_ring), (0, iou_seg)):
            ci = (labels == cls).nonzero(as_tuple=True)[0]
            if ci.numel():
                parts.append(ci[nms(boxes[ci], scores_[ci], thr)])
        idx = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    else:
        idx = nms(boxes, scores_, iou_seg)

    boxes, scores_, labels, qidx = boxes[idx], scores_[idx], labels[idx], qidx[idx]
    m = scores_ > score
    return boxes[m], scores_[m], labels[m], qidx[m]


def ap_of(pairs, per_frame=False):
    """ap_total over a list of (pred_boxes, scores, gt_boxes, gt_conf)."""
    ev = Evaluator()
    for pb, ps, gb, gc in pairs:
        ev.get_exp_metrics(pb, ps, torch.as_tensor(gb), gc)
    try:
        _, df_ap = get_full_conf_results(ev.metrics)
        return float(df_ap['ap_total'].iloc[0])
    except Exception:
        return float('nan')


def is_ring_geom(boxes, mask, frac=0.70):
    """Ring == box spans >= `frac` of the VALID chi rows at its own radius.

    The validated criterion; 41.h5 does not populate is_ring at all (see the eval-dataset
    memory), so geometry is the only trustworthy label on both gates.
    """
    H, W = mask.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes):
        xc = int(np.clip((x0 + x1) / 2, 0, W - 1))
        valid = max(int(mask[:, xc].sum()), 1)
        out.append((y1 - y0) >= frac * valid)
    return np.array(out, bool)


# ----------------------------------------------------------------------------------- caching
_CFG = Config(); _CFG.PREPROCESSING_POLAR_SHAPE = list(POLAR)


def cache_frames(onnx_path, name, path):
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    cfg = Config(); cfg.PREPROCESSING_POLAR_SHAPE = list(POLAR); cfg.INPUT_DATASET = path
    cfg.POSTPROCESSING_SCORE = DEF_SCORE; cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=3, load_labels=True)
          if detect_dataset_type(path) == 'pygid' else
          H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=3))
    frames = []
    try:
        for i, ic in enumerate(ds.iter_images()):
            x = np.asarray(ic.converted_polar_image, np.float32).reshape(1, -1, *POLAR)[:, :1]
            logits, boxes = sess.run(None, {iname: x})
            frames.append(dict(
                i=i,
                logits=logits, boxes=boxes,
                mask=np.asarray(ic.converted_mask).reshape(*POLAR).astype(bool),
                gt=np.asarray(ic.polar_labels.boxes, np.float32),
                gtconf=np.asarray(ic.polar_labels.confidences, np.float32),
            ))
            if MAX_FRAMES and len(frames) >= MAX_FRAMES:
                break
    finally:
        close = getattr(ds, 'close', None)   # PyGIDDataset spawns a NON-daemon write worker
        if callable(close):
            close()
    return frames


# ------------------------------------------------------------------------------------ report
def run(onnx_path):
    tag = os.path.basename(onnx_path).replace('.onnx', '')
    print(f'\n{"="*78}\nMODEL {tag}\n{"="*78}', flush=True)
    cache = {n: cache_frames(onnx_path, n, p) for n, p in DATASETS.items()}

    # --- consistency: our replay at default knobs must equal the shipped postprocessing ----
    for name, frames in cache.items():
        f = frames[0]
        cfg = Config(); cfg.PREPROCESSING_POLAR_SHAPE = list(POLAR)
        cfg.POSTPROCESSING_SCORE = DEF_SCORE; cfg.POSTPROCESSING_CLASSAWARE_NMS = True
        cfg.POSTPROCESSING_NMSIOU_RING, cfg.POSTPROCESSING_NMSIOU_SEG = DEF_RING, DEF_SEG
        ref = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), [f['logits'], f['boxes']]))
        mine, _, _, _ = postprocess(f['logits'], f['boxes'])
        d = float((torch.sort(ref.boxes, 0).values - torch.sort(mine, 0).values).abs().max()) if len(ref.boxes) == len(mine) else float('nan')
        print(f'[replay check] {name}: shipped {len(ref.boxes)} boxes, replay {len(mine)}, max|diff| {d:.3e}')

    # ------------------------------------------------------------------ A. premise: crowding
    print('\n--- A. per-frame AP vs crowding -------------------------------------------')
    rows = []
    for name, frames in cache.items():
        for f in frames:
            pb, ps, pl, _ = postprocess(f['logits'], f['boxes'])
            ap = ap_of([(pb, ps, f['gt'], f['gtconf'])])
            gr = is_ring_geom(f['gt'], f['mask'])
            rows.append(dict(model=tag, set=name, frame=f['i'], gt=len(f['gt']),
                             gt_rings=int(gr.sum()), gt_segs=int((~gr).sum()),
                             pred=len(pb), ap=ap))
    csv_path = os.path.join(OUT, f'postproc_frames_{tag}.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'per-frame table -> {csv_path}')
    for name in DATASETS:
        rs = [r for r in rows if r['set'] == name and np.isfinite(r['ap'])]
        if not rs:
            continue
        print(f'  {name}: binned by GT box count')
        print(f'    {"bin":>12s} {"frames":>7s} {"rings/f":>8s} {"segs/f":>7s} {"pred/f":>7s} {"ap":>7s}')
        for lo, hi in ((0, 25), (25, 60), (60, 120), (120, 10**6)):
            b = [r for r in rs if lo <= r['gt'] < hi]
            if not b:
                continue
            lbl = f'{lo}-{hi}' if hi < 10**6 else f'{lo}+'
            print(f'    {lbl:>12s} {len(b):7d} '
                  f'{np.mean([r["gt_rings"] for r in b]):8.1f} {np.mean([r["gt_segs"] for r in b]):7.1f} '
                  f'{np.mean([r["pred"] for r in b]):7.1f} {np.mean([r["ap"] for r in b]):7.3f}')
        mixed = [r for r in rs if r['gt_rings'] >= 5 and r['gt_segs'] >= 20]
        other = [r for r in rs if not (r['gt_rings'] >= 5 and r['gt_segs'] >= 20)]
        if mixed and other:
            print(f'    MIXED (>=5 rings and >=20 segs): {len(mixed)} frames ap {np.mean([r["ap"] for r in mixed]):.3f}'
                  f'  |  rest: {len(other)} frames ap {np.mean([r["ap"] for r in other]):.3f}')

    # ------------------------------------------------------------------ B. top-k cap binding
    print('\n--- B. does the top-225 cap truncate real candidates? ----------------------')
    print(f'  {"set":9s} {"frames":>7s} {"225th score":>12s} {"n>thr of 225":>13s} {"cap-bound":>10s} {"kept":>7s}')
    for name, frames in cache.items():
        last, above, kept, bound = [], [], [], 0
        for f in frames:
            prob = torch.from_numpy(f['logits']).sigmoid().view(-1)
            v, _ = torch.topk(prob, min(DEF_SELECT, prob.numel()))
            last.append(float(v[-1])); above.append(int((v > DEF_SCORE).sum()))
            if float(v[-1]) > DEF_SCORE:
                bound += 1                       # the 225th slot was still above threshold
            pb, _, _, _ = postprocess(f['logits'], f['boxes'])
            kept.append(len(pb))
        print(f'  {name:9s} {len(frames):7d} {np.mean(last):12.3f} {np.mean(above):13.1f} '
              f'{f"{bound}/{len(frames)}":>10s} {np.mean(kept):7.1f}')

    # -------------------------------------------------------------- C. cross-class duplicates
    print('\n--- C. same query selected as BOTH ring and segment -------------------------')
    print(f'  {"set":9s} {"kept":>7s} {"dup pairs":>10s} {"% kept":>8s} {"ap default":>11s} {"ap dedupe":>10s}')
    for name, frames in cache.items():
        dup = tot = 0
        pd_, pdd = [], []
        for f in frames:
            pb, ps, pl, q = postprocess(f['logits'], f['boxes'])
            qs = q.tolist()
            dup += sum(1 for x in set(qs) if qs.count(x) > 1)
            tot += len(qs)
            pd_.append((pb, ps, f['gt'], f['gtconf']))
            b2, s2, _, _ = postprocess(f['logits'], f['boxes'], dedupe=True)
            pdd.append((b2, s2, f['gt'], f['gtconf']))
        print(f'  {name:9s} {tot:7d} {dup:10d} {100*dup/max(tot,1):7.1f}% '
              f'{ap_of(pd_):11.4f} {ap_of(pdd):10.4f}')

    # ------------------------------------------------------- D. angular extent / class sanity
    print('\n--- D. angular extent: predicted vs GT, and head-class vs geometry ----------')
    print(f'  {"set":9s} {"GT y-ext":>9s} {"pred y-ext":>11s} {"GT ring%":>9s} {"head ring%":>11s} {"geom ring%":>11s} {"disagree":>9s}')
    for name, frames in cache.items():
        gy, py, gr, hr, geo, dis, n = [], [], [], [], [], 0, 0
        for f in frames:
            pb, ps, pl, _ = postprocess(f['logits'], f['boxes'])
            g = np.asarray(f['gt'], np.float32)
            if len(g):
                gy += list(g[:, 3] - g[:, 1]); gr += list(is_ring_geom(g, f['mask']))
            if len(pb):
                b = pb.numpy()
                py += list(b[:, 3] - b[:, 1])
                gm = b[:, 3] - b[:, 1] >= POLAR[0] * 0.35     # the old util/nms.py heuristic
                hd = pl.numpy() == 1
                geo += list(gm); hr += list(hd)
                dis += int((gm != hd).sum()); n += len(b)
        print(f'  {name:9s} {np.mean(gy):9.1f} {np.mean(py):11.1f} {100*np.mean(gr):8.1f}% '
              f'{100*np.mean(hr):10.1f}% {100*np.mean(geo):10.1f}% {100*dis/max(n,1):8.1f}%')

    # ------------------------------------------------------------------------- E. knob sweep
    print('\n--- E. postprocessing sweep (ap_total) -------------------------------------')
    grid = []
    for ns in (225, 450, 900):
        grid.append(dict(num_select=ns))
    for ir in (0.05, 0.1, 0.2, 0.3, 0.5):
        grid.append(dict(iou_ring=ir))
    for isg in (0.3, 0.4, 0.5, 0.6):
        grid.append(dict(iou_seg=isg))
    grid.append(dict(classaware=False))
    grid.append(dict(dedupe=True))
    grid.append(dict(dedupe=True, num_select=450))
    grid.append(dict(dedupe=True, iou_ring=0.3))
    grid.append(dict(dedupe=True, num_select=450, iou_ring=0.3))

    base = {n: ap_of([(*postprocess(f['logits'], f['boxes'])[:2], f['gt'], f['gtconf'])
                      for f in frames]) for n, frames in cache.items()}
    print(f'  {"config":42s} ' + ' '.join(f'{n:>18s}' for n in DATASETS))
    print(f'  {"DEFAULT (225, ring .1, seg .4)":42s} ' +
          ' '.join(f'{base[n]:18.4f}' for n in DATASETS))
    for kw in grid:
        label = ', '.join(f'{k}={v}' for k, v in kw.items())
        cells = []
        for n, frames in cache.items():
            a = ap_of([(*postprocess(f['logits'], f['boxes'], **kw)[:2], f['gt'], f['gtconf'])
                       for f in frames])
            cells.append(f'{a:11.4f} ({a-base[n]:+.4f})')
        print(f'  {label:42s} ' + ' '.join(f'{c:>18s}' for c in cells), flush=True)


if __name__ == '__main__':
    for m in MODELS:
        run(m)
