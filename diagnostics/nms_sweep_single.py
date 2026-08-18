"""
Does this DETR need NMS at all?  — SINGLE MODEL (ssl1), no ensemble.

Follows MODIFICATIONS.md S/T: the recall gap is peak SEPARATION along chi (84.5% of misses sit
within 8 q-px of a detected peak; median chi-gap 3.9 px against ~8.5 px-tall boxes). NMS encodes
"high overlap => duplicate", which is false for genuinely adjacent peaks: two 8.5 px boxes at
separation d have IoU (8.5-d)/(8.5+d), crossing the deployed seg threshold 0.4 at d ~ 3.6 px. So
NMS cannot be tuned to keep real close pairs and drop real duplicates — it can only trade one for
the other.

DETR-family models are trained with one-to-one Hungarian matching precisely so that duplicate
suppression is LEARNED, not post-hoc; DINO's reference inference runs no NMS. This probe tests
whether NMS is earning its keep here. `diagnostics/sweep_nms.py` previously swept only DOWNWARD
(0.4 -> 0.1, AP flat). The loosening direction — up to fully off — has never been tested.

Method: cache each frame's pre-NMS top-225 ONCE, then re-run NMS at every setting on the cache.
Reports, per setting:
  - ap_total / ap_high / ap_med / ap_low via the SAME Evaluator as the --eval gate
  - recall / precision / detections-per-frame at the operating point
  - **recall stratified by chi-gap to the nearest same-q labeled peak** -- the population this is
    all about, and the pre-registered gate. AP over all peaks will barely move; the effect lives
    in the tight-sibling stratum.
Also counts top-k duplicate queries: onnx_to_xyxy selects over (query x class) flattened, so ONE
query can be selected twice as both classes, and class-aware NMS (per class) cannot remove that.

GPU, ~5 min. See tmp_diag/run_nms_sweep.sbatch.
"""
import os, sys, json
import numpy as np
import torch
from torchvision.ops import nms

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import box_cxcywh_to_xyxy
from util.matchers import get_matcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, DSETS, OUT

CKPT = CKPT_A                 # dino_ssl1 — best SINGLE model. No ensemble (user directive).
NUM_SELECT = 225              # must match util.postprocessing.onnx_to_xyxy
SCORE_FLOOR = 0.1             # mirrors main.evaluate_giwaxs_ap / ensemble_eval AP path
ST = 0.30                     # operating point for recall/precision, as in probes S and T
QTOL = 8.0                    # same-q tolerance defining a sibling cluster
GAP_BINS = [(0, 5), (5, 10), (10, 20), (20, 33), (33, 1e9)]
SETTINGS = ([('seg_iou=%.2f' % s, s, 0.1) for s in
             (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
            + [('seg OFF (ring 0.1)', 1.01, 0.1), ('NMS FULLY OFF', 1.01, 1.01)])


def topk_like_deployed(config, out):
    """Byte-identical replica of onnx_to_xyxy, but also returns the query index per selection."""
    logits = out['pred_logits'].detach().cpu()
    bbox = out['pred_boxes'].detach().cpu()
    prob = logits.sigmoid()
    ncls = prob.shape[2]
    k = min(NUM_SELECT, prob.shape[1] * ncls)
    vals, idx = torch.topk(prob.view(logits.shape[0], -1), k, dim=1)
    qidx = idx[0] // ncls
    labels = idx[0] % ncls
    boxes = box_cxcywh_to_xyxy(config, bbox)[qidx]
    return boxes, vals[0], labels, qidx


def sibling_gap(gt):
    """For each GT box, the chi-distance to the nearest labeled peak at the same q (<QTOL px).
    inf when the peak has no same-q sibling."""
    n = len(gt)
    if n == 0:
        return np.zeros(0)
    q = ((gt[:, 0] + gt[:, 2]) / 2).numpy()
    c = ((gt[:, 1] + gt[:, 3]) / 2).numpy()
    out = np.full(n, np.inf)
    for i in range(n):
        m = (np.abs(q - q[i]) < QTOL)
        m[i] = False
        if m.any():
            out[i] = np.min(np.abs(c[m] - c[i]))
    return out


def collect(tag, path, model, a, device):
    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = SCORE_FLOOR
    config.INPUT_DATASET = path
    ds = (PyGIDDataset(config, path=path, preprocess_func=standard_preprocessing,
                       buffer_size=5, load_labels=True)
          if detect_dataset_type(path) == 'pygid' else
          H5GIWAXSDataset(config, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    frames = []
    dup = tot = 0
    with torch.no_grad():
        for gc in ds.iter_images():
            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device)
            img = img.repeat(1, a.num_channels, 1, 1)
            b, s, l, qi = topk_like_deployed(config, model(img))
            dup += len(qi) - len(torch.unique(qi)); tot += len(qi)
            L = gc.polar_labels
            gt = (torch.tensor(np.array(L.boxes), dtype=torch.float32)
                  if len(L.boxes) else torch.zeros((0, 4)))
            # NOTE: Evaluator.get_full_metrics does torch.from_numpy(intensities) -> must stay numpy
            conf = (np.asarray(L.confidences, dtype=np.float32)
                    if len(L.confidences) else np.zeros(len(gt), dtype=np.float32))
            frames.append(dict(b=b, s=s, l=l, gt=gt, conf=conf, gap=sibling_gap(gt)))
            print(f"  [{tag}] frame {len(frames)}: GT={len(gt)}", flush=True)
    if hasattr(ds, 'close'):
        ds.close()
    print(f"  [{tag}] top-k duplicate queries (same query selected as BOTH classes): "
          f"{dup}/{tot} = {dup/max(tot,1):.1%} of selections", flush=True)
    return frames, dup / max(tot, 1)


def apply_nms(f, seg_iou, ring_iou):
    keep = []
    for cls, thr in ((1, ring_iou), (0, seg_iou)):
        idx = (f['l'] == cls).nonzero(as_tuple=True)[0]
        if idx.numel():
            keep.append(idx[nms(f['b'][idx], f['s'][idx], thr)])
    if not keep:
        return f['b'][:0], f['s'][:0]
    k = torch.cat(keep)
    b, s = f['b'][k], f['s'][k]
    m = s > SCORE_FLOOR
    return b[m], s[m]


def evaluate(frames, seg_iou, ring_iou):
    ev = Evaluator()
    matcher = get_matcher('q', min_iou=0.1)
    ndet = tp = fp = ngt = 0
    hit_gap = {g: [0, 0] for g in GAP_BINS}          # (matched, total)
    for f in frames:
        b, s = apply_nms(f, seg_iou, ring_iou)
        ev.get_exp_metrics(b, s, f['gt'], f['conf'])
        m = s > ST
        bo = b[m]
        ndet += len(bo); ngt += len(f['gt'])
        row = col = np.array([], int)
        if len(f['gt']) and len(bo):
            try:
                _, row, col = matcher(f['gt'], bo)
            except IndexError:
                pass
        mset = set(row.tolist())
        tp += len(mset); fp += len(bo) - len(set(col.tolist()))
        for i in range(len(f['gt'])):
            g = f['gap'][i]
            for lo, hi in GAP_BINS:
                if lo <= g < hi:
                    hit_gap[(lo, hi)][1] += 1
                    if i in mset:
                        hit_gap[(lo, hi)][0] += 1
                    break
    _, df_ap = get_full_conf_results(ev.metrics, name='x')   # returns (df, df_ap) — AP is 2nd
    r = dict(ap_total=float(df_ap['ap_total'].iloc[0]), ap_high=float(df_ap['ap_high'].iloc[0]),
             ap_med=float(df_ap['ap_med'].iloc[0]), ap_low=float(df_ap['ap_low'].iloc[0]),
             recall=tp / max(ngt, 1), precision=tp / max(tp + fp, 1),
             det_per_frame=ndet / max(len(frames), 1))
    for g in GAP_BINS:
        h, n = hit_gap[g]
        r[f'gap_{g[0]}'] = (h / n) if n else float('nan')
        r[f'ngap_{g[0]}'] = n
    return r


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  SINGLE MODEL = {CKPT}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT, device)

    summary = {}
    for tag, path in DSETS:
        print(f"\n########## {tag} ##########", flush=True)
        frames, dupfrac = collect(tag, path, model, a, device)
        gaps = np.concatenate([f['gap'] for f in frames])
        print(f"\n{'='*112}\n{tag.upper()}  ssl1 single model, {len(frames)} frames, "
              f"{len(gaps)} peaks   ({np.mean(np.isfinite(gaps)):.1%} have a same-q sibling)\n{'='*112}")
        hdr = (f"  {'setting':20s} {'ap_total':>9s} {'ap_high':>8s} {'recall':>7s} {'prec':>6s} "
               f"{'det/fr':>7s} | recall by chi-gap to nearest same-q peak")
        print(hdr)
        print(f"  {'':20s} {'':>9s} {'':>8s} {'':>7s} {'':>6s} {'':>7s} | " +
              " ".join(f"{('<%d' % hi) if hi < 1e8 else '>=33':>7s}" for lo, hi in GAP_BINS))
        rows = {}
        for name, si, ri in SETTINGS:
            r = evaluate(frames, si, ri)
            rows[name] = r
            print(f"  {name:20s} {r['ap_total']:9.4f} {r['ap_high']:8.4f} {r['recall']:7.3f} "
                  f"{r['precision']:6.3f} {r['det_per_frame']:7.1f} | " +
                  " ".join(f"{r[f'gap_{lo}']:7.3f}" for lo, hi in GAP_BINS), flush=True)
        base = rows['seg_iou=0.40']
        print(f"\n  counts per chi-gap stratum: " +
              "  ".join(f"{('<%d' % hi) if hi < 1e8 else '>=33'}: n={base[f'ngap_{lo}']}"
                        for lo, hi in GAP_BINS))
        print(f"\n  DELTA vs deployed seg_iou=0.40:")
        for name in ('seg_iou=0.90', 'seg OFF (ring 0.1)', 'NMS FULLY OFF'):
            r = rows[name]
            print(f"    {name:20s} ap_total {r['ap_total']-base['ap_total']:+.4f}   "
                  f"recall {r['recall']-base['recall']:+.3f}   "
                  f"prec {r['precision']-base['precision']:+.3f}   "
                  f"tight-pair recall (<5px) {r['gap_0']-base['gap_0']:+.3f}")
        summary[tag] = dict(rows=rows, dup_frac=dupfrac,
                            sibling_frac=float(np.mean(np.isfinite(gaps))))
    json.dump(summary, open(os.path.join(OUT, 'nms_sweep_single.json'), 'w'), indent=2, default=str)
    print("\nwrote nms_sweep_single.json")
    print("PROBE DONE")


if __name__ == '__main__':
    main()
