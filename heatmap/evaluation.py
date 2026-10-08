"""Shared eval for ssl1 (DINO) and the heatmap detector -- one code path for both.

Per frame we keep ALL detections (top-K cap and class-aware NMS only, NO score floor: the evaluator builds its own
precision-recall curve over the scores; a floor would cut the tail of the curve), then score them identically:
  ap_total (Evaluator, same as --eval), recall / precision / FP count at a score threshold,
  and recall bucketed by each GT segment's nearest-neighbour distance.
Matching = the repo's q-matcher (min_iou 0.1), as in gap41_split.py.
"""
import numpy as np
import torch

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results
from util.matchers import get_matcher

import os
CUR = os.environ.get('HM_DATA_DIR', '/mnt/lustre/work/schreiber/szb389/datasets')   # dir holding organic_labeled.h5 + 41.h5
DATASETS = {'organic': f'{CUR}/organic_labeled.h5', '41': f'{CUR}/41.h5'}
POLAR = (512, 1024)
matcher = get_matcher('q', min_iou=0.1)


class _P:
    pass


def make_config(path):
    c = Config()
    c.EVAL_EPOCH = '0'; c.EVAL_OUTPUT_FOLDER = '/tmp'
    c.INPUT_DATASET = path; c.PREPROCESSING_POLAR_SHAPE = list(POLAR)
    # NO score floor: AP is computed from the precision-recall curve over ALL detection scores (user decision 2026-10-08).
    # Score cuts belong only to the operating points (recall/precision at >0.1, >0.3) and to the drawn boxes.
    c.POSTPROCESSING_SCORE = 0.0; c.POSTPROCESSING_CLASSAWARE_NMS = True
    return c


def iter_frames(path):
    cfg = make_config(path)
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=True)
          if detect_dataset_type(path) == 'pygid' else
          H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    try:
        for ic in ds.iter_images():
            yield cfg, ic
    finally:
        if callable(getattr(ds, 'close', None)):
            ds.close()


def frame_inputs(ic, device, num_channels=1):
    img = torch.tensor(ic.converted_polar_image[:, 0]).unsqueeze(0).to(device)
    return img.repeat(1, num_channels, 1, 1)


def gt_of(ic):
    return dict(gt=np.asarray(ic.polar_labels.boxes, np.float32),
                gtconf=np.asarray(ic.polar_labels.confidences, np.float32),
                mask=np.asarray(ic.converted_mask).reshape(*POLAR).astype(bool))


def dino_dets(cfg, outputs):
    """DINO path: top-225 + class-aware NMS, no score floor (the --eval path had score>0.1)."""
    c = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), [outputs['pred_logits'].detach().cpu().numpy(),
                                                   outputs['pred_boxes'].detach().cpu().numpy()]))
    return np.asarray(c.boxes, np.float32).reshape(-1, 4), np.asarray(c.scores, np.float32)


def heatmap_dets(cfg, per_image, use_nms):
    boxes, scores, cls = per_image
    boxes, scores, cls = boxes.cpu(), scores.cpu(), cls.cpu()
    if use_nms:                                  # shared filter_boxes (class-aware NMS) for an A/B
        c = _P(); c.boxes, c.scores, c.pred_labels = boxes, scores, cls
        c = filter_boxes(cfg, c)
        return np.asarray(c.boxes, np.float32).reshape(-1, 4), np.asarray(c.scores, np.float32)
    k = scores > cfg.POSTPROCESSING_SCORE        # native: peak picking IS the NMS (floor 0 = keep every candidate)
    return boxes[k].numpy().astype(np.float32), scores[k].numpy().astype(np.float32)


def is_ring_geom(boxes, mask, frac=0.70):
    H, W = mask.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes):
        xc = int(np.clip((x0 + x1) / 2, 0, W - 1))
        out.append((y1 - y0) >= frac * max(int(mask[:, xc].sum()), 1))
    return np.array(out, bool)


def nn_distances(gt, mask):
    """For each GT: Euclid centre distance to the nearest other non-ring GT, and chi-gap (|dy|) to the
    nearest same-radius one (|dx| <= max half-width). Rings get NaN (not in the buckets)."""
    n = len(gt)
    eu = np.full(n, np.nan); chi = np.full(n, np.nan)
    if n == 0:
        return eu, chi, np.zeros(0, bool)
    ring = is_ring_geom(gt, mask)
    cx = (gt[:, 0] + gt[:, 2]) / 2; cy = (gt[:, 1] + gt[:, 3]) / 2; w = gt[:, 2] - gt[:, 0]
    seg = np.where(~ring)[0]
    for i in seg:
        o = seg[seg != i]
        if len(o) == 0:
            eu[i] = chi[i] = np.inf; continue
        dx = np.abs(cx[o] - cx[i]); dy = np.abs(cy[o] - cy[i])
        eu[i] = np.sqrt(dx ** 2 + dy ** 2).min()
        same = dx <= np.maximum(w[o], w[i]) / 2
        chi[i] = dy[same].min() if same.any() else np.inf
    return eu, chi, ring


BUCKETS = [('<5', 0, 5), ('5-10', 5, 10), ('>10', 10, np.inf)]


def evaluate_dets(dets, gts, cfg, thr_list=(0.1, 0.3)):
    """dets: list of (boxes, scores); gts: list of gt dicts. Returns a result dict."""
    ev = Evaluator()
    for (b, s), g in zip(dets, gts):
        ev.get_exp_metrics(torch.tensor(b), torch.tensor(s), torch.tensor(g['gt']), g['gtconf'])
    _, df2 = get_full_conf_results(ev.metrics)
    res = dict(ap=float(df2['ap_total'].values[0]), thr={})
    for thr in thr_list:
        n_gt = n_tp = n_pred = 0
        bk = {k: [0, 0] for k, _, _ in BUCKETS}; bkc = {k: [0, 0] for k, _, _ in BUCKETS}
        seg_gt = seg_tp = ring_gt = ring_tp = 0
        for (b, s), g in zip(dets, gts):
            k = s > thr if len(s) else np.zeros(0, bool)
            b, s = b[k], s[k]
            gt = g['gt']
            eu, chi, ring = nn_distances(gt, g['mask'])
            hit = np.zeros(len(gt), bool)
            if len(b) and len(gt):
                _, ri, _ = matcher(torch.tensor(gt).float(), torch.tensor(b).float())
                hit[np.asarray(ri, int)] = True
            n_gt += len(gt); n_tp += int(hit.sum()); n_pred += len(b)
            if len(gt):
                seg_gt += int((~ring).sum()); seg_tp += int(hit[~ring].sum())
                ring_gt += int(ring.sum()); ring_tp += int(hit[ring].sum())
            for name, lo, hi in BUCKETS:
                m = (eu >= lo) & (eu < hi); bk[name][0] += int(m.sum()); bk[name][1] += int(hit[m].sum())
                m = (chi >= lo) & (chi < hi); bkc[name][0] += int(m.sum()); bkc[name][1] += int(hit[m].sum())
        res['thr'][thr] = dict(n_gt=n_gt, recall=n_tp / max(n_gt, 1), precision=n_tp / max(n_pred, 1),
                               n_pred=n_pred, fp=n_pred - n_tp,
                               seg_recall=seg_tp / max(seg_gt, 1), ring_recall=ring_tp / max(ring_gt, 1),
                               euclid={k: (v[0], v[1] / max(v[0], 1)) for k, v in bk.items()},
                               chigap={k: (v[0], v[1] / max(v[0], 1)) for k, v in bkc.items()})
    return res


def format_result(name, ds, r):
    lines = [f'  [{name}] {ds}: ap_total {r["ap"]:.4f}']
    for thr, t in r['thr'].items():
        eu = ' '.join(f'{k}:{v[1]:.3f}(n={v[0]})' for k, v in t['euclid'].items())
        ch = ' '.join(f'{k}:{v[1]:.3f}(n={v[0]})' for k, v in t['chigap'].items())
        lines.append(f'    score>{thr}: recall {t["recall"]:.3f} (seg {t["seg_recall"]:.3f}, ring {t["ring_recall"]:.3f}) '
                     f'precision {t["precision"]:.3f} preds {t["n_pred"]} FP {t["fp"]}')
        lines.append(f'      recall by NN Euclid dist  {eu}')
        lines.append(f'      recall by same-radius chi-gap  {ch}')
    return '\n'.join(lines)
