"""Are the physics models' extra false positives DUPLICATES on real peaks, or noise firings?

Splits every FP at the deployed operating point (top-225, NMS, score>0.1) by IoU with the
nearest GT box:
  dup   IoU > 0.3   -- a second box on a peak that is already detected: a dedup/NMS failure
  near  0 < IoU     -- overlapping something real but badly placed
  bg    IoU == 0    -- fired on empty detector: a background/noise firing
Also reports the score distribution in each bucket, because only high-score FPs cost AP.

RESULT 2026-09-15: the excess is a DUPLICATE SPRAY, not noise. organic near-miss FPs
(0<IoU<=0.3) physics4 53.2/frame vs lr4e5 10.2; background firings comparable, 39.2 vs 27.6.
Full record: MODIFICATIONS.md section M.
"""
import os, sys
import numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, cache_raw, DATASETS, _P
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from torchvision.ops import box_iou

matcher = get_matcher('q', min_iou=0.1)

for name, ckpt in [a.split('=', 1) for a in sys.argv[1:]]:
    model, args, epoch = load_model(ckpt)
    print(f'\n##### {name} (ep {epoch})', flush=True)
    for dname, dpath in DATASETS.items():
        cache, config = cache_raw(model, args, dpath)
        buckets = {k: [] for k in ('dup', 'near', 'bg')}
        ntp = nf = 0
        for logits, boxes, gt_boxes, _ in cache:
            c = filter_boxes(config, onnx_to_xyxy(config, _P(), [logits, boxes], num_select=225))
            tgt = torch.tensor(np.asarray(gt_boxes)).float()
            _, row_ind, col_ind = matcher(tgt, c.boxes)
            ntp += len(row_ind); nf += 1
            sc = np.asarray(c.scores)
            fp = np.ones(len(sc), bool); fp[np.asarray(col_ind, int)] = False
            if not fp.any():
                continue
            iou = box_iou(c.boxes[torch.from_numpy(fp)], tgt).max(1).values.numpy()
            s = sc[fp]
            buckets['dup'] += list(s[iou > 0.3])
            buckets['near'] += list(s[(iou > 0) & (iou <= 0.3)])
            buckets['bg'] += list(s[iou == 0])
        tot = sum(len(v) for v in buckets.values())
        print(f'  {dname}: {ntp/nf:.1f} TP/frame, {tot/nf:.1f} FP/frame', flush=True)
        for k, v in buckets.items():
            v = np.array(v)
            if not len(v):
                print(f'    {k:>5}:   0'); continue
            print(f'    {k:>5}: {len(v)/nf:6.1f}/frame ({100*len(v)/tot:4.1f}%)  '
                  f'median score {np.median(v):.3f}  >0.5: {(v>0.5).sum()/nf:5.1f}/frame', flush=True)
    del model; torch.cuda.empty_cache()
