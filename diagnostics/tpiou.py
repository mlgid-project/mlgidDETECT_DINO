"""Localisation quality of the TPs themselves: median IoU of matched pairs on real data.
RESULT 2026-09-15: the physics model localises its TPs BETTER, not worse -- organic
median matched IoU 0.394 (physics4) vs 0.343 (lr4e5). Rules out sloppy regression as the cause
of the near-miss spray. Full record: MODIFICATIONS.md section M.
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
    print(f'##### {name} (ep {epoch})', flush=True)
    for dname, dpath in DATASETS.items():
        cache, config = cache_raw(model, args, dpath)
        ious = []
        for logits, boxes, gt_boxes, _ in cache:
            c = filter_boxes(config, onnx_to_xyxy(config, _P(), [logits, boxes], num_select=225))
            tgt = torch.tensor(np.asarray(gt_boxes)).float()
            m, row_ind, col_ind = matcher(tgt, c.boxes)
            if not len(row_ind): continue
            ious += list(box_iou(tgt[row_ind], c.boxes[col_ind]).diagonal().numpy())
        ious = np.array(ious)
        print(f'  {dname}: matched IoU  p25 {np.percentile(ious,25):.3f}  '
              f'median {np.median(ious):.3f}  p75 {np.percentile(ious,75):.3f}  '
              f'frac>0.5 {(ious>0.5).mean():.3f}', flush=True)
    del model; torch.cuda.empty_cache()
