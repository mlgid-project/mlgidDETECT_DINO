"""Why does raising num_select LOWER ap_total on the physics models?

Two candidate mechanisms, from reading util/evaluation.py + util/matchers.py:

  (R) REASSIGNMENT. The evaluator matches GT to predictions with `linear_sum_assignment`
      over |dq| (QMatcher, min_iou=0.1, thresh=10) and NEVER looks at the scores. Admitting
      more candidates lets the solver hand a GT to a geometrically closer but LOWER-scored
      box; the box that used to hold that GT becomes a false positive AT A HIGH SCORE, which
      hurts precision at low recall -- the most expensive part of the curve.
  (O) OVER-DETECTION. The extra boxes are simply new false positives. Being lower-scored than
      everything already kept, they land in the tail, where they should cost almost nothing
      (zero recall width, and _interpolate_precisions does not propagate them backwards).

Decides between them off the same cached forward pass as numselect_sweep.py:
  - recall (GT matched) at each cap -- does admitting more boxes find more real peaks?
  - per-GT matched score at 225 vs the higher cap: how many GT get REASSIGNED to a
    lower-scored prediction, and by how much.
  - false positives above 0.5 / 0.3 / 0.1 at each cap -- (R) predicts high-score FPs appear,
    (O) predicts they only pile up near 0.1.

  python capsplit.py name=/path/to/checkpoint.pth [...]

RESULT 2026-09-15: OVER-DETECTION dominates. physics4 on 41, cap 225->450 adds
+135.8 FP/frame but only +1.0 above score 0.5; recall RISES 0.856->0.887. Reassignment is real
but second-order: 98/1437 GT (6.8%) demoted, median drop 0.283, vs ssl1 17/1460 and lr4e5 3/1492.
Full record: MODIFICATIONS.md section M.
"""
import os, sys
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, cache_raw, DATASETS, _P
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CAPS = [225, 450, 900]
FP_LEVELS = [0.5, 0.3, 0.1]
matcher = get_matcher('q', min_iou=0.1)   # exactly what Evaluator(match_criterion='q', 0.1) builds


def analyse(cache, config, num_select):
    """Per frame: which GT are matched, to what score, and the FP score distribution."""
    per_gt = []        # list of dicts gt_index -> matched score, one per frame
    n_gt = n_tp = 0
    fp_scores, tp_scores = [], []
    for logits, boxes, gt_boxes, gt_conf in cache:
        c = onnx_to_xyxy(config, _P(), [logits, boxes], num_select=num_select)
        c = filter_boxes(config, c)
        tgt = torch.tensor(np.asarray(gt_boxes)).float()
        if len(c.boxes) == 0 or len(tgt) == 0:
            per_gt.append({}); n_gt += len(tgt); continue
        _, row_ind, col_ind = matcher(tgt, c.boxes)
        sc = np.asarray(c.scores)
        d = {int(g): float(sc[p]) for g, p in zip(row_ind, col_ind)}
        per_gt.append(d)
        n_gt += len(tgt); n_tp += len(row_ind)
        tp_scores += [sc[p] for p in col_ind]
        fp_mask = np.ones(len(sc), bool); fp_mask[np.asarray(col_ind, int)] = False
        fp_scores += list(sc[fp_mask])
    return dict(per_gt=per_gt, n_gt=n_gt, n_tp=n_tp,
                tp=np.array(tp_scores), fp=np.array(fp_scores), n_frames=len(cache))


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    for name, ckpt in targets:
        model, args, epoch = load_model(ckpt)
        print(f'\n########## {name} (epoch {epoch})', flush=True)
        for dname, dpath in DATASETS.items():
            cache, config = cache_raw(model, args, dpath)
            res = {ns: analyse(cache, config, ns) for ns in CAPS}
            nf = res[225]['n_frames']
            print(f'\n=== {dname}: {nf} frames, {res[225]["n_gt"]/nf:.1f} GT/frame', flush=True)
            print(f'{"cap":>6} {"recall":>8} {"TP/fr":>8} {"FP/fr":>8} |'
                  f'{"  FP>0.5":>9}{"  FP>0.3":>9}{"  FP>0.1":>9} | {"medTP":>7} {"medFP":>7}')
            for ns in CAPS:
                r = res[ns]
                fps = [(r['fp'] > L).sum() / nf for L in FP_LEVELS]
                print(f'{ns:>6} {r["n_tp"]/r["n_gt"]:>8.4f} {r["n_tp"]/nf:>8.1f} '
                      f'{len(r["fp"])/nf:>8.1f} |{fps[0]:>9.1f}{fps[1]:>9.1f}{fps[2]:>9.1f} | '
                      f'{np.median(r["tp"]):>7.3f} {np.median(r["fp"]):>7.3f}', flush=True)
            # (R): per-GT score change 225 -> higher cap
            for ns in CAPS[1:]:
                lost = gained = same = down = up = 0
                drops = []
                for a, b in zip(res[225]['per_gt'], res[ns]['per_gt']):
                    for g in set(a) | set(b):
                        if g in a and g not in b:   lost += 1
                        elif g in b and g not in a: gained += 1
                        elif abs(a[g] - b[g]) < 1e-9: same += 1
                        elif b[g] < a[g]: down += 1; drops.append(a[g] - b[g])
                        else: up += 1
                tot = same + down + up
                print(f'  225 -> {ns}: of {tot} GT matched at BOTH caps, {same} keep the same '
                      f'prediction, {down} move to a LOWER-scored one '
                      f'(median drop {np.median(drops) if drops else 0:.3f}), {up} to a higher. '
                      f'{gained} GT newly matched, {lost} lost.', flush=True)
        del model; torch.cuda.empty_cache()
