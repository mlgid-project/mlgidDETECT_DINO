"""Can NMS kill the physics models' spray of near-miss boxes? Deployed: ring 0.1 / seg 0.4.
RESULT 2026-09-15: segment IoU 0.4->0.10 is worth physics4 +0.042 on 41 and +0.032
on organic, while lr4e5 moves <=0.002 and ssl1 <=0.007 -- the duplicates are real and suppressible.
NOT shipped: still below baseline on both gates (0.6827 vs 0.7636 on 41, 0.6124 vs 0.6222 on
organic), and it would change the deployed mlgidDETECT path. Full record: MODIFICATIONS.md M.
"""
import os, sys, itertools
import numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, cache_raw, DATASETS, _P, replay
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results

RING = [0.1]
SEG  = [0.4, 0.2, 0.1, 0.05, 0.02]
for name, ckpt in [a.split('=', 1) for a in sys.argv[1:]]:
    model, args, epoch = load_model(ckpt)
    print(f'\n##### {name} (ep {epoch})', flush=True)
    for dname, dpath in DATASETS.items():
        cache, config = cache_raw(model, args, dpath)
        print(f'  -- {dname}', flush=True)
        for r, s in itertools.product(RING, SEG):
            config.POSTPROCESSING_NMSIOU_RING, config.POSTPROCESSING_NMSIOU_SEG = r, s
            ev = Evaluator(); kept = []
            for logits, boxes, gt, conf in cache:
                c = filter_boxes(config, onnx_to_xyxy(config, _P(), [logits, boxes], num_select=225))
                kept.append(len(c.boxes))
                ev.get_exp_metrics(c.boxes, c.scores, torch.tensor(np.asarray(gt)), conf)
            _, df2 = get_full_conf_results(ev.metrics)
            tag = '  <- deployed' if (r, s) == (0.1, 0.4) else ''
            print(f'     ring {r:.2f}  seg {s:.2f}   ap_total {df2["ap_total"].values[0]:.4f}   '
                  f'kept/frame {np.mean(kept):6.1f}{tag}', flush=True)
    del model; torch.cuda.empty_cache()
