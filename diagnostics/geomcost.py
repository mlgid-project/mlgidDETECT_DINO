"""What does the quazipolar half of the training stream cost?

`FastSimulation.filter_dark_area` (simulation.py:799) draws one random number per frame:
>0.5 gives a QUAZIPOLAR wedge, else POLAR. Measured 2026-09-15 the split is 50/50 in every
simulator, and the two geometries leave very different amounts of detector visible -- valid-mask
fraction median 0.773 polar vs 0.455 quazipolar. DINO evaluation of real data is always standard
polar, so half of every training stream is a geometry the eval never contains.

This prices it: run a trained model over SIMULATED frames (GT known), split by the geometry the
simulator actually drew, and compare. If quazipolar AP is much lower, that half of the stream is a
harder task soaking up capacity for nothing. If it matches, the mismatch is cosmetic.

Geometry is read off the mask `simulate_img` returns, not guessed: the two modes are cleanly
bimodal, so a 0.6 cut separates them exactly.

  python geomcost.py name=/path/to/checkpoint.pth [...]   [--frames N]

RESULT 2026-09-15: NO geometry penalty, lever killed. On its own simulator lr4e5
scores 0.9990 polar / 0.9976 quazipolar and physics4 0.8175 / 0.8560 -- the latter confounded, as
quazipolar frames carry 39.9 GT vs 57.6 (the wedge eats peaks). Side finding, more important than
the question asked: lr4e5 SOLVES its own sim (precision 0.954, 46.8 boxes for 44.7 GT) while
physics4 does not (0.507, 96.3 for 57.6). See MODIFICATIONS.md section M.
"""
import os, sys, json, argparse
import numpy as np, torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, _eval_config, _P
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, recall_precision_curve_with_intensities
from util.matchers import get_matcher

N = int(os.environ.get('FRAMES', '300'))
matcher = get_matcher('q', min_iou=0.1)


def build_sim(args):
    from main import SimulationDataset
    return SimulationDataset(args)


def run(model, args, n):
    ds = build_sim(args)
    sim = ds.physics if ds.physics is not None else ds.simulation
    config = _eval_config('/dev/null')          # only PREPROCESSING_POLAR_SHAPE + NMS/score matter
    groups = {'polar': [], 'quazipolar': []}
    for _ in range(n):
        while True:
            try:
                img, boxes, mask, is_ring = sim.simulate_img()
                if img is not None and torch.isfinite(img).all() and len(boxes):
                    break
            except Exception:
                pass
        frac = float(mask.float().mean())
        g = 'polar' if frac > 0.6 else 'quazipolar'
        x = img.repeat(args.num_channels, 1, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            o = model(x)
        groups[g].append((o['pred_logits'].cpu().numpy(), o['pred_boxes'].cpu().numpy(),
                          boxes.cpu().numpy(), frac))
    return groups, config


def score(group, config):
    ev = Evaluator(); ngt = ntp = nfp = 0; ious = []
    for logits, pboxes, gt, _ in group:
        c = filter_boxes(config, onnx_to_xyxy(config, _P(), [logits, pboxes], num_select=225))
        t = torch.tensor(gt).float()
        ev.get_exp_metrics(c.boxes, c.scores, t, np.full(len(gt), -1.0))
        _, r, cc = matcher(t, c.boxes)
        ngt += len(t); ntp += len(r); nfp += len(c.boxes) - len(r)
    ap = recall_precision_curve_with_intensities(ev.metrics)[4]   # av_precision == ap_total
    n = len(group)
    return dict(ap=float(ap), n=n, gt=ngt/n,
                recall=ntp/max(ngt, 1), prec=ntp/max(ntp+nfp, 1),
                kept=(ntp+nfp)/n, frac=np.mean([g[3] for g in group]))


if __name__ == '__main__':
    for name, ckpt in [a.split('=', 1) for a in sys.argv[1:]]:
        model, args, epoch = load_model(ckpt)
        groups, config = run(model, args, N)
        print(f'\n##### {name} (ep {epoch}) on its OWN simulator, {N} frames', flush=True)
        print(f'{"geometry":>12} {"n":>5} {"maskfrac":>9} {"GT/fr":>7} {"ap_total":>9} '
              f'{"recall":>8} {"prec":>8} {"kept/fr":>8}')
        for g in ('polar', 'quazipolar'):
            if not groups[g]:
                print(f'{g:>12}     0'); continue
            s = score(groups[g], config)
            print(f'{g:>12} {s["n"]:>5} {s["frac"]:>9.3f} {s["gt"]:>7.1f} {s["ap"]:>9.4f} '
                  f'{s["recall"]:>8.4f} {s["prec"]:>8.4f} {s["kept"]:>8.1f}', flush=True)
        del model; torch.cuda.empty_cache()
