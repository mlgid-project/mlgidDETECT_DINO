"""Score sweep: how do AP, recall, precision and F1 depend on the score cut and on the top-K cap?

  python heatmap/score_sweep.py <name=checkpoint.pth> [...] --out <dir> [--decode nms|native|both] [--topk 225 900]

AP depends on the LOWEST score the evaluator sees (the deployed evaluation drops everything below 0.1, a cut tuned
for DINO's score scale; a heatmap's scores may be lower for true peaks). So per set, decode and top-K:
  (1) AP when only detections with score > FLOOR are kept, FLOOR in {0 (all), .01, .02, .05, .1, .2, .3}
  (2) recall / precision / F1 / FP at operating thresholds .05 ... .5 (computed on the all-detections set)
  (3) the best-F1 threshold.
Writes <out>/<name>_score_sweep.txt and .json. Runs on colorbox1 (GPU for the forward pass, CPU for the metrics)."""
import os, sys, json, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from models.heatmap_head import decode
from heatmap import evaluation as E
from heatmap.visualize import load

FLOORS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
THRS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='name=checkpoint.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--decode', choices=['native', 'nms', 'both'], default='nms')
    p.add_argument('--topk', type=int, nargs='+', default=[225, 900])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    decs = ['native', 'nms'] if a.decode == 'both' else [a.decode]
    for spec in a.runs:
        name, ck = spec.split('=', 1)
        model, ep = load(ck, a.device)
        lines, js = [f'SCORE SWEEP  run={name} (epoch {ep})  ckpt={ck}', ''], {}
        for ds, path in E.DATASETS.items():
            gts, per_frame, cfgs = [], [], []
            for cfg, ic in E.iter_frames(path):
                with torch.no_grad():
                    o = model(E.frame_inputs(ic, a.device), E.frame_mask(ic, a.device))
                gts.append(E.gt_of(ic)); cfgs.append(cfg)
                per_frame.append({K: decode(o, model.out_stride, num_select=K, score_floor=0.0)[0] for K in a.topk})
            n_gt = sum(len(g['gt']) for g in gts)
            for dec in decs:
                for K in a.topk:
                    key = f'{ds}|{dec}|top{K}'
                    lines.append(f'=== {ds} ({len(gts)} frames, {n_gt} GT) | decode {dec} | top-K {K} ===')
                    aps = {}
                    lines.append('  AP vs score FLOOR (detections with score > floor are kept):')
                    for fl in FLOORS:
                        dets = []
                        for cfg, pf in zip(cfgs, per_frame):
                            cfg.POSTPROCESSING_SCORE = fl
                            dets.append(E.heatmap_dets(cfg, pf[K], use_nms=(dec == 'nms')))
                        # floor 0 keeps everything with score > 0; the evaluator then sees every candidate
                        r = E.evaluate_dets(dets, gts, cfgs[0], thr_list=tuple(THRS) if fl == 0.0 else (0.3,))
                        aps[fl] = r['ap']
                        nd = sum(len(d[0]) for d in dets) / len(dets)
                        lines.append(f'    floor {fl:<5}: AP {r["ap"]:.4f}   ({nd:.0f} detections/frame)')
                        if fl == 0.0:
                            ops = r['thr']
                    lines.append('  operating points (all detections, then cut at score > thr):')
                    best = (-1, None)
                    for t in THRS:
                        o_ = ops[t]; f1 = 2 * o_['recall'] * o_['precision'] / max(o_['recall'] + o_['precision'], 1e-9)
                        lines.append(f'    thr {t:<4}: recall {o_["recall"]:.3f} (seg {o_["seg_recall"]:.3f} ring {o_["ring_recall"]:.3f})'
                                     f'  precision {o_["precision"]:.3f}  F1 {f1:.3f}  FP {o_["fp"]}')
                        if f1 > best[0]:
                            best = (f1, t)
                    lines.append(f'  best F1 {best[0]:.3f} at thr {best[1]};  best AP {max(aps.values()):.4f} at floor {max(aps, key=aps.get)}')
                    lines.append('')
                    js[key] = dict(ap_by_floor=aps, best_f1=best, ops={str(t): ops[t] for t in THRS})
        txt = '\n'.join(lines)
        print(txt, flush=True)
        open(os.path.join(a.out, f'{name}_score_sweep.txt'), 'w').write(txt + '\n')
        json.dump(js, open(os.path.join(a.out, f'{name}_score_sweep.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
