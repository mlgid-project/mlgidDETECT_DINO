"""q-dependent ("gradient") score threshold on an EXISTING heatmap checkpoint: no training involved.

  python heatmap/q_threshold.py <name=checkpoint.pth> [...] --out <dir> [--device cuda|cpu]

Idea (user, 2026-10-09): peaks at high q are typically faint, so they should need a LOWER score to count.
Threshold as a function of the box-centre column x (x runs along q, W = 1024):
        thr(x) = t0 + (t1 - t0) * x / W          t0 = threshold at x = 0 (low q), t1 = at x = W (high q)
t1 = t0 is the ordinary constant threshold (the control: same level, no gradient). One forward pass per checkpoint
(top-225 + class-aware NMS, NO score floor); everything else is done on those detections on the CPU.

Reported per set (organic, 41):
  1. PREMISE: per q-third, GT count and share of faint GT (conf 0.1), recall, TP/FP counts and median TP / FP score
     at the deployed floor 0.1 -- are high-q peaks really faint, and are the low-score high-q detections TPs?
  2. GRID over (t0, t1), two ways of using the threshold:
       filter : keep detections with score > thr(x), scores unchanged (what the evaluator's floor does today)
       norm   : keep them and rescale s' = (s - thr(x)) / (1 - thr(x)), so the ranking is by distance above the local threshold
     for each: AP (repo Evaluator), recall / precision / F1 / FP of the kept set, ring recall, recall in the NN<5 px bucket.
  3. CROSS-SET check: the (t0, t1) with the best AP (or F1) on one set, evaluated on the other (the grid is tuned on the eval
     sets, so a single-set optimum is optimistic).
Baselines in the same table: the headline constant floor 0.1 (t0 = t1 = 0.1) and the constant 0.3 operating point."""
import os, sys, json, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from models.heatmap_head import decode
from heatmap import evaluation as E
from heatmap.visualize import load, classify

W = 1024.0
T0 = [0.1, 0.15, 0.2, 0.3, 0.4]
T1 = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
THIRDS = [(0, 341.3), (341.3, 682.7), (682.7, 1024.1)]


def thr_x(t0, t1, b):
    cx = (b[:, 0] + b[:, 2]) / 2
    return t0 + (t1 - t0) * cx / W


def apply(dets, t0, t1, mode):
    out = []
    for b, s in dets:
        if len(s) == 0:
            out.append((b, s)); continue
        th = thr_x(t0, t1, b)
        k = s > th
        out.append((b[k], s[k] if mode == 'filter' else ((s[k] - th[k]) / (1 - th[k])).astype(np.float32)))
    return out


def f1(r, p):
    return 2 * r * p / max(r + p, 1e-9)


def metrics(dets, gts, cfg):
    r = E.evaluate_dets(dets, gts, cfg, thr_list=(0.0,))
    t = r['thr'][0.0]
    return dict(ap=r['ap'], recall=t['recall'], precision=t['precision'], f1=f1(t['recall'], t['precision']), fp=t['fp'],
                n_pred=t['n_pred'], ring_recall=t['ring_recall'], eu5=t['euclid']['<5'][1])


def premise(dets, gts, lines):
    lines.append('  PREMISE (deployed set: score > 0.1; thirds of the q axis by box centre)')
    lines.append('    third(x)      GT   faint(conf .1)  recall@.1   dets  TP   FP   median score TP / FP   TP with score<0.3')
    for lo, hi in THIRDS:
        n_gt = n_faint = n_hit = n_det = n_tp = low_tp = 0
        s_tp, s_fp = [], []
        for (b, s), g in zip(dets, gts):
            k = s > 0.1
            b, s = b[k], s[k]
            gt = g['gt']; conf = g['gtconf']
            ci, ri = classify(gt, b)
            gx = (gt[:, 0] + gt[:, 2]) / 2 if len(gt) else np.zeros(0)
            m = (gx >= lo) & (gx < hi)
            n_gt += int(m.sum()); n_faint += int((m & (conf <= 0.11)).sum())
            hit = np.zeros(len(gt), bool); hit[ri] = True; n_hit += int((hit & m).sum())
            dx = (b[:, 0] + b[:, 2]) / 2 if len(b) else np.zeros(0)
            dm = (dx >= lo) & (dx < hi)
            tp = np.zeros(len(b), bool); tp[ci] = True
            n_det += int(dm.sum()); n_tp += int((tp & dm).sum())
            s_tp += list(s[tp & dm]); s_fp += list(s[~tp & dm])
            low_tp += int((tp & dm & (s < 0.3)).sum())
        med = lambda a: float(np.median(a)) if len(a) else float('nan')
        lines.append(f'    {int(lo):4d}-{min(int(hi),1024):4d}   {n_gt:5d}   {n_faint/max(n_gt,1):.2f}            {n_hit/max(n_gt,1):.3f}     '
                     f'{n_det:5d} {n_tp:4d} {n_det-n_tp:4d}   {med(s_tp):.3f} / {med(s_fp):.3f}        {low_tp}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='name=checkpoint.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for spec in a.runs:
        name, ck = spec.split('=', 1)
        model, ep = load(ck, a.device)
        lines, js = [f'Q-DEPENDENT SCORE THRESHOLD  run={name} (epoch {ep})  ckpt={ck}',
                     'thr(x) = t0 + (t1 - t0) * x / 1024 (x = box-centre column, q increases with x); t0 = t1 is the constant control', ''], {}
        data = {}
        for ds, path in E.DATASETS.items():
            dets, gts, cfg0 = [], [], None
            for cfg, ic in E.iter_frames(path):
                with torch.no_grad():
                    x = (E.frame_inputs(ic, a.device) if model.chan_mode == 'he'
                         else E.frame_inputs(ic, a.device, chan_mode=model.chan_mode))      # works on commit e4d2cc4 too
                    o = model(x, E.frame_mask(ic, a.device))
                gts.append(E.gt_of(ic)); cfg.POSTPROCESSING_SCORE = 0.0; cfg0 = cfg
                dets.append(E.heatmap_dets(cfg, decode(o, model.out_stride, num_select=225, score_floor=0.0)[0], use_nms=True))
            data[ds] = (dets, gts, cfg0)
        res = {}
        for ds, (dets, gts, cfg) in data.items():
            lines.append(f'=== {ds} ({len(gts)} frames, {sum(len(g["gt"]) for g in gts)} GT) ===')
            premise(dets, gts, lines)
            res[ds] = {}
            for mode in ('filter', 'norm'):
                lines.append(f'  GRID, mode={mode}   (AP | recall precision F1 FP | ring-recall | NN<5px recall of the kept set)')
                lines.append('    t0    t1    AP      recall prec   F1     FP    ring   eu<5')
                for t0 in T0:
                    for t1 in T1:
                        if t1 > t0:
                            continue
                        m = metrics(apply(dets, t0, t1, mode), gts, cfg)
                        res[ds][(mode, t0, t1)] = m
                        tag = '  <- constant' if t1 == t0 else ''
                        lines.append(f'    {t0:<5} {t1:<5} {m["ap"]:.4f}  {m["recall"]:.3f}  {m["precision"]:.3f} {m["f1"]:.3f} {m["fp"]:5d}  {m["ring_recall"]:.3f}  {m["eu5"]:.3f}{tag}')
            lines.append('')
        # cross-set: best on one set, evaluated on the other
        lines.append('=== CROSS-SET (best on A, reported on B; the constant headline floor 0.1 for reference) ===')
        for mode in ('filter', 'norm'):
            for key in ('ap', 'f1'):
                for A, B in (('organic', '41'), ('41', 'organic')):
                    if A not in res or B not in res:
                        continue
                    best = max(res[A], key=lambda k: res[A][k][key] if k[0] == mode else -1)
                    ref = res[B][(mode, 0.1, 0.1)]
                    lines.append(f'  {mode:6s} best-{key} on {A}: (t0={best[1]}, t1={best[2]}) {key}={res[A][best][key]:.4f} on {A}; on {B}: '
                                 f'AP {res[B][best]["ap"]:.4f} F1 {res[B][best]["f1"]:.3f} (const 0.1: AP {ref["ap"]:.4f} F1 {ref["f1"]:.3f})')
        txt = '\n'.join(lines)
        print(txt, flush=True)
        open(os.path.join(a.out, f'{name}_q_threshold.txt'), 'w').write(txt + '\n')
        json.dump({ds: {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in d.items()} for ds, d in res.items()},
                  open(os.path.join(a.out, f'{name}_q_threshold.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
