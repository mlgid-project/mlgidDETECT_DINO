"""Box-size sweep on cached detections: how does AP depend on the predicted box size, separately for segments and rings?

  python heatmap/box_sweep.py <dets.pkl> [--sets organic 41] [--out <file>.txt]

dets.pkl = heatmap/dump_dets.py output (boxes, scores, cls per frame; gt = the stored GT box convention of the dump, i.e. unscaled when HM_GT_BOX_COEF was unset).
The predicted boxes are rescaled about their centre and scored against that fixed GT:
  segments: x (radial |q|) factor fx_s and y (angular chi) factor fy_s, scored on segment GT only (is_ring_geom GT removed / ring detections removed),
  rings   : x factor fx_r (rings span the full chi range, y untouched), scored on ring GT only.
Reading: the factor where AP peaks times the training coefficient (3.0) is the box size, in coef units (full width in sigma), that matches the stored GT best for that kind.
A model trained at coefficient c predicts boxes of coefficient c, so factor f means coefficient c*f. CPU only."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E
from util.evaluation import Evaluator, get_full_conf_results

FACT = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]


def scale(b, fx, fy):
    if len(b) == 0:
        return b
    b = np.asarray(b, np.float64)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    hw, hh = (b[:, 2] - b[:, 0]) / 2 * fx, (b[:, 3] - b[:, 1]) / 2 * fy
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1).astype(np.float32)


def ap(frames, kind, fx, fy):
    """AP over the frames for one kind (0 segment, 1 ring) with the predicted boxes of that kind rescaled by (fx, fy)."""
    ev = Evaluator()
    for fr in frames:
        gt = np.asarray(fr['gt'], np.float32).reshape(-1, 4)
        ring = E.is_ring_geom(gt, fr['mask']) if len(gt) else np.zeros(0, bool)
        gsel = ring if kind == 1 else ~ring
        cls = np.asarray(fr['cls']).reshape(-1)
        psel = cls == kind
        b = scale(np.asarray(fr['boxes']).reshape(-1, 4)[psel], fx, fy)
        ev.get_exp_metrics(torch.tensor(b), torch.tensor(np.asarray(fr['scores'])[psel]), torch.tensor(gt[gsel]), np.asarray(fr['gtconf'])[gsel])
    df1, df2 = get_full_conf_results(ev.metrics)
    return float(df2['ap_total'].values[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dets'); p.add_argument('--sets', nargs='*', default=['organic', '41']); p.add_argument('--out', default=None)
    p.add_argument('--coef', type=float, default=3.0, help='training coefficient of the model that made the detections')
    a = p.parse_args()
    d = pickle.load(open(a.dets, 'rb'))
    L = [f'BOX SWEEP  run={d.get("run")} epoch={d.get("epoch")}  dets={a.dets}', f'training coefficient {a.coef}; factor f -> coefficient {a.coef}*f; AP of one kind against the stored GT of that kind', '']
    for ds in a.sets:
        fr = d['sets'][ds]
        n_s = sum(int((~E.is_ring_geom(np.asarray(f['gt']).reshape(-1, 4), f['mask'])).sum()) if len(f['gt']) else 0 for f in fr)
        n_r = sum(len(f['gt']) for f in fr) - n_s
        L.append(f'== {ds}: {len(fr)} frames, {n_s} segment GT, {n_r} ring GT')
        L.append('  rings: radial factor fx_r -> AP   ' + '  '.join(f'{f:.2f}:{ap(fr, 1, f, 1.0):.3f}' for f in FACT))
        L.append('  segments: AP(fx radial rows x fy angular cols), cols = ' + ' '.join(f'{f:.2f}' for f in FACT))
        best = (-1, None)
        for fx in FACT:
            row = [ap(fr, 0, fx, fy) for fy in FACT]
            L.append(f'    fx {fx:.2f}: ' + ' '.join(f'{v:.3f}' for v in row))
            m = int(np.argmax(row))
            if row[m] > best[0]:
                best = (row[m], (fx, FACT[m]))
        L.append(f'  best segment cell: fx {best[1][0]:.2f} fy {best[1][1]:.2f} -> coefficients (radial {a.coef * best[1][0]:.2f}, angular {a.coef * best[1][1]:.2f}), AP {best[0]:.3f}')
        L.append('')
        print('\n'.join(L[-(len(FACT) + 6):]), flush=True)
    if a.out:
        open(a.out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
