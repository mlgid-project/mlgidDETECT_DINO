"""Phase-U gate, done correctly: chi-gap recall at a MATCHED OPERATING POINT.

The first pass compared clusters1 to ssl1 at a fixed score threshold (ST=0.30). That is not a
matched operating point, and the phase-P calibration lesson says exactly this: never compare
recall at a fixed score threshold across models. clusters1's precision at ST=0.30 collapsed
(organic 0.841 -> 0.621), so its recall rose in EVERY stratum -- the signature of a model
predicting more boxes, not of a model that learned to separate close siblings.

This probe removes the threshold from the comparison:
  * both models, identical code path, deployed NMS (seg 0.40 / ring 0.1);
  * sweep the score threshold over a grid;
  * report each stratum's recall as a function of OVERALL PRECISION;
  * read both models off at the SAME precision (the control's deployed operating point), by
    linear interpolation in precision.

If clusters1 genuinely fixed the tight-chi defect, its <5px recall is higher at equal precision,
and higher by MORE than the other strata gain. If the gain is uniform across strata, it is an
operating-point shift and phase U did not do what it was built to do.

GPU, ~35 min (two models x two datasets x cached frames).
"""
import os, sys, json
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.evaluation import Evaluator, get_full_conf_results
from util.matchers import get_matcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, DSETS, OUT
from diagnostics.nms_sweep_single import collect, apply_nms, GAP_BINS

SEG_IOU, RING_IOU = 0.40, 0.10          # deployed NMS
ST_GRID = np.round(np.arange(0.05, 0.96, 0.025), 4)
CLUSTERS = os.path.join(OUT, 'clusters1_snapshot.pth')
MODELS = [('ssl1', CKPT_A), ('clusters1_ep318', CLUSTERS)]


def curve(frames):
    """recall per stratum + overall precision/recall, as a function of score threshold."""
    matcher = get_matcher('q', min_iou=0.1)
    pre = [apply_nms(f, SEG_IOU, RING_IOU) for f in frames]
    rows = []
    for st in ST_GRID:
        tp = fp = ngt = ndet = 0
        hit = {g: [0, 0] for g in GAP_BINS}
        for f, (b, s) in zip(frames, pre):
            bo = b[s > st]
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
                for lo, hi in GAP_BINS:
                    if lo <= f['gap'][i] < hi:
                        hit[(lo, hi)][1] += 1
                        if i in mset:
                            hit[(lo, hi)][0] += 1
                        break
        r = dict(st=float(st), recall=tp / max(ngt, 1), precision=tp / max(tp + fp, 1),
                 det_per_frame=ndet / max(len(frames), 1))
        for g in GAP_BINS:
            h, n = hit[g]
            r[f'gap_{g[0]}'] = (h / n) if n else float('nan')
            r[f'ngap_{g[0]}'] = n
        rows.append(r)
    return rows


def read_at_precision(rows, target):
    """Interpolate every recall field at overall precision == target.

    Precision rises monotonically with the threshold (modulo noise), so sort by precision and
    interpolate. Returns None if the target lies outside the achievable range.
    """
    p = np.array([r['precision'] for r in rows])
    o = np.argsort(p)
    p = p[o]
    if not (p.min() <= target <= p.max()):
        return None
    out = {}
    for k in rows[0]:
        if k.startswith('ngap_'):
            out[k] = rows[0][k]
            continue
        v = np.array([rows[i][k] for i in o], float)
        out[k] = float(np.interp(target, p, v))
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)
    summary = {}
    curves = {}
    for name, ckpt in MODELS:
        print(f"\n===== model {name} =====", flush=True)
        model, a = build_model_from_ckpt(CONFIG, ckpt, device)
        for tag, path in DSETS:
            frames, _ = collect(tag, path, model, a, device)
            curves[(name, tag)] = curve(frames)
            print(f"  {name}/{tag}: {len(frames)} frames, curve over {len(ST_GRID)} thresholds",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    for tag, _ in DSETS:
        c, t = curves[('ssl1', tag)], curves[('clusters1_ep318', tag)]
        # reference = the control at its deployed threshold
        ref = min(c, key=lambda r: abs(r['st'] - 0.30))
        target = ref['precision']
        tm = read_at_precision(t, target)
        print(f"\n{'='*104}")
        print(f"{tag.upper()}  —  MATCHED OPERATING POINT (both models read at precision = "
              f"{target:.3f}, the control's deployed point)")
        print('='*104)
        if tm is None:
            pr = [r['precision'] for r in t]
            print(f"  clusters1 CANNOT reach precision {target:.3f} at any threshold "
                  f"(its range is {min(pr):.3f}..{max(pr):.3f}).")
            print("  That is itself the finding: it is strictly less precise than the control.")
            summary[tag] = dict(target_precision=target, reachable=False,
                                clusters_precision_range=[float(min(pr)), float(max(pr))])
            continue
        print(f"  control  ssl1        st={ref['st']:.3f}  recall {ref['recall']:.3f}  "
              f"prec {ref['precision']:.3f}  det/fr {ref['det_per_frame']:.1f}")
        print(f"  test     clusters1   st={tm['st']:.3f}  recall {tm['recall']:.3f}  "
              f"prec {tm['precision']:.3f}  det/fr {tm['det_per_frame']:.1f}")
        print(f"\n  {'chi-gap stratum':16s} {'n':>6s} {'ssl1':>8s} {'clusters1':>10s} {'delta':>8s}")
        print("  " + "-"*52)
        strat = {}
        for lo, hi in GAP_BINS:
            lab = f"<{hi}px" if lo == 0 else (f">={lo}px" if hi > 1e8 else f"{lo}-{hi}px")
            n = int(ref[f'ngap_{lo}']); cv, tv = ref[f'gap_{lo}'], tm[f'gap_{lo}']
            star = '   <== PRIMARY' if lo == 0 else ''
            print(f"  {lab:16s} {n:6d} {cv:8.3f} {tv:10.3f} {tv-cv:+8.3f}{star}")
            strat[lab] = dict(n=n, ctrl=float(cv), test=float(tv), delta=float(tv-cv))
        d0 = strat[f"<{GAP_BINS[0][1]}px"]['delta']
        others = [v['delta'] for k, v in strat.items() if k != f"<{GAP_BINS[0][1]}px"]
        print(f"\n  tight-stratum gain {d0:+.3f}   vs mean gain in the other strata "
              f"{np.mean(others):+.3f}")
        print("  -> targeted fix" if d0 > np.mean(others) + 0.02 else
              "  -> NOT targeted: the tight stratum did not gain more than the rest")
        summary[tag] = dict(target_precision=float(target), reachable=True,
                            control=ref, test=tm, strata=strat,
                            tight_gain=float(d0), other_gain_mean=float(np.mean(others)))

    json.dump(dict(summary=summary,
                   curves={f'{k[0]}|{k[1]}': v for k, v in curves.items()}),
              open(os.path.join(OUT, 'matched_op_probe.json'), 'w'), indent=2, default=str)
    print(f"\nwrote {OUT}/matched_op_probe.json")
    print("PROBE DONE")


if __name__ == '__main__':
    main()
