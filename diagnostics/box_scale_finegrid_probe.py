"""Phase AE — fine grid around AC.5's optimum, with an error bar.

AC.5 swept chi x q on a 5x5 grid and found the sum of `ap_total` over both gates peaked at
chi x0.85, q x1.30 => `a_coef` 2.98, `w_coef` 1.30 (+0.0127 organic, +0.0057 on 41). Two things
were left open and both have to be closed before that node is worth a 3-day training run:

  1  RESOLUTION. The chi optimum 0.85 was interior to [1.20, 1.00, 0.85, 0.70, 0.60], but the q
     optimum 1.30 sat one node above the grid's LOWER EDGE of 1.00 -- nothing below 1.0 was ever
     tested, so the q surface's shape on that side is unknown. This grid runs chi 0.70-1.00 in
     steps of 0.05 and q 0.85-1.60 in steps of ~0.10, which brackets both.

  2  NOISE. Organic is 8 frames. AC.5 already warned that +0.0127 there "should not be read as
     significant on its own". A finer grid makes that worse, not better: with 56 nodes instead of
     25 the argmax is more likely to be a noise excursion. So every node also gets a JACKKNIFE
     standard error (leave-one-frame-out, the delta-method SE for a statistic that is not a mean
     of per-frame numbers, which ap_total is not). The node reported is not the raw argmax but
     the PLATEAU: every node whose sum is within 1 SE of the max. If the plateau is wide, the
     honest answer is "any node in here", and we pick the one that is also defensible physically.

Everything else is AC.5 unchanged: global rescale of the CURRENT ssl1 model's predicted boxes about
their own centres, applied BEFORE NMS, deployed evaluation at every node (same Evaluator as --eval,
deployed NMS seg 0.40 / ring 0.10, deployed score floor). Frames are cached once per gate.

SAME CAVEAT AS AC.5, and it does not go away by making the grid finer: this prices the KNOB on a
model trained at k_chi 1.75 / k_q 0.50, not a retrain. A model trained at the chosen coefficients
learns the size per peak instead of taking a uniform squeeze, so these numbers are a guide to where
to put `a_coef` / `w_coef`, not a prediction of what the retrain will score.

Single model ssl1. GPU, ~1 h. See tmp_diag/run_finegrid.sbatch.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from diagnostics.nms_sweep_single import (collect, evaluate, CONFIG, DSETS,
                                          build_model_from_ckpt, CKPT)

CHI = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
Q = [0.85, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]
SEG_IOU, RING_IOU = 0.40, 0.10
A_COEF, W_COEF = 3.5, 1.0
BASE = (1.00, 1.00)


def scaled(frames, fc, fq):
    out = []
    for fr in frames:
        b = fr['b'].clone()
        cy = (b[:, 1] + b[:, 3]) / 2; hh = (b[:, 3] - b[:, 1]) / 2 * fc
        cx = (b[:, 0] + b[:, 2]) / 2; hw = (b[:, 2] - b[:, 0]) / 2 * fq
        b[:, 1] = cy - hh; b[:, 3] = cy + hh
        b[:, 0] = cx - hw; b[:, 2] = cx + hw
        out.append(dict(fr, b=b))
    return out


def jackknife_se(frames, node, base):
    """SE of (ap_total[node] - ap_total[base]) over frames, leave-one-out."""
    n = len(frames)
    if n < 3:
        return float('nan')
    full = (evaluate(scaled(frames, *node), SEG_IOU, RING_IOU)['ap_total']
            - evaluate(scaled(frames, *base), SEG_IOU, RING_IOU)['ap_total'])
    vals = []
    for i in range(n):
        sub = [f for j, f in enumerate(frames) if j != i]
        vals.append(evaluate(scaled(sub, *node), SEG_IOU, RING_IOU)['ap_total']
                    - evaluate(scaled(sub, *base), SEG_IOU, RING_IOU)['ap_total'])
    vals = np.asarray(vals, dtype=float)
    return float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2))), full


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1  deployed NMS seg={SEG_IOU} ring={RING_IOU}", flush=True)
    print(f"grid: {len(CHI)} chi x {len(Q)} q = {len(CHI) * len(Q)} nodes", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT, dev)
    model.eval()

    res, cache = {}, {}
    for tag, path in DSETS:
        frames, _dup = collect(tag, path, model, a, dev)
        cache[tag] = frames
        print(f"  [{tag}] {len(frames)} frames cached", flush=True)
        res[tag] = {}
        for fc in CHI:
            for fq in Q:
                res[tag][(fc, fq)] = evaluate(scaled(frames, fc, fq), SEG_IOU, RING_IOU)
            print(f"  [{tag}] chi x{fc:.2f} done", flush=True)

    tags = list(res)
    for key, lab in [('ap_total', 'ap_total'), ('recall', 'recall'),
                     ('precision', 'precision'), ('gap_0', '<5 px gap recall')]:
        print("\n" + "=" * 110)
        print(f"  {lab.upper()}  — rows: chi scale,  cols: q scale")
        print("=" * 110)
        for tag in tags:
            print(f"\n  {tag}")
            print(f"  {'chi \\ q':>10s}" + "".join(f"{q:>10.2f}" for q in Q))
            for fc in CHI:
                print(f"  {fc:10.2f}" + "".join(f"{res[tag][(fc, q)][key]:10.4f}" for q in Q))

    print("\n" + "=" * 110)
    print("  SUM of ap_total over both gates, DELTA vs deployed (1.00, 1.00)")
    print("=" * 110)
    base = sum(res[t][BASE]['ap_total'] for t in tags)
    print(f"  {'chi \\ q':>10s}" + "".join(f"{q:>10.2f}" for q in Q))
    for fc in CHI:
        print(f"  {fc:10.2f}" + "".join(
            f"{sum(res[t][(fc, q)]['ap_total'] for t in tags) - base:+10.4f}" for q in Q))

    nodes = [(fc, fq) for fc in CHI for fq in Q]
    tot = {k: sum(res[t][k]['ap_total'] for t in tags) for k in nodes}
    best = max(nodes, key=lambda k: tot[k])
    print(f"\n  raw argmax: chi x{best[0]:.2f}, q x{best[1]:.2f}"
          f"  ->  a_coef {A_COEF * best[0]:.2f}, w_coef {W_COEF * best[1]:.2f}"
          f"  (sum delta {tot[best] - base:+.4f})")

    print("\n" + "=" * 110)
    print("  JACKKNIFE (leave-one-frame-out) SE of the delta at the argmax")
    print("=" * 110)
    se_sum2 = 0.0
    for t in tags:
        se, full = jackknife_se(cache[t], best, BASE)
        se_sum2 += se ** 2
        print(f"  {t:<20s} n={len(cache[t]):3d}  delta {full:+.4f}  SE {se:.4f}"
              f"  ->  {full / se if se else float('nan'):+.2f} sigma")
    se_sum = float(np.sqrt(se_sum2))
    print(f"  {'SUM (indep.)':<20s}          delta {tot[best] - base:+.4f}  SE {se_sum:.4f}"
          f"  ->  {(tot[best] - base) / se_sum if se_sum else float('nan'):+.2f} sigma")

    print(f"\n  PLATEAU — every node within 1 SE ({se_sum:.4f}) of the max sum:")
    plat = [k for k in nodes if tot[k] >= tot[best] - se_sum]
    for k in sorted(plat, key=lambda k: -tot[k]):
        print(f"    chi x{k[0]:.2f} q x{k[1]:.2f}  ->  a_coef {A_COEF * k[0]:5.2f}"
              f"  w_coef {W_COEF * k[1]:5.2f}   sum delta {tot[k] - base:+.4f}"
              + "".join(f"   {t} {res[t][k]['ap_total']:.4f}" for t in tags))
    print(f"  plateau size: {len(plat)} of {len(nodes)} nodes")

    json.dump({t: {f"{k[0]}_{k[1]}": v for k, v in res[t].items()} for t in res},
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/box_scale_finegrid.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
