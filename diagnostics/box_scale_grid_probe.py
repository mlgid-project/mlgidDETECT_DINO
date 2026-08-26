"""Phase AC.5 — the q convention too, and the two axes jointly.

AC.4 priced the chi convention alone and found the gate almost indifferent to it: ap_total moves less
than 0.006 anywhere between 0.70x and 1.20x on organic, and less than 0.002 on 41. But the plan also
calls for changing `w_coef` (1.0 -> ~1.9, since AC.2 measured the simulator at k_q 0.50 against
organic's 1.05 and 41's 0.88), and NOTHING has priced that. Too-small boxes lose IoU faster than
too-large ones, so the q axis cannot be assumed to behave like chi.

Both coefficients are going to change together, so they are swept together: a 2D grid of chi-scale x
q-scale applied to the predicted boxes about their own centres, deployed evaluation at every node
(same Evaluator as --eval, deployed NMS seg 0.40 / ring 0.10, deployed score floor). Frames are
cached once per gate, so the grid is nearly free after the forward passes.

Reported per gate and as the sum over both, plus the `a_coef` / `w_coef` each node corresponds to,
and the <5 px gap-bin recall at every node -- AC.4 found that bucket swings 0.218 to 0.394 with box
height while the >=33 px bucket moves 0.023, which is the sharpest evidence so far that tall boxes
merge close pairs. Whether the q axis carries the same signal is unknown and is the second question
here.

Same caveat as AC.4: this is a global rescale of a model trained at k_chi 1.75 / k_q 0.50. A
retrained model would learn the size per peak instead of taking a uniform squeeze, so these numbers
price the KNOB, not the retrain, and are a lower bound on what matching the convention could buy.

Single model ssl1. GPU, ~20 min. See tmp_diag/run_scalegrid.sbatch.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from diagnostics.nms_sweep_single import (collect, evaluate, CONFIG, DSETS,
                                          build_model_from_ckpt, CKPT)

CHI = [1.20, 1.00, 0.85, 0.70, 0.60]
Q = [1.00, 1.30, 1.60, 1.90, 2.20]
SEG_IOU, RING_IOU = 0.40, 0.10
A_COEF, W_COEF = 3.5, 1.0


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


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1  deployed NMS seg={SEG_IOU} ring={RING_IOU}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT, dev)

    res = {}
    for tag, path in DSETS:
        frames, _dup = collect(tag, path, model, a, dev)
        res[tag] = {}
        for fc in CHI:
            for fq in Q:
                res[tag][(fc, fq)] = evaluate(scaled(frames, fc, fq), SEG_IOU, RING_IOU)
            print(f"  [{tag}] chi x{fc:.2f} done", flush=True)

    tags = list(res)
    for key, lab in [('ap_total', 'ap_total'), ('recall', 'recall'),
                     ('precision', 'precision'), ('gap_0', '<5 px gap recall')]:
        print("\n" + "=" * 104)
        print(f"  {lab.upper()}  — rows: chi scale,  cols: q scale")
        print("=" * 104)
        for tag in tags:
            print(f"\n  {tag}")
            print(f"  {'chi \\ q':>10s}" + "".join(f"{q:>10.2f}" for q in Q))
            for fc in CHI:
                print(f"  {fc:10.2f}" + "".join(f"{res[tag][(fc, q)][key]:10.4f}" for q in Q))

    print("\n" + "=" * 104)
    print("  SUM of ap_total over both gates, and delta vs deployed (1.00, 1.00)")
    print("=" * 104)
    base = sum(res[t][(1.00, 1.00)]['ap_total'] for t in tags)
    print(f"  {'chi \\ q':>10s}" + "".join(f"{q:>10.2f}" for q in Q))
    for fc in CHI:
        print(f"  {fc:10.2f}" + "".join(
            f"{sum(res[t][(fc, q)]['ap_total'] for t in tags) - base:+10.4f}" for q in Q))

    best = max(((fc, fq) for fc in CHI for fq in Q),
               key=lambda k: sum(res[t][k]['ap_total'] for t in tags))
    print(f"\n  best node: chi x{best[0]:.2f}, q x{best[1]:.2f}"
          f"   ->  a_coef {A_COEF * best[0]:.2f}, w_coef {W_COEF * best[1]:.2f}"
          f"   ->  k_chi {A_COEF * best[0] / 2:.2f}, k_q {W_COEF * best[1] / 2:.2f}")
    for t in tags:
        print(f"    {t:<20s} ap_total {res[t][best]['ap_total']:.4f} "
              f"(deployed {res[t][(1.00, 1.00)]['ap_total']:.4f}, "
              f"delta {res[t][best]['ap_total'] - res[t][(1.00, 1.00)]['ap_total']:+.4f})")
    print("\n  AC.2 measured the LABELS at: organic k_chi 0.92 / k_q 1.05,  41 k_chi 1.71 / k_q 0.88")
    print("  simulator is at k_chi 1.75 / k_q 0.50 — the node matching organic's labels exactly is")
    print(f"  chi x{0.92 / 1.75:.2f}, q x{1.05 / 0.50:.2f}")

    json.dump({t: {f"{k[0]}_{k[1]}": v for k, v in res[t].items()} for t in res},
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/box_scale_grid.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
