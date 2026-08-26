"""Phase AC.4 — what does the chi boxing convention actually COST on the gate?

AC.3 settled the measurement and left a choice that cannot be settled by measuring labels. The two
eval gates use different chi conventions -- at MATCHED peak shape, 41's box half-extent is ~2.1x
organic's in every overlapping bin -- so no single `a_coef` satisfies both:

    organic wants k_chi 0.92     41 wants k_chi 1.71     simulator is at 1.75 (a_coef 3.5)

Keep 3.5 and organic carries a 1.9x convention penalty; drop it to 1.84 and 41's boxes come out at
0.54x of what its labels want. Picking between those by argument is guessing. This prices it.

The model is not retrained. Its predicted boxes are rescaled in chi about their own centres by a
global factor, and the deployed evaluation is re-run at each factor -- the SAME `Evaluator` and
`get_full_conf_results` behind the --eval gate, with deployed NMS (seg_iou 0.40, ring_iou 0.10) and
the deployed score floor. Scaling happens BEFORE NMS, so the sweep also captures the second effect
the convention has: shorter boxes overlap less, so duplicate suppression deletes fewer close pairs.
That is the mechanism the `<5 px` recall hole was hypothesised to run through, and the gap-binned
recall is reported at every scale so it can be watched directly.

WHAT THE ANSWER DECIDES:
  ap_total flat over 0.5-1.5x  =>  the gate does not care about the convention, so `a_coef` should be
                                   chosen for downstream peak FITTING (tight, ~1 sigma) and the
                                   simulator work should spend itself entirely on peak SHAPE.
  ap_total sensitive           =>  the convention is worth points, and `a_coef` becomes a real
                                   trade-off to be set by the sum over both gates, reported here.

This measures the CURRENT model's response to a global rescale. It is a lower bound on what matching
the convention in training could buy, and it cannot show what a retrained model would learn per-peak;
it prices the knob, it does not simulate the retrain.

Single model ssl1. GPU, ~15 min. See tmp_diag/run_chiscale.sbatch.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from diagnostics.nms_sweep_single import (collect, evaluate, GAP_BINS, CONFIG, DSETS,
                                          build_model_from_ckpt, CKPT)

SCALES = [1.00, 0.85, 0.70, 0.60, 0.53, 0.45, 0.35, 1.20, 1.50]
SEG_IOU, RING_IOU = 0.40, 0.10     # deployed


def scaled(frames, f):
    """Same frames with every predicted box rescaled in chi about its own centre."""
    out = []
    for fr in frames:
        b = fr['b'].clone()
        cy = (b[:, 1] + b[:, 3]) / 2
        hh = (b[:, 3] - b[:, 1]) / 2 * f
        b[:, 1] = cy - hh; b[:, 3] = cy + hh
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
        for f in SCALES:
            res[tag][f] = evaluate(scaled(frames, f), SEG_IOU, RING_IOU)
            print(f"  [{tag}] chi x{f:.2f}: ap_total {res[tag][f]['ap_total']:.4f}", flush=True)

    print("\n" + "=" * 104)
    print("  ap_total AND friends vs a GLOBAL CHI RESCALE of the predicted boxes")
    print("=" * 104)
    for tag in res:
        print(f"\n  {tag}")
        print(f"  {'chi scale':>10s}{'ap_total':>10s}{'ap_high':>9s}{'recall':>8s}{'prec':>7s}"
              f"{'det/fr':>8s}" + "".join(f"{('gap' + str(g[0])):>9s}" for g in GAP_BINS))
        for f in SCALES:
            r = res[tag][f]
            print(f"  {f:10.2f}{r['ap_total']:10.4f}{r['ap_high']:9.4f}{r['recall']:8.3f}"
                  f"{r['precision']:7.3f}{r['det_per_frame']:8.1f}"
                  + "".join(f"{r['gap_' + str(g[0])]:9.3f}" for g in GAP_BINS))

    print("\n" + "=" * 104)
    print("  DELTA vs deployed (scale 1.00), and the SUM over both gates")
    print("=" * 104)
    print(f"  {'chi scale':>10s}" + "".join(f"{t[:16]:>20s}" for t in res) + f"{'sum':>12s}")
    for f in SCALES:
        ds = [res[t][f]['ap_total'] - res[t][1.00]['ap_total'] for t in res]
        print(f"  {f:10.2f}" + "".join(f"{d:+20.4f}" for d in ds) + f"{sum(ds):+12.4f}")
    print("\n  k_chi the simulator would need for each scale (currently a_coef/2 = 1.75):")
    for f in SCALES:
        print(f"    scale {f:.2f}  ->  k_chi {1.75 * f:.2f}  ->  a_coef {3.5 * f:.2f}")

    json.dump({t: {str(k): v for k, v in res[t].items()} for t in res},
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/chi_scale_ap.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
