"""Phase AD.2 — the sim/real gap where it actually matters: the ENCODER OUTPUT.

AD measured at `input_proj`, before the transformer. The encoder's output memory is the better place
and the user's suggestion: those 10,880 tokens are exactly what `enc_out_bbox_embed` scores to pick
the 900 decoder queries, so a difference there is a difference that changes detections, whereas a
difference at `input_proj` may simply be absorbed by six encoder layers.

Caveat carried, not dodged: the encoder is ssl1's, trained on simulation, so this ruler is if
anything MORE sim-adapted than the backbone was. The **organic vs 41** control does most of the work
against that -- it is two real datasets passed through the identical ruler, so a uniform bias cancels
out of the comparison.

Token -> level is reconstructed deterministically from the 512x1024 polar grid at strides 8/16/32/64:
64x128 + 32x64 + 16x32 + 8x16 = 8192 + 2048 + 512 + 128 = 10880, which matches the known query pool.
Peak vs background split, streaming covariance, rank-based Fisher AUC and the three pairs are AD's,
unchanged.

THE VERDICT LINE, same as AD: is sim-vs-41 below the organic-vs-41 real-real control?

GPU, ~15 min. See tmp_diag/run_encgap.sbatch.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG
from diagnostics.domain_gap_probe import (Acc, coral, meandist, auc_fisher, token_mask,
                                          real_frames, sim_frames, DSETS, N_SIM, PAIRS, CLASSES)

SHAPES = [(64, 128), (32, 64), (16, 32), (8, 16)]      # strides 8/16/32/64 on 512x1024


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("###### AD.2: sim/real gap at the ENCODER OUTPUT ######", flush=True)
    print(f"device={dev}  SINGLE MODEL ssl1  N_SIM={N_SIM}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    model.eval()

    grab = {}

    def hook(_m, _i, out):
        grab['mem'] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
    h = model.transformer.encoder.register_forward_hook(hook)

    names = ["sim"] + [n for n, _ in DSETS]
    acc = {}
    sizes = [H * W for H, W in SHAPES]
    offs = np.cumsum([0] + sizes)

    def run(name, frames):
        n = 0
        with torch.no_grad():
            for img, boxes in frames():
                _ = model(img)
                mem = grab.get('mem')
                if mem is None:
                    continue
                if mem.dim() == 3 and mem.shape[0] != 1:       # [N, B, C] -> [B, N, C]
                    mem = mem.permute(1, 0, 2)
                M = mem[0].float().cpu()                        # [10880, 256]
                if n == 0:
                    print(f"  {name}: encoder memory {tuple(mem.shape)}  "
                          f"(expected total {int(offs[-1])})", flush=True)
                n += 1
                imH, imW = img.shape[-2], img.shape[-1]
                for l, (H, W) in enumerate(SHAPES):
                    lo, hi = int(offs[l]), int(offs[l + 1])
                    if hi > len(M):
                        continue
                    X = M[lo:hi]
                    m = token_mask(boxes, H, W, imH, imW)
                    kp, kb = (name, l, 'peak'), (name, l, 'background')
                    if kp not in acc:
                        acc[kp] = Acc(X.shape[1]); acc[kb] = Acc(X.shape[1])
                    acc[kp].add(X[m]); acc[kb].add(X[~m])
        print(f"  {name}: {n} frames", flush=True)

    run("sim", sim_frames(N_SIM, a.num_channels, dev))
    for nm, pth in DSETS:
        run(nm, real_frames(pth, a.num_channels, dev))
    h.remove()

    lv = sorted({l for (_n, l, _c) in acc})
    print("\n  tokens (peak / background) per level")
    for n in names:
        print(f"  {n:<10s}" + "  ".join(
            f"L{l}: {acc[(n, l, 'peak')].n:>7d}/{acc[(n, l, 'background')].n:>7d}" for l in lv))

    out = {}
    for title, fn in [("CORAL DISTANCE  ||C_A-C_B||_F^2 / 4d^2  — encoder output", coral),
                      ("MEAN DISTANCE  ||mu_A-mu_B||^2 / d  — encoder output", meandist)]:
        print("\n" + "=" * 100); print(f"  {title}"); print("=" * 100)
        for c in CLASSES:
            print(f"\n  {c} tokens")
            print(f"  {'pair':<20s}" + "".join(f"{('level ' + str(l)):>14s}" for l in lv))
            for A, B in PAIRS:
                vals = [fn(acc[(A, l, c)], acc[(B, l, c)]) for l in lv]
                out[f"{fn.__name__}|{c}|{A}-{B}"] = vals
                print(f"  {A + ' vs ' + B:<20s}" + "".join(f"{v:14.4g}" for v in vals))

    print("\n" + "=" * 100)
    print("  SEPARABILITY — held-out AUC, Fisher discriminant.  0.5 = indistinguishable")
    print("=" * 100)
    for c in CLASSES:
        print(f"\n  {c} tokens")
        print(f"  {'pair':<20s}" + "".join(f"{('level ' + str(l)):>14s}" for l in lv))
        for A, B in PAIRS:
            vals = [auc_fisher(acc[(A, l, c)].tokens(), acc[(B, l, c)].tokens()) for l in lv]
            out[f"auc|{c}|{A}-{B}"] = vals
            print(f"  {A + ' vs ' + B:<20s}" + "".join(f"{v:14.3f}" for v in vals))

    print("\n" + "=" * 100)
    print("  VERDICT LINE — sim vs 41 against the organic vs 41 real-real control")
    print("=" * 100)
    for c in CLASSES:
        s = out[f"coral|{c}|sim-41"]; k = out[f"coral|{c}|organic-41"]
        for i, l in enumerate(lv):
            print(f"  {c:<12s} L{l}: {'BELOW' if s[i] < k[i] else 'ABOVE':>5s} control"
                  f"   sim-41 {s[i]:.4g}  vs  organic-41 {k[i]:.4g}"
                  f"   ratio {s[i] / max(k[i], 1e-12):.2f}x")
    json.dump(out, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/encoder_gap.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
