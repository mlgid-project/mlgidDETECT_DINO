"""Phase X.2 — DIRECT confirmation: where do the two selected tokens actually point their boxes?

Established so far:
  * phase X   at 8 px the two peaks fall in DIFFERENT tokens (1.00) and BOTH tokens are selected
              into the top-900 (ssl1 0.75, clusters 1.00). So neither the token grid nor the
              top-k selection is the limit.
  * phase W   yet 94.8% of pairs have no encoder BOX near the second peak.

Those two together IMPLY the proposal-stage box regression maps both tokens onto one midpoint box —
but that was inferred by triangulation, never measured. This measures it.

The encoder's proposal is  box = sigmoid( enc_out_bbox_embed(memory) + output_proposals )  where
`output_proposals` is the per-token anchor (deformable_transformer.py:410). Note what that means:
the two tokens' ANCHORS already differ by exactly one token spacing, so if their features were
merely indistinguishable the two boxes would come out one spacing apart — i.e. roughly correct.
A merge therefore requires the MLP delta to ACTIVELY pull both boxes together. This probe separates:

    d_anchor  chi separation of the two tokens' anchors        (= one token spacing, by construction)
    d_box     chi separation of the two PREDICTED boxes
    d_true    the planted separation

    d_box ~ d_true    -> boxes are right; the loss is later (decoder / matching)
    d_box ~ 0         -> the box head actively collapses them  <- the triangulated hypothesis
    d_box ~ d_anchor  -> the delta is doing nothing; boxes just inherit the grid

`gen_encoder_output_proposals` is a module-level function, so it is monkeypatched to capture the
anchors; `enc_out_bbox_embed` is an nn.Module, so it gets a forward hook for the delta.

GPU, ~3 min. See tmp_diag/run_proposal_box.sbatch.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('LADDER_AXIS', 'chi')
os.environ.setdefault('LADDER_ISO', '1')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import diagnostics.separation_ladder as L
import models.dino.deformable_transformer as DT
from diagnostics.objectness_probe import layout_for, H_IMG, W_IMG
from diagnostics.prominence_probe import build_model_from_ckpt, OUT


def probe(name, ckpt, cfg_file, data, dev):
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    rec = {}

    _orig_gen = DT.gen_encoder_output_proposals

    def _gen(*args, **kw):
        om, op = _orig_gen(*args, **kw)
        rec['anchor'] = op.detach()
        return om, op
    DT.gen_encoder_output_proposals = _gen

    def hook(mod, inp, outp):
        if inp[0].shape[1] > 5000:
            rec['delta'] = outp.detach()
    hnd = model.transformer.enc_out_bbox_embed.register_forward_hook(hook)

    lay, rows = {}, {}
    with torch.no_grad():
        for sep, frames in data.items():
            acc = dict(n=0, d_box=[], d_anc=[], err1=[], err2=[], both=0, h=[], span=[])
            for img, gt in frames:
                rec.clear()
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                model(t.repeat(1, a.num_channels, 1, 1))
                if 'delta' not in rec or 'anchor' not in rec:
                    continue
                anc = rec['anchor'][0]                       # (N,4) unsigmoid
                box = (rec['delta'][0] + anc).sigmoid()      # (N,4) cxcywh in [0,1]
                anc_s = anc.sigmoid()
                chi_box = (box[:, 1] * H_IMG).cpu().numpy()  # y = chi
                h_box = (box[:, 3] * H_IMG).cpu().numpy()    # chi EXTENT of the predicted box
                chi_anc = (anc_s[:, 1] * H_IMG).cpu().numpy()
                if not lay:
                    st, sh, of = layout_for(box.shape[0])
                    lay.update(strides=st, shapes=sh, offs=of)
                    print(f"  pyramid: {box.shape[0]} tokens, finest stride {st[0]}", flush=True)
                st, sh, of = lay['strides'], lay['shapes'], lay['offs']
                s_, (hh, ww) = st[0], sh[0]                  # finest = dominant level for all 3

                for i in range(0, len(gt) - 1, 2):
                    gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                    pc = sorted(float((gt[j, 1] + gt[j, 3]) / 2) for j in (i, i + 1))
                    col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                    r1 = int(np.clip(round(pc[0] / s_ - 0.5), 0, hh - 1))
                    r2 = int(np.clip(round(pc[1] / s_ - 0.5), 0, hh - 1))
                    if r1 == r2:
                        continue                             # grid cannot represent two here
                    acc['n'] += 1
                    k1, k2 = of[0] + r1 * ww + col, of[0] + r2 * ww + col
                    acc['d_box'].append(abs(float(chi_box[k1] - chi_box[k2])))
                    acc['d_anc'].append(abs(float(chi_anc[k1] - chi_anc[k2])))
                    e1 = abs(float(chi_box[k1]) - pc[0])
                    e2 = abs(float(chi_box[k2]) - pc[1])
                    acc['err1'].append(e1); acc['err2'].append(e2)
                    acc['h'].append(0.5 * (float(h_box[k1]) + float(h_box[k2])))
                    # chi extent actually spanned by the pair, outer edge to outer edge
                    acc['span'].append(sep + L.BOX_H)
                    if max(e1, e2) <= max(3.0, 0.35 * sep):
                        acc['both'] += 1
            n = max(acc['n'], 1)
            med = lambda v: float(np.median(v)) if v else float('nan')
            rows[sep] = dict(n=acc['n'], d_box=med(acc['d_box']), d_anc=med(acc['d_anc']),
                             err1=med(acc['err1']), err2=med(acc['err2']),
                             both_on_peak=acc['both'] / n,
                             h_pred=med(acc['h']), h_single=L.BOX_H, h_span=med(acc['span']))
            r = rows[sep]
            print(f"  sep={sep:3d} n={r['n']:4d} | d_anchor {r['d_anc']:5.1f} d_box {r['d_box']:6.2f} "
                  f"| err {r['err1']:5.2f}/{r['err2']:5.2f} | box chi-height {r['h_pred']:6.2f}  "
                  f"(one peak {r['h_single']:.1f}, pair spans {r['h_span']:.1f})", flush=True)
    hnd.remove()
    DT.gen_encoder_output_proposals = _orig_gen
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}", flush=True)
    data = L.make_images(dev)
    summary = {}
    for name, ckpt, cfg_file in L.MODELS:
        print(f"\n########## {name} ##########", flush=True)
        summary[name] = {str(k): v for k, v in probe(name, ckpt, cfg_file, data, dev).items()}

    print("\n" + "=" * 92)
    print("  Do the two selected tokens point their boxes at the two peaks, or at one midpoint?")
    print(f"  {'sep':>5s} {'model':>16s} {'d_true':>7s} {'d_anchor':>9s} {'d_box':>8s} {'both on peak':>13s}")
    for sep in L.SEPS:
        for nm in summary:
            r = summary[nm][str(sep)]
            print(f"  {sep:5d} {nm:>16s} {sep:7d} {r['d_anc']:9.1f} {r['d_box']:8.2f} "
                  f"{r['both_on_peak']:13.3f}")

    print("\n" + "=" * 92)
    print("  BOX SIZE TEST — does the merged box describe ONE peak, or a blob spanning BOTH?")
    print("    h ~ one-peak height   -> single-object hypothesis at the midpoint (unstable matching)")
    print("    h ~ pair span         -> the head deliberately describes one merged object")
    print(f"  {'sep':>5s} {'model':>16s} {'pred chi-height':>16s} {'one peak':>9s} {'pair span':>10s}")
    for sep in L.SEPS:
        for nm in summary:
            r = summary[nm][str(sep)]
            print(f"  {sep:5d} {nm:>16s} {r['h_pred']:16.2f} {r['h_single']:9.1f} {r['h_span']:10.1f}")

    fig, ax = plt.subplots(figsize=(7.4, 5))
    for nm, c in zip(summary, ('tab:blue', 'tab:red', 'tab:green')):
        ax.plot(L.SEPS, [summary[nm][str(s)]['d_box'] for s in L.SEPS], 'o-', color=c, label=nm)
    ax.plot(L.SEPS, L.SEPS, 'k--', lw=1, label='perfect (d_box = d_true)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(L.SEPS); ax.set_xticklabels(L.SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('χ-separation of the two predicted proposal boxes (px)')
    ax.set_title('Phase X.2: does the proposal box head collapse close pairs?')
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proposal_box.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}")
    json.dump(summary, open(os.path.join(OUT, 'proposal_box_probe.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
