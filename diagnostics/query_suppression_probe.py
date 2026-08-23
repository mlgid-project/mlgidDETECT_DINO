"""Phase W (step 0) -- below the resolution wall, is the second peak SUPPRESSED or NEVER PROPOSED?

Phase V established that below ~12-16 px in chi the model emits exactly ONE box for two clearly
separable peaks (discriminator: 1.00 one-box at 8 px), with sub-pixel localisation above the wall.
Stride, input resolution, training distribution, NMS, regression precision and the Swin window are
all eliminated. What is left is the head. This probe asks WHERE in the head it goes wrong, and it
costs nothing -- same images, same checkpoints, no training.

The leading hypothesis is DINO's contrastive denoising. With `dn_box_noise_scale = 0.4`
(dn_components.py:81-89) negative DN queries are placed 0.2-0.4 x box-dimension from a true box --
for an 8.5 px chi box that is 1.7-3.4 px, almost exactly where real second peaks live (median real
chi-gap 3.9 px). The model is therefore trained to answer "no object" to a box sitting where the
neighbour actually is. If that is the mechanism, a query should still LAND on the second peak and
simply be scored as background.

Three-way discrimination, using the fact that DINO is two-stage so both stages are visible:
    out['interm_outputs']  -> the ENCODER's proposals (what the decoder was given)
    out['pred_logits/boxes'] -> the DECODER's final output

  (A) encoder proposes both, decoder keeps a box on peak 2, score < threshold
        => CLASSIFICATION SUPPRESSION. The DN lever is the right one.
  (B) encoder proposes both, decoder has no box on peak 2 (they collapse together)
        => DECODER MERGING (self-attention / query interaction). DN scale will not fix it;
           the fix is in decoder query design.
  (C) encoder never proposes peak 2
        => PROPOSAL SELECTION. Neither of the above; look at two-stage top-k and num_queries.

Everything is read RAW: sigmoid over all 900 queries, no top-k (the deployed path keeps 225), no NMS,
no score threshold -- so a suppressed query is visible even at score 1e-4.

GPU, ~3 min. See tmp_diag/run_query_suppression.sbatch.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('LADDER_AXIS', 'chi')     # the failing axis
os.environ.setdefault('LADDER_ISO', '1')        # matched stimulus (sigma_q == sigma_chi == 2.43 px)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import diagnostics.separation_ladder as L
from util.configuration import Config
from util.postprocessing import box_cxcywh_to_xyxy
from diagnostics.prominence_probe import build_model_from_ckpt, OUT

THR = 0.10          # the matched operating point the iso-chi ladder selected for every model
QTOL = 15.0         # q-window around the pair, as in the ladder's discriminator


def zones(sep):
    """Radius that keeps 'on peak 1' / 'on midpoint' / 'on peak 2' disjoint. Below sep~6 the three
    zones cannot be separated at all, so those rungs are reported but not read."""
    return max(1.5, 0.30 * sep), (0.30 * sep) < (0.5 * sep) and sep >= 6


def stage_arrays(cfg, logits, boxes):
    """Raw sigmoid scores and pixel xyxy for EVERY query -- no top-k, no NMS, no threshold."""
    # to CPU first: box_cxcywh_to_xyxy builds its scale tensor on CPU (the ladder reached it via
    # numpy, so it never hit this)
    logits, boxes = logits.detach().cpu(), boxes.detach().cpu()
    prob = logits.sigmoid()[0]                       # (nq, ncls)
    score = prob.max(-1).values.numpy()              # (nq,)
    bx = box_cxcywh_to_xyxy(cfg, boxes).numpy()      # (nq, 4) pixels, x=q  y=chi
    return score, (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2


def probe(name, ckpt, cfg_file, data, dev):
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    rows = {}
    scatter = {}
    with torch.no_grad():
        for sep, frames in data.items():
            R, readable = zones(sep)
            acc = dict(n=0, dec_both=0, dec_sup=0, dec_absent=0, enc_both=0, enc_absent=0,
                       win=[], lose=[], encwin=[], enclose=[])
            pts = []
            for img, gt in frames:
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                t = t.repeat(1, a.num_channels, 1, 1)
                out = model(t)
                d_s, d_q, d_c = stage_arrays(cfg, out['pred_logits'], out['pred_boxes'])
                e = out.get('interm_outputs')
                e_s, e_q, e_c = stage_arrays(cfg, e['pred_logits'], e['pred_boxes']) \
                    if e is not None else (None, None, None)

                for i in range(0, len(gt) - 1, 2):
                    gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                    p_chi = [float((gt[j, 1] + gt[j, 3]) / 2) for j in (i, i + 1)]
                    mid = sum(p_chi) / 2
                    acc['n'] += 1

                    def best_on(s, q, c, peak):
                        m = (np.abs(q - gq) < QTOL) & (np.abs(c - peak) <= R)
                        return (float(s[m].max()), int(m.sum())) if m.any() else (float('nan'), 0)

                    (b0, n0), (b1, n1) = [best_on(d_s, d_q, d_c, p) for p in p_chi]
                    # winner / loser by score; a peak with NO query counts as absent, not low-scoring
                    pair = sorted([(b0, n0), (b1, n1)],
                                  key=lambda z: (-1e9 if z[1] == 0 else z[0]), reverse=True)
                    (bw, nw), (bl, nl) = pair
                    if nw:
                        acc['win'].append(bw)
                    if nl:
                        acc['lose'].append(bl)
                        acc['dec_both'] += 1
                        if bl < THR:
                            acc['dec_sup'] += 1          # (A) a box IS there, scored as background
                    else:
                        acc['dec_absent'] += 1           # (B) no box on the second peak at all

                    if e_s is not None:
                        (c0, m0), (c1, m1) = [best_on(e_s, e_q, e_c, p) for p in p_chi]
                        ep = sorted([(c0, m0), (c1, m1)],
                                    key=lambda z: (-1e9 if z[1] == 0 else z[0]), reverse=True)
                        if ep[1][1]:
                            acc['enc_both'] += 1
                            acc['enclose'].append(ep[1][0])
                        else:
                            acc['enc_absent'] += 1       # (C) encoder never proposed it
                        if ep[0][1]:
                            acc['encwin'].append(ep[0][0])

                    # raw scatter of every decoder query near the pair: chi offset from mid vs score
                    m = (np.abs(d_q - gq) < QTOL) & (np.abs(d_c - mid) < sep / 2 + 20)
                    for off, sc_ in zip(d_c[m] - mid, d_s[m]):
                        pts.append((float(off), float(sc_)))

            n = max(acc['n'], 1)
            rows[sep] = dict(
                n=acc['n'], readable=readable,
                dec_second_box=acc['dec_both'] / n,
                dec_suppressed=acc['dec_sup'] / n,
                dec_absent=acc['dec_absent'] / n,
                enc_second_box=acc['enc_both'] / n,
                enc_absent=acc['enc_absent'] / n,
                med_win=float(np.median(acc['win'])) if acc['win'] else float('nan'),
                med_lose=float(np.median(acc['lose'])) if acc['lose'] else float('nan'),
                med_encwin=float(np.median(acc['encwin'])) if acc['encwin'] else float('nan'),
                med_enclose=float(np.median(acc['enclose'])) if acc['enclose'] else float('nan'))
            scatter[sep] = pts
            r = rows[sep]
            print(f"  sep={sep:3d}px n={r['n']:4d}  DEC 2nd-box {r['dec_second_box']:.3f} "
                  f"(suppressed {r['dec_suppressed']:.3f} / absent {r['dec_absent']:.3f})  "
                  f"score win {r['med_win']:.3f} lose {r['med_lose']:.3f}   "
                  f"ENC 2nd-box {r['enc_second_box']:.3f} lose {r['med_enclose']:.3f}"
                  f"{'' if r['readable'] else '   [zones overlap - not readable]'}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows, scatter


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  AXIS={L.AXIS} ISO={L.ISO}  thr={THR}\n"
          f"stimulus: box {L.BOX_W:.2f} x {L.BOX_H:.2f} -> sigma_q=sigma_chi={L.SIGMA:.2f} px\n"
          "building the fixed image set", flush=True)
    data = L.make_images(dev)

    summary = {}
    for name, ckpt, cfg_file in L.MODELS:
        print(f"\n########## {name} ##########", flush=True)
        rows, scatter = probe(name, ckpt, cfg_file, data, dev)
        summary[name] = {str(k): v for k, v in rows.items()}
        if name == L.MODELS[0][0]:
            first_scatter = scatter

    print("\n" + "=" * 100)
    print("  VERDICT TABLE -- of pairs where the model failed, what happened to the SECOND peak?")
    print("  (A) suppressed = a decoder box sits on it, score < thr   -> DN / classification")
    print("  (B) absent     = no decoder box on it at all             -> decoder merging")
    print("  (C) enc absent = encoder never proposed it               -> proposal selection")
    for name in summary:
        print(f"\n  --- {name}")
        print(f"  {'sep':>5s} {'(A) suppressed':>15s} {'(B) dec absent':>15s} {'(C) enc absent':>15s}"
              f" {'med score lose':>15s} {'med score win':>14s}")
        for sep in L.SEPS:
            r = summary[name][str(sep)]
            flag = '' if r['readable'] else '  [zones overlap]'
            print(f"  {sep:5d} {r['dec_suppressed']:15.3f} {r['dec_absent']:15.3f} "
                  f"{r['enc_absent']:15.3f} {r['med_lose']:15.4f} {r['med_win']:14.4f}{flag}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for name, c in zip(summary, ('tab:blue', 'tab:red', 'tab:green')):
        s = [L.SEPS.index(x) for x in L.SEPS]
        ax.plot(L.SEPS, [summary[name][str(x)]['med_lose'] for x in L.SEPS], 'o-', color=c,
                label=f'{name} 2nd peak')
        ax.plot(L.SEPS, [summary[name][str(x)]['med_win'] for x in L.SEPS], 's--', color=c,
                alpha=.45, label=f'{name} 1st peak')
    ax.axhline(THR, ls=':', c='k', label=f'threshold {THR}')
    ax.set_xscale('log'); ax.set_xticks(L.SEPS); ax.set_xticklabels(L.SEPS)
    ax.set_xlabel('planted chi-separation (px)'); ax.set_ylabel('median best raw score')
    ax.set_title('Is the second peak scored as background?'); ax.legend(fontsize=7)

    ax = axes[1]
    for sep, c in zip((6, 8, 12, 16), ('tab:purple', 'tab:orange', 'tab:brown', 'tab:cyan')):
        p = np.array(first_scatter[sep])
        if len(p):
            ax.scatter(p[:, 0], np.maximum(p[:, 1], 1e-5), s=5, alpha=.35, color=c, label=f'{sep} px')
            for d in (-sep / 2, sep / 2):
                ax.axvline(d, ls=':', lw=.8, color=c)
    ax.set_yscale('log'); ax.axhline(THR, ls='--', c='k', lw=1)
    ax.set_xlabel('chi offset of query from pair midpoint (px)'); ax.set_ylabel('raw score')
    ax.set_title(f'{L.MODELS[0][0]}: every decoder query near a pair'); ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_suppression.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nsaved {out}")
    json.dump(summary, open(os.path.join(OUT, 'query_suppression.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
