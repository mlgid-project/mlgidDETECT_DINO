"""Phase X — is the second peak MISSING from the objectness map, or just out-ranked?

Phase W showed the second peak of a close pair is absent from the 900 selected proposals. That was
read as "the encoder never proposes it", but `out['interm_outputs']` is built from `tgt_undetach`,
i.e. the ALREADY-GATHERED top-900 (deformable_transformer.py:420). The full objectness map over all
10,880 tokens is `enc_outputs_class_unselected` (line 409) and is never returned. So phase W cannot
distinguish:

  (i)  SELECTION  — the map has two maxima at the pair, but the second ranks below 900th globally.
                    Fix is cheap: more queries, or local-max selection instead of a global top-k.
  (ii) REPRESENTATION — the map has ONE broad maximum at the midpoint. Fix is the encoder itself.

This probe captures the unselected map with a forward hook on `enc_out_class_embed` (the module is
called twice per forward: once with ~10,880 tokens, once with the 900 — we keep the large call) and
asks, per planted pair:

  * n_max   how many local maxima along chi the objectness has in the pair's neighbourhood
  * rank2   the GLOBAL rank (1..10880) of the best token near the second peak
            rank2 <= 900  -> it WAS selected (contradicts phase W; would mean a matching artefact)
            rank2 >  900  -> selection problem, and by how much
            no token      -> representation problem
  * level   which pyramid level the near-pair tokens live on, and the level mix of the whole top-900

The level question matters on its own: at stride 8 two peaks 8 px apart in chi are ADJACENT tokens,
so no valley can exist between them at that level. The 5-scale model has a stride-4 level where they
are 2 tokens apart, yet showed the same 16 px wall -- which would be explained if its proposals still
come overwhelmingly from the stride-8 level.

GPU, ~3 min. See tmp_diag/run_objectness.sbatch.
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
from diagnostics.prominence_probe import build_model_from_ckpt, OUT

H_IMG, W_IMG = 512, 1024


def layout_for(n_tokens):
    """Pyramid layout implied by the token count: 4-scale is [8,16,32,64] = 10,880 tokens,
    5-scale adds a stride-4 level = 43,648. Derived, not hardcoded, so both models work."""
    for strides in ([8, 16, 32, 64], [4, 8, 16, 32, 64]):
        shapes = [(H_IMG // s, W_IMG // s) for s in strides]
        if sum(h * w for h, w in shapes) == n_tokens:
            offs, o = [], 0
            for h, w in shapes:
                offs.append(o); o += h * w
            return strides, shapes, offs
    raise ValueError(f"no known pyramid layout with {n_tokens} tokens")


def probe(name, ckpt, cfg_file, data, dev):
    """For each planted pair ask, ON THE TOKEN GRID ITSELF:
         can this level even represent two peaks here (do they fall in DIFFERENT token rows)?
         if so, are BOTH of those tokens in the selected top-900, and where do they rank?
       Working in token units rather than pixels avoids the confound in the first version, where a
       +-4 px window at a stride of 8 px could not tell 'the token on peak 2' from 'the one token of
       the merged blob'."""
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    rec = {}

    def hook(mod, inp, outp):
        if inp[0].shape[1] > 5000:              # the UNSELECTED call, every token
            rec['logits'] = outp.detach()
    hnd = model.transformer.enc_out_class_embed.register_forward_hook(hook)

    lay = {}
    lvl_all = None
    rows, profiles = {}, {}
    with torch.no_grad():
        for sep, frames in data.items():
            acc = None
            prof = []
            for img, gt in frames:
                rec.clear()
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                model(t.repeat(1, a.num_channels, 1, 1))
                if 'logits' not in rec:
                    continue
                sc = rec['logits'][0].sigmoid().max(-1).values.cpu().numpy()
                if not lay:
                    st, sh, of = layout_for(len(sc))
                    lay.update(strides=st, shapes=sh, offs=of, n=len(sc))
                    lvl_all = np.zeros(len(st), np.int64)
                    print(f"  pyramid: {len(sc)} tokens, strides {st}", flush=True)
                st, sh, of = lay['strides'], lay['shapes'], lay['offs']
                nlv = len(st)
                if acc is None:
                    acc = dict(n=0, distinct=np.zeros(nlv), both900=np.zeros(nlv),
                               rank2=[[] for _ in range(nlv)])
                order = np.argsort(-sc)
                rank_of = np.empty(len(sc), np.int64)
                rank_of[order] = np.arange(1, len(sc) + 1)
                if lvl_all.sum() == 0:
                    lv_of = np.searchsorted(of, np.arange(len(sc)), side='right') - 1
                    for i in order[:900]:
                        lvl_all[lv_of[i]] += 1

                for i in range(0, len(gt) - 1, 2):
                    gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                    pc = sorted(float((gt[j, 1] + gt[j, 3]) / 2) for j in (i, i + 1))
                    acc['n'] += 1
                    for l in range(nlv):
                        s_, (hh, ww) = st[l], sh[l]
                        col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                        r1 = int(np.clip(round(pc[0] / s_ - 0.5), 0, hh - 1))
                        r2 = int(np.clip(round(pc[1] / s_ - 0.5), 0, hh - 1))
                        if r1 == r2:
                            continue                      # grid cannot represent two peaks here
                        acc['distinct'][l] += 1
                        k1 = of[l] + r1 * ww + col
                        k2 = of[l] + r2 * ww + col
                        weak = max(int(rank_of[k1]), int(rank_of[k2]))
                        acc['rank2'][l].append(weak)
                        if weak <= 900:
                            acc['both900'][l] += 1
                    # objectness profile along chi at the finest level, for the figure
                    if sep in (8, 16) and len(prof) < 40:
                        s_, (hh, ww) = st[0], sh[0]
                        col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                        lo = int(np.clip(round((pc[0] - 24) / s_), 0, hh - 1))
                        hi = int(np.clip(round((pc[1] + 24) / s_), 0, hh - 1))
                        grid = sc[of[0]:of[0] + hh * ww].reshape(hh, ww)
                        seg = grid[lo:hi + 1, col]
                        if len(seg) >= 4:
                            prof.append((np.arange(lo, hi + 1) * s_ + s_ / 2 -
                                         (pc[0] + pc[1]) / 2, seg.copy()))
            n = max(acc['n'], 1)
            rows[sep] = dict(
                n=acc['n'], strides=lay['strides'],
                distinct=(acc['distinct'] / n).tolist(),
                both900=[(acc['both900'][l] / acc['distinct'][l]) if acc['distinct'][l] else float('nan')
                         for l in range(len(lay['strides']))],
                med_rank2=[float(np.median(acc['rank2'][l])) if acc['rank2'][l] else float('nan')
                           for l in range(len(lay['strides']))])
            profiles[sep] = prof
            r = rows[sep]
            f = lambda v: "  ".join(f"{x:.2f}" if x == x else "  - " for x in v)
            g = lambda v: "  ".join(f"{x:6.0f}" if x == x else "     -" for x in v)
            print(f"  sep={sep:3d} n={r['n']:4d} | distinct rows {f(r['distinct'])} "
                  f"| both in top-900 {f(r['both900'])} | median worse rank {g(r['med_rank2'])}",
                  flush=True)
    hnd.remove()
    del model
    torch.cuda.empty_cache()
    tot = max(int(lvl_all.sum()), 1)
    print("  level mix of the selected top-900: " + "  ".join(
        f"stride{lay['strides'][i]} {int(lvl_all[i])} ({100*lvl_all[i]/tot:.1f}%)"
        for i in range(len(lay['strides']))), flush=True)
    return rows, profiles, lvl_all.tolist(), lay['strides']


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  (pyramid layout derived per model)", flush=True)
    data = L.make_images(dev)
    summary = {}
    figdata = {}
    for name, ckpt, cfg_file in L.MODELS:
        print(f"\n########## {name} ##########", flush=True)
        rows, prof, lvl, st = probe(name, ckpt, cfg_file, data, dev)
        summary[name] = dict(rows={str(k): v for k, v in rows.items()},
                             top900_levels=lvl, strides=st)
        figdata[name] = prof

    print("\n" + "=" * 96)
    print("  Can the token grid even SEPARATE the pair, and are both tokens selected?")
    for nm in summary:
        st = summary[nm]['strides']
        print(f"\n  --- {nm}   (levels: {'  '.join('stride'+str(x) for x in st)})")
        print(f"  {'sep':>5s}  {'pair in DISTINCT token rows':>30s}   {'both tokens in top-900':>26s}")
        for sep in L.SEPS:
            r = summary[nm]['rows'][str(sep)]
            f = lambda v: " ".join(f"{x:6.2f}" if x == x else "     -" for x in v)
            print(f"  {sep:5d}  {f(r['distinct']):>30s}   {f(r['both900']):>26s}")
    print("\n  level mix of the selected top-900 (where proposals actually come from):")
    for nm in summary:
        lv, st = summary[nm]['top900_levels'], summary[nm]['strides']
        tot = max(sum(lv), 1)
        print(f"    {nm:>16s}  " + "  ".join(
            f"stride{st[i]} {lv[i]:4d} ({100*lv[i]/tot:4.1f}%)" for i in range(len(st))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, sep in zip(axes, (8, 16)):
        for nm, c in zip(figdata, ('tab:blue', 'tab:red', 'tab:green')):
            ps = figdata[nm].get(sep, [])
            for k, (xx, yy) in enumerate(ps[:25]):
                ax.plot(xx, yy, color=c, alpha=.28, lw=.9,
                        label=nm if k == 0 else None)
        for d in (-sep / 2, sep / 2):
            ax.axvline(d, ls=':', c='k', lw=1)
        ax.set_title(f'objectness along χ through a {sep} px pair (stride-8 level)')
        ax.set_xlabel('χ offset from pair midpoint (px)'); ax.set_ylabel('objectness')
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'objectness_profile.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}")
    json.dump(summary, open(os.path.join(OUT, 'objectness_probe.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
