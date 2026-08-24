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
STRIDES = [8, 16, 32, 64]
SHAPES = [(H_IMG // s, W_IMG // s) for s in STRIDES]
OFFS, _o = [], 0
for _h, _w in SHAPES:
    OFFS.append(_o); _o += _h * _w
NTOK = _o


IDXS = np.arange(NTOK)
LVS = np.searchsorted(OFFS, IDXS, side='right') - 1
CHIS = np.empty(NTOK); QS = np.empty(NTOK)
for _li, (_hh, _ww) in enumerate(SHAPES):
    _sl = slice(OFFS[_li], OFFS[_li] + _hh * _ww)
    _rr = IDXS[_sl] - OFFS[_li]
    CHIS[_sl] = (_rr // _ww + 0.5) * STRIDES[_li]
    QS[_sl] = (_rr % _ww + 0.5) * STRIDES[_li]


def token_xy(idx):
    """token index -> (level, chi_px, q_px) at the token's centre."""
    lvl = int(np.searchsorted(OFFS, idx, side='right') - 1)
    r = idx - OFFS[lvl]
    h, w = SHAPES[lvl]
    s = STRIDES[lvl]
    return lvl, (r // w + 0.5) * s, (r % w + 0.5) * s


def probe(name, ckpt, cfg_file, data, dev):
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    rec = {}

    def hook(mod, inp, outp):
        if inp[0].shape[1] > 5000:              # the UNSELECTED call, all ~10,880 tokens
            rec['logits'] = outp.detach()
    hnd = model.transformer.enc_out_class_embed.register_forward_hook(hook)

    lvl_all = np.zeros(len(STRIDES), dtype=np.int64)     # level mix of the whole top-900
    rows, profiles = {}, {}
    with torch.no_grad():
        for sep, frames in data.items():
            R = max(4.0, 0.45 * sep)                     # px window around a peak, in chi
            acc = dict(n=0, none=0, sel=0, rank=[], nmax=[], lvl=np.zeros(len(STRIDES), np.int64))
            prof = []
            for img, gt in frames:
                rec.clear()
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                out = model(t.repeat(1, a.num_channels, 1, 1))
                if 'logits' not in rec:
                    continue
                sc = rec['logits'][0].sigmoid().max(-1).values.cpu().numpy()   # (NTOK,)
                order = np.argsort(-sc)
                rank_of = np.empty(len(sc), np.int64)
                rank_of[order] = np.arange(1, len(sc) + 1)
                if lvl_all.sum() == 0:
                    for i in order[:900]:
                        lvl_all[token_xy(int(i))[0]] += 1

                lv0_h, lv0_w = SHAPES[0]
                grid = sc[OFFS[0]:OFFS[0] + lv0_h * lv0_w].reshape(lv0_h, lv0_w)
                for i in range(0, len(gt) - 1, 2):
                    gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                    pc = sorted(float((gt[j, 1] + gt[j, 3]) / 2) for j in (i, i + 1))
                    acc['n'] += 1
                    col = int(np.clip(round(gq / STRIDES[0] - 0.5), 0, lv0_w - 1))
                    lo = int(np.clip(round((pc[0] - 24) / STRIDES[0]), 0, lv0_h - 1))
                    hi = int(np.clip(round((pc[1] + 24) / STRIDES[0]), 0, lv0_h - 1))
                    seg = grid[lo:hi + 1, col]
                    if len(seg) >= 3:                    # count interior local maxima
                        acc['nmax'].append(int(((seg[1:-1] > seg[:-2]) &
                                                (seg[1:-1] >= seg[2:])).sum()))
                    if sep in (8, 16) and len(prof) < 60 and len(seg) >= 5:
                        prof.append((np.arange(lo, hi + 1) * STRIDES[0] + 4 -
                                     (pc[0] + pc[1]) / 2, seg.copy()))
                    # best token near the SECOND peak, anywhere in the pyramid
                    near = (np.abs(QS - gq) < 12) & (np.abs(CHIS - pc[1]) <= R)
                    if not near.any():
                        acc['none'] += 1
                    else:
                        j = np.flatnonzero(near)[np.argmax(sc[near])]
                        acc['rank'].append(int(rank_of[j]))
                        acc['lvl'][int(LVS[j])] += 1
                        if rank_of[j] <= 900:
                            acc['sel'] += 1
            n = max(acc['n'], 1)
            rows[sep] = dict(n=acc['n'], no_token=acc['none'] / n, selected=acc['sel'] / n,
                             med_rank=float(np.median(acc['rank'])) if acc['rank'] else float('nan'),
                             p25_rank=float(np.percentile(acc['rank'], 25)) if acc['rank'] else float('nan'),
                             med_nmax=float(np.median(acc['nmax'])) if acc['nmax'] else float('nan'),
                             frac_2max=float(np.mean(np.array(acc['nmax']) >= 2)) if acc['nmax'] else float('nan'),
                             lvl=acc['lvl'].tolist())
            profiles[sep] = prof
            r = rows[sep]
            print(f"  sep={sep:3d}  n={r['n']:4d}  2+ maxima {r['frac_2max']:.3f}  "
                  f"median #maxima {r['med_nmax']:.1f}  |  no token {r['no_token']:.3f}  "
                  f"in top-900 {r['selected']:.3f}  median rank {r['med_rank']:8.0f}  "
                  f"(p25 {r['p25_rank']:.0f})", flush=True)
    hnd.remove()
    del model
    torch.cuda.empty_cache()
    print(f"  level mix of the top-900: " +
          "  ".join(f"stride{STRIDES[i]} {lvl_all[i]}" for i in range(len(STRIDES))))
    return rows, profiles, lvl_all.tolist()


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  tokens={NTOK}  shapes={SHAPES}", flush=True)
    data = L.make_images(dev)
    summary = {}
    figdata = {}
    for name, ckpt, cfg_file in L.MODELS:
        print(f"\n########## {name} ##########", flush=True)
        rows, prof, lvl = probe(name, ckpt, cfg_file, data, dev)
        summary[name] = dict(rows={str(k): v for k, v in rows.items()}, top900_levels=lvl)
        figdata[name] = prof

    print("\n" + "=" * 96)
    print("  VERDICT — is the second peak in the objectness map at all, and where does it rank?")
    print(f"  {'sep':>5s} {'model':>16s} {'2+ maxima':>10s} {'no token':>9s} {'in top-900':>11s} {'median rank':>12s}")
    for sep in L.SEPS:
        for nm in summary:
            r = summary[nm]['rows'][str(sep)]
            print(f"  {sep:5d} {nm:>16s} {r['frac_2max']:10.3f} {r['no_token']:9.3f} "
                  f"{r['selected']:11.3f} {r['med_rank']:12.0f}")
    print("\n  level mix of the selected top-900 (where proposals actually come from):")
    for nm in summary:
        lv = summary[nm]['top900_levels']
        tot = max(sum(lv), 1)
        print(f"    {nm:>16s}  " + "  ".join(
            f"stride{STRIDES[i]} {lv[i]:4d} ({100*lv[i]/tot:4.1f}%)" for i in range(len(STRIDES))))

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
