"""Phase Y — is the offset MISSING from the feature, or present and unused?

Phase X.2/X.3 established that the encoder's box head, for a close pair, emits ONE box spanning both
peaks: at a true 8 px gap the two tokens' boxes come out 0.77 px apart and each sits ~4 px from its own
peak. The head is not averaging (the box tracks the pair's full span) — it perceives one elongated
object and describes it accurately.

That leaves two possibilities, and they need opposite responses:
  (a) the 256-number feature the head reads does NOT contain the offset -> the encoder merged the pair
      before the head saw it. Nothing done to the head or its loss can help.
  (b) the offset IS in the feature and the head does not use it -> a training/loss problem, fixable.

The test. The box head is MLP(output_memory[i]) for a SINGLE token i. So take that exact vector and
try to extract the same quantity a completely different way: ridge regression — a straight-line fit,
no capacity to memorise. Target = the signed chi-offset from the token's centre to ITS OWN nearest
planted peak, which is precisely what the head is supposed to output.

If a straight line recovers what a 217M-parameter model trained on this task cannot, the information
is plainly present and the failure is training, not perception. Two controls make that hard to argue
with: a shuffled-label fit (must collapse to the target's own spread) and a frame-level train/test
split (tokens from one frame are correlated; splitting by token would leak).

A second probe point sits on the PRE-ENCODER features (the projected backbone pyramid, captured by a
forward-pre-hook on the encoder). If the offset survives there but not in the encoder output, the loss
is localised to the six encoder layers.

Every frame holds exactly two peaks, so total flux is constant across rungs and only the separation
varies — the fit cannot cheat on brightness or object count.

GPU, ~8 min. See tmp_diag/run_linear_readout.sbatch.
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
L.N_IMG = 50                                   # more frames: a 256-dim fit needs N >> 256
import models.dino.deformable_transformer as DT
from diagnostics.objectness_probe import layout_for, H_IMG
from diagnostics.prominence_probe import build_model_from_ckpt, OUT

SEPS = [4, 6, 8, 12, 16, 24]


def ridge_fit(X, y, lam):
    """Closed form: w = (X'X + lam I)^-1 X'y, float64, with an intercept column."""
    X = torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64)
    A[-1, -1] -= lam                                   # do not penalise the intercept
    return torch.linalg.solve(A, X.T @ y)


def ridge_pred(X, w):
    return torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1) @ w


def evaluate(feat, targ, frame, seps, tag):
    """Frame-level split, lambda chosen on a validation slice, report median |err| per separation."""
    X = torch.tensor(np.asarray(feat), dtype=torch.float64)
    y = torch.tensor(np.asarray(targ), dtype=torch.float64)
    fr = np.asarray(frame)
    sp = np.asarray(seps)
    uf = np.unique(fr)
    rng = np.random.RandomState(0); rng.shuffle(uf)
    n_tr = int(0.6 * len(uf)); n_va = int(0.2 * len(uf))
    tr = np.isin(fr, uf[:n_tr]); va = np.isin(fr, uf[n_tr:n_tr + n_va]); te = np.isin(fr, uf[n_tr + n_va:])
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Xn = (X - mu) / sd
    best, bestv = None, None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        w = ridge_fit(Xn[tr], y[tr], lam)
        e = (ridge_pred(Xn[va], w) - y[va]).abs().median().item()
        if bestv is None or e < bestv:
            best, bestv, blam = w, e, lam
    ysh = y[tr][torch.randperm(int(tr.sum()))]
    wsh = ridge_fit(Xn[tr], ysh, blam)
    out = {}
    for s in SEPS:
        m = te & (sp == s)
        if m.sum() < 20:
            continue
        out[s] = dict(
            n=int(m.sum()),
            ridge=float((ridge_pred(Xn[m], best) - y[m]).abs().median()),
            shuffled=float((ridge_pred(Xn[m], wsh) - y[m]).abs().median()),
            spread=float(y[m].abs().median()))
    print(f"    [{tag}] lambda={blam:g}  train frames={n_tr}  test samples={int(te.sum())}", flush=True)
    return out


def probe(name, ckpt, cfg_file, data, dev):
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    rec = {}
    _orig = DT.gen_encoder_output_proposals

    def _gen(*args, **kw):
        om, op = _orig(*args, **kw)
        rec['anchor'] = op.detach()
        return om, op
    DT.gen_encoder_output_proposals = _gen

    def hk(mod, inp, outp):
        if inp[0].shape[1] > 5000:
            rec['mem'] = inp[0].detach()        # EXACTLY what the box head reads
            rec['delta'] = outp.detach()
    h1 = model.transformer.enc_out_bbox_embed.register_forward_hook(hk)

    def pre(mod, args, kwargs=None):
        src = args[0] if args else None
        if src is not None and src.dim() == 3 and src.shape[1] > 5000:
            rec['pre'] = src.detach()           # projected pyramid, BEFORE the encoder
    h2 = model.transformer.encoder.register_forward_pre_hook(pre)

    F, Fp, Y, FR, SP, head_err = [], [], [], [], [], {s: [] for s in SEPS}
    fid = 0
    lay = {}
    with torch.no_grad():
        for sep in SEPS:
            for img, gt in data[sep]:
                rec.clear(); fid += 1
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                model(t.repeat(1, a.num_channels, 1, 1))
                if 'mem' not in rec or 'anchor' not in rec:
                    continue
                mem = rec['mem'][0]
                pre_f = rec.get('pre')
                pre_f = pre_f[0] if pre_f is not None else None
                box_chi = ((rec['delta'][0] + rec['anchor'][0]).sigmoid()[:, 1] * H_IMG).cpu().numpy()
                if not lay:
                    st, sh, of = layout_for(mem.shape[0]); lay.update(st=st, sh=sh, of=of)
                    print(f"  finest stride {st[0]}, {mem.shape[0]} tokens, dim {mem.shape[1]}", flush=True)
                st, sh, of = lay['st'], lay['sh'], lay['of']
                s_, (hh, ww) = st[0], sh[0]
                for i in range(0, len(gt) - 1, 2):
                    gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                    pc = sorted(float((gt[j, 1] + gt[j, 3]) / 2) for j in (i, i + 1))
                    col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                    for peak in pc:
                        r = int(np.clip(round(peak / s_ - 0.5), 0, hh - 1))
                        k = of[0] + r * ww + col
                        tok_chi = (r + 0.5) * s_
                        F.append(mem[k].cpu().numpy())
                        if pre_f is not None:
                            Fp.append(pre_f[k].cpu().numpy())
                        Y.append(peak - tok_chi)                 # signed offset, the head's job
                        FR.append(fid); SP.append(sep)
                        head_err[sep].append(abs(float(box_chi[k]) - peak))
    h1.remove(); h2.remove()
    DT.gen_encoder_output_proposals = _orig
    del model
    torch.cuda.empty_cache()

    print(f"  samples: {len(F)}   (pre-encoder captured: {len(Fp)})", flush=True)
    res = dict(head={s: float(np.median(v)) for s, v in head_err.items() if v},
               enc=evaluate(F, Y, FR, SP, 'encoder output'))
    if len(Fp) == len(F):
        res['pre'] = evaluate(Fp, Y, FR, SP, 'pre-encoder')
    return res


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  frames/rung={L.N_IMG}", flush=True)
    L.SEPS = SEPS
    data = L.make_images(dev)
    summary = {}
    for name, ckpt, cfg_file in L.MODELS:
        print(f"\n########## {name} ##########", flush=True)
        summary[name] = probe(name, ckpt, cfg_file, data, dev)

    print("\n" + "=" * 100)
    print("  Median |error| in px when predicting a token's offset to ITS OWN peak")
    print("  trained head  vs  a straight-line fit on the very same feature vector")
    for nm in summary:
        r = summary[nm]
        print(f"\n  --- {nm}")
        print(f"  {'sep':>5s} {'TRAINED HEAD':>13s} {'ridge (enc out)':>16s} "
              f"{'ridge (pre-enc)':>16s} {'shuffled ctrl':>14s} {'target spread':>14s}")
        for s in SEPS:
            e = r['enc'].get(s)
            if not e:
                continue
            p = r.get('pre', {}).get(s, {})
            print(f"  {s:5d} {r['head'].get(s, float('nan')):13.2f} {e['ridge']:16.2f} "
                  f"{p.get('ridge', float('nan')):16.2f} {e['shuffled']:14.2f} {e['spread']:14.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for nm, c in zip(summary, ('tab:blue', 'tab:red', 'tab:green')):
        r = summary[nm]
        ss = [s for s in SEPS if s in r['enc']]
        ax.plot(ss, [r['head'][s] for s in ss], 'o-', color=c, label=f'{nm}: trained head')
        ax.plot(ss, [r['enc'][s]['ridge'] for s in ss], 's--', color=c, alpha=.6,
                label=f'{nm}: straight-line fit')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('median |error| predicting the offset to its own peak (px)')
    ax.set_title('Phase Y: is the offset in the feature, or missing?')
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linear_readout.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}")
    json.dump(summary, open(os.path.join(OUT, 'linear_readout_probe.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
