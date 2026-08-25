"""Phase Z.7 — is the ladder's ROUND peak shape doing part of the work? (the Z.2 loose end)

Every arm that WORKS in this sequence was trained and tested on ISO=1 stimuli: box 2.43 x 8.5, which
renders sigma_q = sigma_chi = 2.43 px, a round peak. Every arm that FAILS was trained on real frames,
boxes ~10.6 x 8.5, sigma_q = 10.6 -- a peak 4.4x wider in q.

An important correction to how this was first framed. Along CHI -- the axis the separation is measured
on -- the two are IDENTICAL: sigma_chi = box_height / a_coef = 8.5 / 3.5 = 2.43 px in BOTH cases,
because box_height is 8.5 either way. So it is NOT that round peaks are easier to tell apart in chi;
the chi profile is the same and two peaks 8 px apart are 3.3 sigma apart in both. What differs is the
q extent, the total flux, and the box WIDTH the head must regress (2.43 vs 10.6 normalised).

What is genuinely unexplained is one number from Z.2: the ladder-trained head scores **0.31** on ISO=1
stimuli and **2.55** on ISO=0 stimuli, against the real-trained head's 3.67. That was written off as
the ladder head being out of its training distribution -- it learned to emit 2.43-wide boxes and was
tested where 10.6 is correct -- which is plausible and was never tested. 2.55 is large enough that it
should not sit unexplained underneath five phases of conclusions.

The test: train on ladder frames built at ISO=0 and score at ISO=0, so training and test agree.

  ~0.3  -> the 2.55 was purely the train/test box-width mismatch. The stimulus is innocent, and
           Z.2-Z.6 stand as written.
  ~2.5  -> something about the anisotropic stimulus is genuinely harder for a reason not yet
           identified, and Z.2's framing needs restating -- as would phase Y's "the information is
           there", which was also measured on ISO=1 only.

Three training sets x two eval sets, all on the same planted peak POSITIONS (make_images seeds per
frame, so ISO=1 and ISO=0 differ only in box size):

    trained on ladder ISO=1  /  ladder ISO=0  /  real frames
    scored on  ladder ISO=1 test  and  ladder ISO=0 test

PROTOCOL NOTE: each arm's lr and epoch are selected on the VALIDATION slice of the eval set being
reported, tracked simultaneously during one training run. That gives every arm its best shot on each
domain rather than penalising an arm for being selected out of domain, which is the fairest way to
read a cross-domain table.

GPU, ~25 min. See tmp_diag/run_isoshape.sbatch.
"""
import os, sys, json, time

os.environ.setdefault('LADDER_AXIS', 'chi')
os.environ.setdefault('LADDER_ISO', '1')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import diagnostics.separation_ladder as L
L.N_IMG = 50
from models.dino.matcher import HungarianMatcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, OUT
from diagnostics.head_only_probe import (new_head, box_loss, eval_boxbased, per_sep, split_frames,
                                         SEPS, LRS, EPOCHS, BATCH, C_CLASS, C_BBOX, C_GIOU)
from diagnostics.real_frame_head_probe import cache, make_ladder_frames, N_REAL
from diagnostics.gap_distribution_probe import make_real


def train_dual(tr, vals, dev, matcher, tag):
    """One training run, but the best checkpoint is tracked SEPARATELY for each validation slice."""
    best = {k: None for k in vals}
    for lr in LRS:
        torch.manual_seed(0)
        head = new_head(4, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        order = np.arange(len(tr)); rng = np.random.RandomState(1)
        for ep in range(EPOCHS):
            rng.shuffle(order); head.train()
            for b0 in range(0, len(order), BATCH):
                idx = order[b0:b0 + BATCH]
                mem = torch.stack([tr[i]['mem'] for i in idx]).to(dev).float()
                anc = torch.stack([tr[i]['anc'] for i in idx]).to(dev)
                cls = torch.stack([tr[i]['cls'] for i in idx]).to(dev)
                pred = (head(mem) + anc).sigmoid()
                tg = [dict(boxes=tr[i]['boxes'].to(dev), labels=tr[i]['labels'].to(dev))
                      for i in idx]
                ind = matcher(dict(pred_logits=cls, pred_boxes=pred), tg)
                src = torch.cat([pred[b][si.to(dev)] for b, (si, _) in enumerate(ind)])
                dst = torch.cat([tg[b]['boxes'][ti.to(dev)] for b, (_, ti) in enumerate(ind)])
                if len(dst) == 0:
                    continue
                loss = box_loss(src, dst)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
                head.eval()
                snap = None
                for k, va in vals.items():
                    v = float(np.median(eval_boxbased(head, va, dev)[0]))
                    if best[k] is None or v < best[k][0]:
                        if snap is None:
                            snap = {n: t.detach().clone() for n, t in head.state_dict().items()}
                        best[k] = (v, lr, ep + 1, snap)
    out = {}
    for k in vals:
        h = new_head(4, dev); h.load_state_dict(best[k][3]); h.eval()
        out[k] = h
        print(f"    [{tag}] val[{k}] {best[k][0]:.2f} px (lr={best[k][1]:g}, ep={best[k][2]})",
              flush=True)
    return out


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    print(f"device={dev}", flush=True)

    print("\n--- building frames", flush=True)
    lad1 = make_ladder_frames(dev, iso=True)
    lad0 = make_ladder_frames(dev, iso=False)
    real, _ = make_real(dev, N_REAL, (1.0, 6.0))
    print(f"  built in {time.time() - t0:.0f}s", flush=True)

    print("\n--- caching", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    c1 = cache(model, a, lad1, dev, ladder=True)
    c0 = cache(model, a, lad0, dev, ladder=True)
    cr = cache(model, a, real, dev, ladder=False)
    del model; torch.cuda.empty_cache()
    tr1, va1, te1 = split_frames(c1)
    tr0, va0, te0 = split_frames(c0)
    vals = {'ISO1': va1, 'ISO0': va0}
    tests = {'ISO1': te1, 'ISO0': te0}
    print(f"  ISO1 {len(c1)} -> {len(tr1)}/{len(va1)}/{len(te1)};  "
          f"ISO0 {len(c0)} -> {len(tr0)}/{len(va0)}/{len(te0)};  real {len(cr)}", flush=True)

    matcher = HungarianMatcher(cost_class=C_CLASS, cost_bbox=C_BBOX, cost_giou=C_GIOU,
                               focal_alpha=0.25).to(dev)
    ARMS = [('ladder ISO=1', tr1), ('ladder ISO=0', tr0), ('real frames', cr[:N_REAL])]
    res = {}
    for tag, tr in ARMS:
        print(f"\n--- arm: trained on {tag}  ({len(tr)} frames)", flush=True)
        heads = train_dual(tr, vals, dev, matcher, tag)
        res[tag] = {k: per_sep(*eval_boxbased(heads[k], tests[k], dev)) for k in tests}

    print("\n" + "=" * 100)
    print("  Median |chi-centre error| px.  Z.2 recorded: ladder ISO1-trained 0.31 on ISO1 / 2.55 on")
    print("  ISO0;  real-trained 3.22 on ISO1 / 3.67 on ISO0.")
    for k in ('ISO1', 'ISO0'):
        print(f"\n  scored on ladder {k} test")
        print("  " + f"{'trained on':<16s}" + "".join(f"{('sep ' + str(s)):>9s}" for s in SEPS))
        for tag, _ in ARMS:
            ps = res[tag][k]
            row = "".join((f"{ps[s]['chi']:9.2f}" if s in ps else f"{'-':>9s}") for s in SEPS)
            print(f"  {tag:<16s}{row}")

    json.dump(res, open(os.path.join(OUT, 'iso_shape_probe.json'), 'w'), indent=2, default=str)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, k in zip(axs, ('ISO1', 'ISO0')):
        for tag, _ in ARMS:
            ps = res[tag][k]; ss = [s for s in SEPS if s in ps]
            ax.plot(ss, [ps[s]['chi'] for s in ss], 'o-', label=f'trained on {tag}')
        ax.set_title(f'scored on {k} stimuli')
        ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
        ax.set_xlabel('planted χ-separation (px)'); ax.set_ylabel('median |χ error| (px)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iso_shape.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}\nPROBE DONE  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
