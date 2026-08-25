"""Phase Z.4 + Z.5 — is close-pair resolution a skill that must DOMINATE the training signal?

Z.2 showed training this head on real frames reproduces the merge (3.22 px at an 8 px gap) where
ladder frames give 0.31. Z.3 then neutralised four properties of the real distribution one at a time
— brightness variation, rings, object count, crowding — and none of them moved it (3.20 / 3.35 / 3.21
/ 3.35 against a 3.22 baseline), with `fixed box size` acting as an unintended positive control at
6.12 that proves the machinery bites. So the property list was wrong.

What Z.3 forces is a reframing. EVERY ladder frame is nothing but planted pairs: each object has a
same-q partner within 24 px, where real frames have a median same-q gap of 43 px. And phase Z's diet
sweep never varied that — it varied WHICH separations appeared (0.50 -> 0.06 of frames from the close
rungs), never whether an object had a partner at all, because the "far" rungs are pairs too. Note
also that on the usual statistic the two distributions ALREADY match: the ladder's `<5 px` fraction is
about 1/6 (only the sep-4 rung) and real frames measure 0.173. Rarity was never the axis.

Two experiments, sharing one feature cache. Anchors throughout: **real 3.22**, **ladder 0.31**,
deployed head 3.83. All arms are scored on the same unmodified ladder test slice as Z/Z.2/Z.3.

--------------------------------------------------------------------------------------------------
Z.4 — PAIR DILUTION (finds the cause). Ladder frames with real, unpaired peaks added AROUND the
planted pairs, sweeping how much of the frame is pair versus context: ~16 / 32 / 60 / 120 objects per
frame against a constant 16 planted ones.

The planted pairs are byte-identical across every arm — same seeds, same positions, same count, same
intensity — so the close-pair supervision is CONSTANT in absolute terms and only the context grows.
That is the distinction phase Z's diet sweep could not make: "pairs diluted" vs "fewer pairs".

  error climbs 0.31 -> ~3.2 as context grows  -> CONFIRMED. Close-pair resolution is only learned when
     it dominates the training signal, and the lever is loss-side, not data-side.
  error stays ~0.31 at real-like composition  -> REFUTED, and it narrows things again: that arm is
     close to real frames in makeup, so the difference would have to be in the NATURE of the pairs —
     real siblings come from the cluster generator with varied brightness and width, sitting on rings
     at correlated q, where ladder pairs are identical twins planted in isolation.

The paired fraction is MEASURED per arm (share of objects with a same-q neighbour within 24 px) and
printed alongside real frames' value, rather than assumed from the object counts.

--------------------------------------------------------------------------------------------------
Z.5 — CLOSE-PAIR LOSS WEIGHT (finds a fix, and does not depend on Z.4 landing). Train on real frames,
but up-weight the box loss on ground-truth peaks that have a close same-q neighbour. Weight 1 (the
Z.2 baseline), 3, 10, 30.

The weighted loss is normalised by the SUM OF WEIGHTS, not by the box count. That matters: dividing
by count would make w=30 a 30x larger loss, i.e. just a learning-rate change that the lr sweep would
absorb and report as nothing. Normalising by weight sum makes it pure RE-ALLOCATION of gradient
toward close pairs, which is the actual lever under test.

  recovers toward 0.31 -> a candidate lever regardless of whether Z.4 explains the mechanism, and the
     first thing in this whole sequence worth spending a full joint training run on.
  flat -> re-weighting is not enough, and the fix has to be structural.

Everything here is still a frozen trunk with only the box terms active, so any lever found must still
survive full joint training against the organic gate. That run is the endgame, not this.

GPU, ~60 min. See tmp_diag/run_pairfocus.sbatch.
"""
import os, sys, json, random, time

os.environ.setdefault('LADDER_AXIS', 'chi')
os.environ.setdefault('LADDER_ISO', '1')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import simulation as S
from simulation import FastSimulation
import diagnostics.separation_ladder as L
L.N_IMG = 50
from models.dino.matcher import HungarianMatcher
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, OUT
from diagnostics.head_only_probe import (new_head, eval_boxbased, per_sep, split_frames, SEPS,
                                         LRS, EPOCHS, BATCH, W_L1, W_GIOU,
                                         C_CLASS, C_BBOX, C_GIOU)
from diagnostics.real_frame_head_probe import cache, make_ladder_frames, sibling_gap, N_REAL

SIGMA = 8.5 / S.SimulationConfig.a_coef
LADDER_INT = 30.0
PAIR_PX = 10.0          # "has a close same-q neighbour" -> gets the Z.5 up-weight
P24_PX = 24.0           # the ladder's widest planted separation: the honest "is it paired" threshold
CLEAR_Q, CLEAR_CHI = 8.0, 40.0    # context peaks this close to a planted one are dropped
TOTALS = [16, 32, 60, 120]        # objects per frame; 16 planted throughout
WEIGHTS = [1.0, 3.0, 10.0, 30.0]
if os.environ.get('HEADPROBE_SMOKE') == '1':
    TOTALS, WEIGHTS = [16, 60], [1.0, 10.0]


def frac_paired(b, within=P24_PX, qtol=8.0):
    """Share of objects with a same-q neighbour within `within` px. The ladder is ~1.0 by
    construction; real frames are not. This is the axis Z.4 sweeps, so it is measured, not assumed."""
    g = sibling_gap(b)
    return float((g < within).mean()) if len(g) else 0.0


def make_diluted(dev, n_total):
    """Ladder frames plus real unpaired context. Planted pairs identical across arms by construction:
    the same per-frame seeds and the same RandomState stream the ladder itself uses."""
    L.BOX_W, L.BOX_H = SIGMA * S.SimulationConfig.w_coef, SIGMA * S.SimulationConfig.a_coef
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = True
    _orig = FastSimulation.simulate_labels
    state = {}

    def patched(_self=sim):
        gt = state['gt']                                     # the planted pairs, fixed per frame
        n_ctx = max(0, n_total - len(gt))
        cb = np.zeros((0, 4), np.float32); ci = np.zeros(0, np.float32); cr = np.zeros(0, bool)
        if n_ctx:
            rb, ri, rr = _orig(_self)
            b = rb.detach().cpu().numpy()
            it = ri.detach().cpu().numpy()
            r = rr.detach().cpu().numpy().astype(bool)
            if len(b):
                q = (b[:, 0] + b[:, 2]) / 2; c = (b[:, 1] + b[:, 3]) / 2
                pq = (gt[:, 0] + gt[:, 2]) / 2; pc = (gt[:, 1] + gt[:, 3]) / 2
                near = (np.abs(q[:, None] - pq[None]) < CLEAR_Q) & \
                       (np.abs(c[:, None] - pc[None]) < CLEAR_CHI)
                keep = ~near.any(1)                          # never disturb a planted pair
                b, it, r = b[keep], it[keep], r[keep]
            if len(b) > n_ctx:
                idx = np.random.RandomState(len(b)).choice(len(b), n_ctx, replace=False)
                b, it, r = b[idx], it[idx], r[idx]
            cb, ci, cr = b, it, r
        allb = np.concatenate([gt, cb]) if len(cb) else gt
        alli = np.concatenate([np.full(len(gt), LADDER_INT, np.float32), ci]) if len(ci) \
            else np.full(len(gt), LADDER_INT, np.float32)
        allr = np.concatenate([np.zeros(len(gt), bool), cr]) if len(cr) else np.zeros(len(gt), bool)
        return (torch.tensor(allb, dtype=torch.float32, device=_self.device),
                torch.tensor(alli, dtype=torch.float32, device=_self.device),
                torch.tensor(allr, device=_self.device).bool())
    sim.simulate_labels = patched

    frames, ps, nobj = [], [], []
    for sep in SEPS:
        rng = np.random.RandomState(1234 + sep)              # the ladder's own stream
        for k in range(L.N_IMG):
            _sd = 9000 + sep * 100 + k
            random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
            state['gt'] = L.plant(sep, rng)
            try:
                img, bx, mask, isr = sim.simulate_img()
            except Exception:
                continue
            b = bx.detach().cpu().numpy()
            if len(b) == 0:
                continue
            frames.append((img.detach().cpu(), b, isr.detach().cpu().numpy().astype(np.int64), sep))
            ps.append(frac_paired(b)); nobj.append(len(b))
    st = dict(obj_per_frame=float(np.mean(nobj)), frac_p24=float(np.mean(ps)), frames=len(frames))
    print(f"  [dilute n={n_total}] {st['frames']} frames, {st['obj_per_frame']:.1f} obj/frame, "
          f"paired-within-24px {st['frac_p24']:.3f}", flush=True)
    return frames, st


def make_real(dev, n):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = True
    frames, ps, nobj = [], [], []
    k = 0
    while len(frames) < n and k < n * 4:
        _sd = 4242 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        k += 1
        try:
            img, bx, mask, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy()
        if len(b) == 0:
            continue
        frames.append((img.detach().cpu(), b, isr.detach().cpu().numpy().astype(np.int64)))
        ps.append(frac_paired(b)); nobj.append(len(b))
    st = dict(obj_per_frame=float(np.mean(nobj)), frac_p24=float(np.mean(ps)), frames=len(frames))
    print(f"  [real] {st['frames']} frames, {st['obj_per_frame']:.1f} obj/frame, "
          f"paired-within-24px {st['frac_p24']:.3f}", flush=True)
    return frames, st


def add_weights(frames, w_close):
    """Per-GT weight for Z.5: w_close on any peak with a close same-q neighbour, 1 elsewhere."""
    for f in frames:
        b = f['boxes'].numpy().copy()
        xy = np.stack([(b[:, 0] - b[:, 2] / 2) * 1024, (b[:, 1] - b[:, 3] / 2) * 512,
                       (b[:, 0] + b[:, 2] / 2) * 1024, (b[:, 1] + b[:, 3] / 2) * 512], 1)
        g = sibling_gap(xy)
        f['wgt'] = torch.tensor(np.where(g < PAIR_PX, w_close, 1.0), dtype=torch.float32)
    return frames


def wbox_loss(pred, tgt, w):
    """L1 + GIoU, normalised by the SUM OF WEIGHTS.

    Dividing by the box count instead would make w=30 simply a 30x bigger loss — a learning-rate
    change the lr sweep absorbs, reported as a null. Normalising by weight sum keeps the loss scale
    fixed and RE-ALLOCATES gradient toward close pairs, which is the lever actually under test."""
    l1 = (torch.nn.functional.l1_loss(pred, tgt, reduction='none').sum(-1) * w).sum()
    giou = ((1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred),
                                                box_cxcywh_to_xyxy(tgt)))) * w).sum()
    return (W_L1 * l1 + W_GIOU * giou) / w.sum().clamp_min(1e-6)


def train(tr, va, dev, matcher, tag):
    """Phase Z's arm A, with per-target weights. lr AND epoch chosen on `va` only."""
    best = None
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
                w = torch.cat([tr[idx[b]]['wgt'].to(dev)[ti.to(dev)]
                               for b, (_, ti) in enumerate(ind)])
                if len(dst) == 0:
                    continue
                loss = wbox_loss(src, dst, w)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
                head.eval()
                v = float(np.median(eval_boxbased(head, va, dev)[0]))
                if best is None or v < best[0]:
                    best = (v, lr, ep + 1,
                            {k: t.detach().clone() for k, t in head.state_dict().items()})
        print(f"    [{tag}] lr={lr:g} -> best val {best[0]:.2f} px "
              f"(lr={best[1]:g}, ep={best[2]})", flush=True)
    head = new_head(4, dev); head.load_state_dict(best[3]); head.eval()
    return head, best[1], best[2]


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    print(f"device={dev}  totals={TOTALS}  weights={WEIGHTS}", flush=True)

    print("\n--- building frames", flush=True)
    lad = make_ladder_frames(dev, iso=True)
    dil = {n: make_diluted(dev, n) for n in TOTALS}
    real, st_real = make_real(dev, N_REAL)
    print(f"  built in {time.time() - t0:.0f}s", flush=True)

    print("\n--- caching through the frozen trunk", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    c_lad = cache(model, a, lad, dev, ladder=True)
    c_dil = {n: cache(model, a, dil[n][0], dev, ladder=False) for n in TOTALS}
    c_real = cache(model, a, real, dev, ladder=False)
    del model; torch.cuda.empty_cache()
    _, va, te = split_frames(c_lad)
    print(f"  ladder {len(c_lad)} (val {len(va)} / test {len(te)});  "
          + "  ".join(f"n={n}:{len(c_dil[n])}" for n in TOTALS) + f";  real {len(c_real)}", flush=True)

    matcher = HungarianMatcher(cost_class=C_CLASS, cost_bbox=C_BBOX, cost_giou=C_GIOU,
                               focal_alpha=0.25).to(dev)
    z4, z5 = {}, {}
    for n in TOTALS:
        tr = add_weights(c_dil[n][:N_REAL], 1.0)
        print(f"\n--- Z.4 dilution, {n} obj/frame  ({len(tr)} frames)", flush=True)
        head, lr, ep = train(tr, va, dev, matcher, f'dilute{n}')
        z4[n] = dict(per_sep=per_sep(*eval_boxbased(head, te, dev)), lr=lr, epoch=ep,
                     stats=dil[n][1])
    for w in WEIGHTS:
        tr = add_weights(c_real[:N_REAL], w)
        print(f"\n--- Z.5 real frames, close-pair weight {w:g}  ({len(tr)} frames)", flush=True)
        head, lr, ep = train(tr, va, dev, matcher, f'w{w:g}')
        z5[w] = dict(per_sep=per_sep(*eval_boxbased(head, te, dev)), lr=lr, epoch=ep)

    print("\n" + "=" * 100)
    print("  Median |chi-centre error| px on the ladder test slice")
    print("  anchors: real 3.22 (Z.2)   ladder 0.31 (Z)   deployed head 3.83")
    print(f"\n  Z.4 — pair dilution (16 planted objects held constant; real frames "
          f"paired-24px {st_real['frac_p24']:.3f})")
    print("  " + f"{'obj/frame':<12s}{'paired24':>10s}" +
          "".join(f"{('sep ' + str(s)):>9s}" for s in SEPS))
    for n in TOTALS:
        ps, stt = z4[n]['per_sep'], z4[n]['stats']
        row = "".join((f"{ps[s]['chi']:9.2f}" if s in ps else f"{'-':>9s}") for s in SEPS)
        print(f"  {stt['obj_per_frame']:<12.1f}{stt['frac_p24']:10.3f}{row}")
    print(f"\n  Z.5 — close-pair loss weight on REAL frames")
    print("  " + f"{'weight':<12s}" + "".join(f"{('sep ' + str(s)):>9s}" for s in SEPS))
    for w in WEIGHTS:
        ps = z5[w]['per_sep']
        row = "".join((f"{ps[s]['chi']:9.2f}" if s in ps else f"{'-':>9s}") for s in SEPS)
        print(f"  {w:<12g}{row}")

    json.dump(dict(z4={str(k): v for k, v in z4.items()},
                   z5={str(k): v for k, v in z5.items()}, real=st_real),
              open(os.path.join(OUT, 'pair_focus_probe.json'), 'w'), indent=2, default=str)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in TOTALS:
        ps = z4[n]['per_sep']; ss = [s for s in SEPS if s in ps]
        a1.plot(ss, [ps[s]['chi'] for s in ss], 'o-',
                label=f"{z4[n]['stats']['obj_per_frame']:.0f} obj/fr "
                      f"(paired {z4[n]['stats']['frac_p24']:.2f})")
    a1.set_title('Z.4: pairs held constant, context added')
    for w in WEIGHTS:
        ps = z5[w]['per_sep']; ss = [s for s in SEPS if s in ps]
        a2.plot(ss, [ps[s]['chi'] for s in ss], 's-', label=f'close-pair weight {w:g}')
    a2.set_title('Z.5: re-weighting close pairs on real frames')
    for ax in (a1, a2):
        ax.axhline(3.83, ls=':', c='k', lw=1, label='deployed head')
        ax.axhline(0.31, ls='--', c='g', lw=1, label='ladder-trained')
        ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
        ax.set_xlabel('planted χ-separation (px)'); ax.set_ylabel('median |χ error| (px)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pair_focus.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}\nPROBE DONE  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
