"""Phase Z.2 (experiment 1) — train the head on REAL frames instead of ladder frames.

Phase Z trained `enc_out_bbox_embed` alone, on a frozen trunk, with the real loss, and it reached
0.31 px at an 8 px gap where the same head in the real run manages 3.83. Three differences from the
real run survived: the trunk was frozen (and frozen at convergence), every competing loss term was
absent, and the FRAMES were the ladder's simplified ones. Z.1 then showed the head was already stuck
at epoch 279 with the information already fully available, so it is not slowly converging — but two
checkpoints cannot separate the remaining explanations.

This fills the missing cell of a 2x2:

                     ladder frames        real frames
    frozen trunk     Z: 0.31 px  OK       <- THIS
    moving trunk     (a full run)         real run: 3.83 px  FAIL

Everything is held at phase Z's settings except the SOURCE OF THE TRAINING FRAMES. Evaluation is on
the very same ladder test slice phase Z used (same seed, same 60/20/20 frame split), so the number
lands directly against 0.31 and 3.83.

  ~3.8  -> the real training DISTRIBUTION is what teaches merging. The trajectory is innocent, and
           the lever is the data, not joint optimisation.
  ~0.3  -> the distribution is cleared too, and what remains is the trajectory: the head can learn
           this from real data and simply never does while the trunk moves underneath it.

THE CONFOUND THIS MUST AVOID. The base simulator places peaks independently at random and produces
`<5 px` chi-gaps at only 0.029 — BELOW the 0.06 floor of phase Z's diet sweep, where the isolated head
still managed 0.57 px. Run this with clusters OFF alone and a failure would just be the rarity result
again, wearing a distribution costume. So the PRIMARY arm runs with `use_peak_clusters = True`
(measured `<5 px` = 0.177, phase U's rate, ABOVE everything the diet sweep tested); clusters OFF is
carried only as a secondary contrast. The realised gap fraction of each training set is measured here
and printed, so both arms can be placed on the diet sweep's f-axis rather than assumed onto it.

TWO CONTROLS.
  * Box shape. The ladder at ISO=1 plants 2.43 x 8.5 px boxes; real frames carry ~10.6 x 8.5. A head
    trained on real frames and tested on isotropic stimuli could fail for that reason alone, so a
    SECOND ladder eval set is built at ISO=0 (real box shape, identical peak positions — `make_images`
    seeds per frame, so only the box size differs). A failure has to show up on both to count.
  * A ladder-trained arm is RE-RUN here rather than quoted from phase Z, so the baseline and the real
    arms are scored on byte-identical eval sets.

Training-set size is held at 180 frames for every arm, matching phase Z. Note the real frames carry
far more objects each, so the real arms get MORE box supervision, not less — which biases toward the
real arm succeeding, and makes a failure the cleaner of the two possible results.

GPU, ~35 min. See tmp_diag/run_realframe.sbatch.
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
L.N_IMG = 50                                    # phase Z's ladder frame count -> same 60/20/20 split
import models.dino.deformable_transformer as DT
from models.dino.matcher import HungarianMatcher
from diagnostics.objectness_probe import layout_for, H_IMG, W_IMG
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, OUT
from diagnostics.head_only_probe import (new_head, box_loss, eval_boxbased, per_sep, split_frames,
                                         NQ, ANC_CLAMP, LRS, EPOCHS, BATCH, SEPS,
                                         C_CLASS, C_BBOX, C_GIOU)

N_REAL = 180            # == phase Z's train slice, so training-set SIZE is not a variable
QTOL = 8.0              # same-q tolerance used by clusters_gate / verify_clusters
SIGMA = 8.5 / S.SimulationConfig.a_coef
if os.environ.get('HEADPROBE_SMOKE') == '1':      # wiring check only -- numbers are meaningless
    N_REAL = 12                                   # LRS/EPOCHS already come down via head_only_probe


def sibling_gap(bx):
    """chi-distance from each peak to its nearest neighbour at the SAME q (within QTOL px).

    This is the statistic the diet sweep's f-axis approximates and that MODIFICATIONS.md quotes as
    12.5% under 5 px for real organic labels, so measuring it here puts each training set on that
    same axis instead of assuming where it sits."""
    if len(bx) == 0:
        return np.zeros(0)
    q = (bx[:, 0] + bx[:, 2]) / 2
    c = (bx[:, 1] + bx[:, 3]) / 2
    out = np.full(len(bx), np.inf)
    for i in range(len(bx)):
        m = np.abs(q - q[i]) < QTOL
        m[i] = False
        if m.any():
            out[i] = np.min(np.abs(c[m] - c[i]))
    return out


def make_real_frames(dev, clusters, n):
    """Frames from the REAL simulator, the same call the training pipeline makes."""
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    frames, gaps, nobj = [], [], []
    k = 0
    while len(frames) < n and k < n * 3:
        _sd = 4242 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        k += 1
        try:
            img, bx, mask, isr = sim.simulate_img()
        except Exception as e:
            print(f"    real frame {k} failed: {type(e).__name__}: {e}", flush=True)
            continue
        b = bx.detach().cpu().numpy()
        if len(b) == 0:
            continue
        frames.append((img.detach().cpu(), b, isr.detach().cpu().numpy().astype(np.int64)))
        gaps.append(sibling_gap(b))
        nobj.append(len(b))
    g = np.concatenate(gaps) if gaps else np.zeros(0)
    finite = g[np.isfinite(g)]
    stat = dict(frames=len(frames), obj_per_frame=float(np.mean(nobj)) if nobj else 0.0,
                frac_lt5=float((finite < 5).mean()) if len(finite) else 0.0,
                frac_lt10=float((finite < 10).mean()) if len(finite) else 0.0,
                median_gap=float(np.median(finite)) if len(finite) else float('nan'))
    print(f"  real frames (clusters={'ON' if clusters else 'OFF'}): {stat['frames']} frames, "
          f"{stat['obj_per_frame']:.1f} obj/frame, <5px {stat['frac_lt5']:.3f}, "
          f"<10px {stat['frac_lt10']:.3f}, median gap {stat['median_gap']:.1f} px", flush=True)
    return frames, stat


def make_ladder_frames(dev, iso):
    """The phase-V/Y/Z ladder. `make_images` seeds per frame, so ISO=1 and ISO=0 differ ONLY in box
    size — same peak positions, which is what makes the box-shape control clean."""
    if iso:
        L.BOX_W, L.BOX_H = SIGMA * S.SimulationConfig.w_coef, SIGMA * S.SimulationConfig.a_coef
    else:
        L.BOX_W, L.BOX_H = 10.6, 8.5
    L.SEPS = SEPS
    data = L.make_images(dev)
    out = []
    for sep in SEPS:
        for img, gt in data[sep]:
            out.append((img, np.asarray(gt), np.zeros(len(gt), dtype=np.int64), sep))
    return out


def cache(model, a, frames, dev, ladder):
    """Record exactly what enc_out_bbox_embed is fed, for each frame. One frozen forward pass each.

    `ladder=True` additionally records the two evaluation tokens per planted pair (the phase-Y
    measurement points); real frames need none, since nothing is evaluated on them."""
    rec = {}
    _orig = DT.gen_encoder_output_proposals

    def _gen(*args, **kw):
        om, op = _orig(*args, **kw)
        rec['anchor'] = op.detach()
        return om, op
    DT.gen_encoder_output_proposals = _gen

    def hk(mod, inp, outp):
        if inp[0].shape[1] > 5000:
            rec['mem'] = inp[0].detach()
    h1 = model.transformer.enc_out_bbox_embed.register_forward_hook(hk)

    def hc(mod, inp, outp):
        if inp[0].shape[1] > 5000:
            rec['cls'] = outp.detach()
    h2 = model.transformer.enc_out_class_embed.register_forward_hook(hc)

    out, lay, fid = [], {}, 0
    with torch.no_grad():
        for item in frames:
            img, gt, lab = item[0], item[1], item[2]
            sep = item[3] if len(item) > 3 else -1
            rec.clear(); fid += 1
            t = img.to(dev)
            if t.dim() == 2:
                t = t[None, None]
            elif t.dim() == 3:
                t = t[None]
            model(t.repeat(1, a.num_channels, 1, 1))
            if not {'mem', 'anchor', 'cls'} <= set(rec):
                continue
            mem, anc, cls = rec['mem'][0], rec['anchor'][0], rec['cls'][0]
            if not lay:
                st, sh, of = layout_for(mem.shape[0]); lay.update(st=st, sh=sh, of=of)
            st, sh, of = lay['st'], lay['sh'], lay['of']
            s_, (hh, ww) = st[0], sh[0]
            topk = torch.topk(cls.max(-1)[0], NQ, dim=0)[1]
            inv = torch.full((mem.shape[0],), -1, dtype=torch.long, device=dev)
            inv[topk] = torch.arange(NQ, device=dev)

            g = torch.tensor(np.asarray(gt), dtype=torch.float32, device=dev)
            boxes = torch.stack([(g[:, 0] + g[:, 2]) / 2 / W_IMG,
                                 (g[:, 1] + g[:, 3]) / 2 / H_IMG,
                                 (g[:, 2] - g[:, 0]).abs() / W_IMG,
                                 (g[:, 3] - g[:, 1]).abs() / H_IMG], 1)
            if bool((boxes[:, 2] <= 0).any() or (boxes[:, 3] <= 0).any()):
                continue                    # a degenerate box would trip the GIoU assert
            toks = []
            if ladder:
                for i in range(0, len(g) - 1, 2):
                    gq = float((g[i, 0] + g[i, 2] + g[i + 1, 0] + g[i + 1, 2]) / 4)
                    col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                    for j in (i, i + 1):
                        peak = float((g[j, 1] + g[j, 3]) / 2)
                        r = int(np.clip(round(peak / s_ - 0.5), 0, hh - 1))
                        p = int(inv[of[0] + r * ww + col])
                        if p >= 0:
                            toks.append((p, peak, (r + 0.5) * s_, j))
                if not toks:
                    continue
            out.append(dict(mem=mem[topk].half().cpu(),
                            anc=anc[topk].clamp(-ANC_CLAMP, ANC_CLAMP).cpu(),
                            cls=cls[topk].float().cpu(),
                            boxes=boxes.cpu(),
                            labels=torch.tensor(np.asarray(lab), dtype=torch.long),
                            toks=toks, sep=sep, fid=fid))
    h1.remove(); h2.remove()
    DT.gen_encoder_output_proposals = _orig
    return out


def train_A(tr, va, dev, matcher, tag):
    """Phase Z's arm A, unchanged: the real interm_outputs loss, lr AND epoch chosen on `va` only."""
    best = None
    for lr in LRS:
        torch.manual_seed(0)
        head = new_head(4, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        order = np.arange(len(tr))
        rng = np.random.RandomState(1)
        for ep in range(EPOCHS):
            rng.shuffle(order)
            head.train()
            for b0 in range(0, len(order), BATCH):
                idx = order[b0:b0 + BATCH]
                mem = torch.stack([tr[i]['mem'] for i in idx]).to(dev).float()
                anc = torch.stack([tr[i]['anc'] for i in idx]).to(dev)
                cls = torch.stack([tr[i]['cls'] for i in idx]).to(dev)
                pred = (head(mem) + anc).sigmoid()
                tgts = [dict(boxes=tr[i]['boxes'].to(dev), labels=tr[i]['labels'].to(dev))
                        for i in idx]
                ind = matcher(dict(pred_logits=cls, pred_boxes=pred), tgts)
                src = torch.cat([pred[b][si.to(dev)] for b, (si, _) in enumerate(ind)])
                dst = torch.cat([tgts[b]['boxes'][ti.to(dev)] for b, (_, ti) in enumerate(ind)])
                if len(dst) == 0:
                    continue
                loss = box_loss(src, dst)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
                head.eval()
                v = float(np.median(eval_boxbased(head, va, dev)[0]))
                if best is None or v < best[0]:
                    best = (v, lr, ep + 1,
                            {k: t.detach().clone() for k, t in head.state_dict().items()})
        print(f"    [{tag}] lr={lr:g} -> best val {best[0]:.2f} px (lr={best[1]:g}, ep={best[2]})",
              flush=True)
    head = new_head(4, dev); head.load_state_dict(best[3]); head.eval()
    return head, best[1], best[2]


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}", flush=True)
    t0 = time.time()

    print("\n--- building frames", flush=True)
    lad_iso = make_ladder_frames(dev, iso=True)
    lad_raw = make_ladder_frames(dev, iso=False)
    real_on, st_on = make_real_frames(dev, True, N_REAL)
    real_off, st_off = make_real_frames(dev, False, N_REAL)
    print(f"  frames built in {time.time() - t0:.0f}s", flush=True)

    print("\n--- caching through the frozen trunk", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    c_iso = cache(model, a, lad_iso, dev, ladder=True)
    c_raw = cache(model, a, lad_raw, dev, ladder=True)
    c_on = cache(model, a, real_on, dev, ladder=False)
    c_off = cache(model, a, real_off, dev, ladder=False)
    del model; torch.cuda.empty_cache()

    tr_iso, va_iso, te_iso = split_frames(c_iso)        # phase Z's split, same seed
    _, va_raw, te_raw = split_frames(c_raw)
    print(f"  ladder ISO cached {len(c_iso)} -> {len(tr_iso)}/{len(va_iso)}/{len(te_iso)}", flush=True)
    print(f"  ladder RAW cached {len(c_raw)}   real ON {len(c_on)}   real OFF {len(c_off)}",
          flush=True)

    matcher = HungarianMatcher(cost_class=C_CLASS, cost_bbox=C_BBOX, cost_giou=C_GIOU,
                               focal_alpha=0.25).to(dev)
    ARMS = [('real_clusters_ON', c_on[:N_REAL]),
            ('real_clusters_OFF', c_off[:N_REAL]),
            ('ladder (Z baseline)', tr_iso)]
    res = {}
    for tag, tr in ARMS:
        print(f"\n--- arm: trained on {tag}  ({len(tr)} frames)", flush=True)
        head, lr, ep = train_A(tr, va_iso, dev, matcher, tag)
        res[tag] = dict(iso=per_sep(*eval_boxbased(head, te_iso, dev)),
                        raw=per_sep(*eval_boxbased(head, te_raw, dev)), lr=lr, epoch=ep)

    print("\n" + "=" * 96)
    print("  Median |chi-centre error| in px on the LADDER test slice, by what the head trained on")
    print("  (phase Z ladder-trained 0.31 px at sep 8; the real trained head 3.83 px)")
    for key, lbl in (('iso', 'ISO=1 eval (2.43x8.5 boxes, == phase Z)'),
                     ('raw', 'ISO=0 eval (10.6x8.5 boxes, real shape)')):
        print(f"\n  {lbl}")
        print("  " + f"{'trained on':<22s}" + "".join(f"{('sep ' + str(s)):>10s}" for s in SEPS))
        for tag, _ in ARMS:
            ps = res[tag][key]
            row = "".join((f"{ps[s]['chi']:10.2f}" if s in ps else f"{'-':>10s}") for s in SEPS)
            print(f"  {tag:<22s}{row}")

    print("\n  training-set gap statistics (real organic reference: <5px 0.125)")
    print(f"  {'set':<22s}{'obj/frame':>11s}{'<5px':>9s}{'<10px':>9s}{'median gap':>12s}")
    for nm, st in (('real_clusters_ON', st_on), ('real_clusters_OFF', st_off)):
        print(f"  {nm:<22s}{st['obj_per_frame']:11.1f}{st['frac_lt5']:9.3f}"
              f"{st['frac_lt10']:9.3f}{st['median_gap']:12.1f}")

    json.dump(dict(results=res, stats=dict(clusters_on=st_on, clusters_off=st_off)),
              open(os.path.join(OUT, 'real_frame_head_probe.json'), 'w'), indent=2, default=str)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for (tag, _), c in zip(ARMS, ('tab:red', 'tab:orange', 'tab:blue')):
        ss = [s for s in SEPS if s in res[tag]['iso']]
        ax.plot(ss, [res[tag]['iso'][s]['chi'] for s in ss], 'o-', color=c, label=f'trained on {tag}')
    ax.axhline(3.83, ls=':', c='k', lw=1, label='the real trained head (3.83 px)')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('median |χ-centre error| (px)')
    ax.set_title('Phase Z.2: does training on REAL frames reproduce the merge?')
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_frame_head.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}\nPROBE DONE  ({time.time() - t0:.0f}s total)")


if __name__ == '__main__':
    main()
