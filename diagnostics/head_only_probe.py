"""Phase Z — head-only training: can the real objective teach the head what a straight line knows?

Phase Y settled that the encoder's box head receives everything it needs. Ridge regression on the
exact 256-number vector `enc_out_bbox_embed` reads recovers each token's chi-offset to ITS OWN peak
to 0.49 px at an 8 px separation, where the trained head is off by 3.83 px. So the ceiling is a
TRAINING-SIGNAL fault, not a perceptual one. Four candidate mechanisms have already been proposed and
refuted in this investigation, so this probe does not propose a fifth: it MEASURES which part of the
training signal is responsible, by holding everything else identical.

Only four things touch that head: the features (settled — fine), the target parameterization, the
loss, and the assignment that decides which query is supervised by which peak. This runs four arms on
ONE cached feature set, so consecutive arms differ in exactly one of those.

  A  detection loss     sigmoid(MLP(mem) + anchor), Hungarian-matched, 5*L1 + 2*GIoU.
                        Byte-for-byte the `interm_outputs` loss that trains this head in the real run
                        (dino.py:617-633, interm_loss_coef 1.0, no_interm_box_loss False).
  B  oracle assignment  IDENTICAL parameterization and loss, but each ladder token is supervised by
                        its own peak instead of by the Hungarian match.        A vs B = ASSIGNMENT.
  C  pixel target       Same MLP, same features, but predicts the signed chi-offset in PIXELS under a
                        plain L1 -- no anchor, no sigmoid, no GIoU.        B vs C = PARAMETERIZATION.
  D  ridge (pixel)      Phase Y's straight line, recomputed here as the reference.  C vs D = CAPACITY.

  D' ridge (logit)      D's straight line fitted to the head's OWN target -- the logit-space delta
                        over the anchor -- and read back out in px. D vs D' isolates the anchor+sigmoid
                        encoding from the regression itself, independent of any optimizer.

Reading it. Every arm reports the SAME phase-Y metric, so the numbers sit on one scale: median
|predicted box chi-centre - its own peak| in px, on held-out FRAMES, per separation rung.

  A ~ 0.5   the objective is fine in isolation and the full run fails for a reason outside this head
            (it loses to the rest of the loss, or to optimization) -> lever: reweight / curriculum.
  A ~ 3.8, B ~ 0.5    the loss and parameterization are fine and the ASSIGNMENT is the fault.
  A ~ B ~ 3.8, C ~ 0.5   the anchor+sigmoid+GIoU target itself is what cannot be learned.
  C ~ 3.8                the MLP cannot do what ridge does -- unlikely (it strictly contains a linear
                         map) and would mean an optimization pathology, not a signal one.

Nothing here is a full training run: the trunk is frozen, its features are cached once, and only the
2-layer-hidden box MLP moves. Everything downstream of this file is unchanged, and no checkpoint is
written -- this is a measurement, not a candidate model.

Controls. Frame-level 60/20/20 split with phase Y's seed, so the rungs are directly comparable.
Learning rate AND epoch are chosen on the VALIDATION slice only, never on test, and the lr sweep
spans three decades so "it did not learn" cannot be an lr artefact. Every frame carries exactly two
peaks per pair, so total flux is constant across rungs.

Also reported per arm: the median |height error| in px. The chi extent is the coordinate that a
merged box gets wrong (X.3: the box SPANS the pair), and it sits far out in the sigmoid tail --
anchor h = 0.05 against a true 8.5/512 = 0.0166 -- where dp/dlogit is ~15x smaller than it is for the
centre near mid-image. That is arithmetic off gen_encoder_output_proposals, not a claim; these two
columns are what would turn it into one, or kill it.

GPU, ~25 min. See tmp_diag/run_head_only.sbatch (env: HEADPROBE_MODELS, LADDER_ISO).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('LADDER_AXIS', 'chi')
os.environ.setdefault('LADDER_ISO', '1')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import diagnostics.separation_ladder as L
L.N_IMG = 50                                   # phase Y's frame count; a 256-dim fit needs N >> 256
import models.dino.deformable_transformer as DT
from models.dino.utils import MLP
from models.dino.matcher import HungarianMatcher
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from diagnostics.objectness_probe import layout_for, H_IMG, W_IMG
from diagnostics.prominence_probe import build_model_from_ckpt, OUT
from diagnostics.linear_readout_probe import ridge_fit, ridge_pred

SEPS = [4, 6, 8, 12, 16, 24]
NQ = 900                                       # num_queries: the two-stage top-k
# The real coefficients, read off config/DINO/DINO_4scale_swin.py and dino.py:798-836.
W_L1, W_GIOU = 5.0, 2.0
C_CLASS, C_BBOX, C_GIOU = 2.0, 5.0, 2.0
LRS = (1e-5, 1e-4, 1e-3)
EPOCHS = 300
BATCH = 8
MIN_N = 20              # rungs with fewer held-out tokens than this are not reported
ANC_CLAMP = 20.0        # gen_encoder_output_proposals writes +inf for border cells; sigmoid(20)==1.0
                        # to float precision and its gradient is ~0, so this reproduces the real
                        # forward pass exactly while staying finite.
if os.environ.get('HEADPROBE_SMOKE') == '1':      # wiring check only -- numbers are meaningless
    L.N_IMG, LRS, EPOCHS, MIN_N = 4, (1e-4,), 25, 4


# --------------------------------------------------------------------------------- feature cache
def build_cache(ckpt, cfg_file, data, dev):
    """Run the frozen model once and keep, per frame, exactly what enc_out_bbox_embed is fed."""
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    rec = {}
    _orig = DT.gen_encoder_output_proposals

    def _gen(*args, **kw):
        om, op = _orig(*args, **kw)
        rec['anchor'] = op.detach()
        return om, op
    DT.gen_encoder_output_proposals = _gen

    def hk(mod, inp, outp):
        if inp[0].shape[1] > 5000:              # the UNSELECTED call, over the whole pyramid
            rec['mem'] = inp[0].detach()
    h1 = model.transformer.enc_out_bbox_embed.register_forward_hook(hk)

    def hc(mod, inp, outp):
        if inp[0].shape[1] > 5000:
            rec['cls'] = outp.detach()
    h2 = model.transformer.enc_out_class_embed.register_forward_hook(hc)

    frames, lay, fid, n_tok_seen, n_tok_kept = [], {}, 0, 0, 0
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
                if not {'mem', 'anchor', 'cls'} <= set(rec):
                    continue
                mem, anc, cls = rec['mem'][0], rec['anchor'][0], rec['cls'][0]
                if not lay:
                    st, sh, of = layout_for(mem.shape[0]); lay.update(st=st, sh=sh, of=of)
                    print(f"  finest stride {st[0]}, {mem.shape[0]} tokens, dim {mem.shape[1]}",
                          flush=True)
                st, sh, of = lay['st'], lay['sh'], lay['of']
                s_, (hh, ww) = st[0], sh[0]

                # the model's own selection: top-900 by max class logit (transformer:411-412)
                topk = torch.topk(cls.max(-1)[0], NQ, dim=0)[1]
                inv = torch.full((mem.shape[0],), -1, dtype=torch.long, device=dev)
                inv[topk] = torch.arange(NQ, device=dev)

                # GT in normalized cxcywh, the criterion's format
                g = torch.tensor(np.asarray(gt), dtype=torch.float32, device=dev)
                cx = (g[:, 0] + g[:, 2]) / 2 / W_IMG
                cy = (g[:, 1] + g[:, 3]) / 2 / H_IMG
                bw = (g[:, 2] - g[:, 0]).abs() / W_IMG
                bh = (g[:, 3] - g[:, 1]).abs() / H_IMG
                boxes = torch.stack([cx, cy, bw, bh], 1)

                # the two ladder tokens per pair -- the phase-Y evaluation points
                toks = []
                for i in range(0, len(g) - 1, 2):
                    gq = float((g[i, 0] + g[i, 2] + g[i + 1, 0] + g[i + 1, 2]) / 4)
                    col = int(np.clip(round(gq / s_ - 0.5), 0, ww - 1))
                    for j in (i, i + 1):
                        peak = float((g[j, 1] + g[j, 3]) / 2)
                        r = int(np.clip(round(peak / s_ - 0.5), 0, hh - 1))
                        k = of[0] + r * ww + col
                        n_tok_seen += 1
                        p = int(inv[k])
                        if p < 0:                       # not among the 900 -- phase X.1 says rare
                            continue
                        n_tok_kept += 1
                        toks.append((p, peak, (r + 0.5) * s_, j))
                if not toks:
                    continue
                frames.append(dict(
                    mem=mem[topk].half().cpu(),
                    anc=anc[topk].clamp(-ANC_CLAMP, ANC_CLAMP).cpu(),
                    cls=cls[topk].float().cpu(),
                    boxes=boxes.cpu(),
                    labels=torch.zeros(len(g), dtype=torch.long),   # ladder plants segments only
                    toks=toks, sep=sep, fid=fid))
    h1.remove(); h2.remove()
    DT.gen_encoder_output_proposals = _orig
    del model
    torch.cuda.empty_cache()
    print(f"  cached {len(frames)} frames; ladder tokens inside the top-{NQ}: "
          f"{n_tok_kept}/{n_tok_seen} = {n_tok_kept / max(n_tok_seen, 1):.3f}", flush=True)
    return frames


def split_frames(frames):
    """Phase Y's split, same seed, at FRAME level -- tokens from one frame are correlated."""
    uf = np.unique([f['fid'] for f in frames])
    rng = np.random.RandomState(0); rng.shuffle(uf)
    n_tr, n_va = int(0.6 * len(uf)), int(0.2 * len(uf))
    s = {f: 'train' for f in uf[:n_tr]}
    s.update({f: 'val' for f in uf[n_tr:n_tr + n_va]})
    s.update({f: 'test' for f in uf[n_tr + n_va:]})
    return ([f for f in frames if s[f['fid']] == 'train'],
            [f for f in frames if s[f['fid']] == 'val'],
            [f for f in frames if s[f['fid']] == 'test'])


# ------------------------------------------------------------------------------------ the arms
def new_head(out_dim, dev):
    """The real head: MLP(256, 256, 4, 3) with a zero-initialised last layer (dino.py:133-139),
    so it starts by emitting the anchor unchanged -- identical to a fresh detector."""
    m = MLP(256, 256, out_dim, 3).to(dev)
    nn.init.constant_(m.layers[-1].weight.data, 0)
    nn.init.constant_(m.layers[-1].bias.data, 0)
    return m


def box_loss(pred, tgt):
    """SetCriterion.loss_boxes: L1 + GIoU, summed and normalised by the box count."""
    l1 = torch.nn.functional.l1_loss(pred, tgt, reduction='none').sum()
    giou = (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred),
                                               box_cxcywh_to_xyxy(tgt)))).sum()
    return (W_L1 * l1 + W_GIOU * giou) / max(len(tgt), 1)


def eval_boxbased(head, frames, dev):
    """Median |chi-centre error| and |height error| in px, per rung, at the ladder tokens."""
    ec, eh, sp = [], [], []
    with torch.no_grad():
        for f in frames:
            mem = f['mem'].to(dev).float(); anc = f['anc'].to(dev)
            box = (head(mem) + anc).sigmoid()
            for p, peak, _tok, j in f['toks']:
                ec.append(abs(float(box[p, 1]) * H_IMG - peak))
                eh.append(abs(float(box[p, 3]) * H_IMG - float(f['boxes'][j, 3]) * H_IMG))
                sp.append(f['sep'])
    return np.array(ec), np.array(eh), np.array(sp)


def eval_pixelbased(head, frames, dev):
    """Arm C: the head emits the signed offset directly, so the centre is token + offset."""
    ec, sp = [], []
    with torch.no_grad():
        for f in frames:
            mem = f['mem'].to(dev).float()
            off = head(mem)[:, 0] * H_IMG          # trained in normalized units, read out in px
            for p, peak, tok, _j in f['toks']:
                ec.append(abs(tok + float(off[p]) - peak))
                sp.append(f['sep'])
    return np.array(ec), np.full(len(ec), np.nan), np.array(sp)


def assignment_stats(head, frames, matcher, dev):
    """Arm A only: does each peak's Hungarian match land on that peak's OWN ladder token?

    A vs B differs solely in the assignment, so this says WHAT the assignment does wrong -- whether
    a peak is supervised by a query sitting somewhere else entirely, or by its own token which then
    still emits the wrong box. Reported per rung as the hit rate plus the median chi-distance from
    the matched query's token centre to the peak it was handed."""
    hit, dist, sp = [], [], []
    with torch.no_grad():
        for f in frames:
            mem = f['mem'].to(dev).float()[None]
            anc = f['anc'].to(dev)[None]
            pred = (head(mem) + anc).sigmoid()
            tg = [dict(boxes=f['boxes'].to(dev), labels=f['labels'].to(dev))]
            (si, ti), = matcher(dict(pred_logits=f['cls'].to(dev)[None], pred_boxes=pred), tg)
            own = {j: (p, peak) for p, peak, _tok, j in f['toks']}
            qtok = {p: tok for p, _peak, tok, _j in f['toks']}
            for q, t in zip(si.tolist(), ti.tolist()):
                if t not in own:
                    continue
                p_own, peak = own[t]
                hit.append(1.0 if q == p_own else 0.0)
                dist.append(abs(qtok.get(q, float('nan')) - peak) if q in qtok else float('nan'))
                sp.append(f['sep'])
    hit, dist, sp = np.array(hit), np.array(dist), np.array(sp)
    return {s: dict(hit=float(hit[sp == s].mean()),
                    dist=(float(np.nanmedian(dist[sp == s]))
                          if np.any(~np.isnan(dist[sp == s])) else None),
                    n=int((sp == s).sum()))
            for s in SEPS if (sp == s).sum() >= MIN_N}


def train_arm(arm, tr, va, dev, matcher=None):
    """One arm, lr swept over three decades, lr AND epoch chosen on validation only."""
    best = None
    for lr in LRS:
        torch.manual_seed(0)
        head = new_head(1 if arm == 'C' else 4, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        order = np.arange(len(tr))
        rng = np.random.RandomState(1)
        for ep in range(EPOCHS):
            rng.shuffle(order)
            head.train()
            for b0 in range(0, len(order), BATCH):
                idx = order[b0:b0 + BATCH]
                loss, nb = 0.0, 0
                if arm == 'A':
                    mem = torch.stack([tr[i]['mem'] for i in idx]).to(dev).float()
                    anc = torch.stack([tr[i]['anc'] for i in idx]).to(dev)
                    cls = torch.stack([tr[i]['cls'] for i in idx]).to(dev)
                    pred = (head(mem) + anc).sigmoid()
                    tgts = [dict(boxes=tr[i]['boxes'].to(dev), labels=tr[i]['labels'].to(dev))
                            for i in idx]
                    ind = matcher(dict(pred_logits=cls, pred_boxes=pred), tgts)
                    src = torch.cat([pred[b][s.to(dev)] for b, (s, _) in enumerate(ind)])
                    dst = torch.cat([tgts[b]['boxes'][t.to(dev)]
                                     for b, (_, t) in enumerate(ind)])
                    nb = len(dst)
                    loss = box_loss(src, dst)
                else:
                    src, dst = [], []
                    for i in idx:
                        f = tr[i]
                        pos = [p for p, _, _, _ in f['toks']]
                        mem = f['mem'][pos].to(dev).float()
                        if arm == 'B':
                            src.append((head(mem) + f['anc'][pos].to(dev)).sigmoid())
                            dst.append(f['boxes'].to(dev)[[j for _, _, _, j in f['toks']]])
                        else:                                   # arm C: pixels, no anchor
                            src.append(head(mem)[:, 0])
                            dst.append(torch.tensor(
                                [(peak - tok) / H_IMG for _, peak, tok, _ in f['toks']],
                                dtype=torch.float32, device=dev))
                    src, dst = torch.cat(src), torch.cat(dst)
                    nb = len(dst)
                    loss = box_loss(src, dst) if arm == 'B' else \
                        W_L1 * torch.nn.functional.l1_loss(src, dst, reduction='sum') / len(dst)
                if nb == 0:
                    continue
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
                head.eval()
                ev = eval_pixelbased if arm == 'C' else eval_boxbased
                ec, _, _ = ev(head, va, dev)
                v = float(np.median(ec))
                if best is None or v < best[0]:
                    best = (v, lr, ep + 1, {k: t.detach().clone() for k, t in head.state_dict().items()})
        print(f"    arm {arm} lr={lr:g}: best val so far {best[0]:.2f} px "
              f"(lr={best[1]:g}, ep={best[2]})", flush=True)
    head = new_head(1 if arm == 'C' else 4, dev)
    head.load_state_dict(best[3]); head.eval()
    return head, best[1], best[2], best[0]


def ridge_arm(tr, va, te, dev, space):
    """Arms D / D'. `space` = 'px' (phase Y's target) or 'logit' (the head's own target)."""
    def pack(frames):
        X, y, s = [], [], []
        for f in frames:
            mem = f['mem'].float().numpy()
            for p, peak, tok, j in f['toks']:
                X.append(mem[p])
                if space == 'px':
                    y.append(peak - tok)
                else:
                    cy = float(f['boxes'][j, 1])
                    y.append(float(np.log(cy / (1 - cy))) - float(f['anc'][p, 1]))
                s.append((f['sep'], peak, tok, float(f['anc'][p, 1])))
        return (torch.tensor(np.asarray(X), dtype=torch.float64),
                torch.tensor(np.asarray(y), dtype=torch.float64), s)

    Xtr, ytr, _ = pack(tr); Xva, yva, _ = pack(va); Xte, yte, ste = pack(te)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    best = None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        w = ridge_fit(Xtr, ytr, lam)
        e = (ridge_pred(Xva, w) - yva).abs().median().item()
        if best is None or e < best[0]:
            best = (e, lam, w)
    pr = ridge_pred(Xte, best[2]).numpy()
    ec, sp = [], []
    for v, (sep, peak, tok, anc_cy) in zip(pr, ste):
        chi = (tok + v) if space == 'px' else float(1 / (1 + np.exp(-(v + anc_cy))) * H_IMG)
        ec.append(abs(chi - peak)); sp.append(sep)
    print(f"    ridge[{space}] lambda={best[1]:g}, val {best[0]:.3f}", flush=True)
    return np.array(ec), np.full(len(ec), np.nan), np.array(sp)


CLOSE = (4, 6, 8)
# Real organic: 12.5% of adjacent-peak gaps are <5 px; phase U's clusters diet reached 17.7%. The
# sweep brackets both. f is the fraction of training FRAMES drawn from the close rungs -- a frame
# fraction, not a gap fraction, so it is the right order of magnitude and not the same statistic.
DILUTIONS = (0.50, 0.25, 0.12, 0.06)


def dilution_sweep(tr, va, te, dev, matcher, lr):
    """Arm A again at four close-pair diets, training-set SIZE held fixed so only the mix varies.

    Arm A as built sees rungs 4/6/8 in half its frames. If it learns the offset there but stops as
    the diet thins toward the real 12.5%, rarity is sufficient to explain the full run's failure and
    the lever is the data mix. If it keeps learning down to 0.06, rarity is NOT the explanation and
    the fault is in how the head is trained jointly. lr is inherited from arm A rather than re-swept,
    to keep this a single-variable sweep over the diet.
    """
    close = [f for f in tr if f['sep'] in CLOSE]
    far = [f for f in tr if f['sep'] not in CLOSE]
    if not close or not far:
        print('    diet sweep skipped: a rung pool is empty', flush=True)
        return {}
    out = {}
    for frac in DILUTIONS:
        rng = np.random.RandomState(7)
        n_c = int(round(frac * len(tr)))
        pool = ([close[i] for i in rng.randint(0, len(close), n_c)] +
                [far[i] for i in rng.randint(0, len(far), len(tr) - n_c)])
        torch.manual_seed(0)
        head = new_head(4, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        order = np.arange(len(pool))
        r2 = np.random.RandomState(1)
        best = None
        for ep in range(EPOCHS):
            r2.shuffle(order)
            head.train()
            for b0 in range(0, len(order), BATCH):
                idx = order[b0:b0 + BATCH]
                mem = torch.stack([pool[i]['mem'] for i in idx]).to(dev).float()
                anc = torch.stack([pool[i]['anc'] for i in idx]).to(dev)
                cls = torch.stack([pool[i]['cls'] for i in idx]).to(dev)
                pred = (head(mem) + anc).sigmoid()
                tgts = [dict(boxes=pool[i]['boxes'].to(dev), labels=pool[i]['labels'].to(dev))
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
                    best = (v, {k: t.detach().clone() for k, t in head.state_dict().items()})
        head = new_head(4, dev); head.load_state_dict(best[1]); head.eval()
        out[frac] = per_sep(*eval_boxbased(head, te, dev))
        got = out[frac].get(8, {}).get('chi')
        print(f"    diet f={frac:.2f} (close frames {n_c}/{len(tr)}): "
              f"sep-8 test {got if got is None else round(got, 2)} px", flush=True)
    return out


def per_sep(ec, eh, sp):
    return {s: dict(chi=float(np.median(ec[sp == s])),
                    h=(float(np.median(eh[sp == s])) if not np.all(np.isnan(eh[sp == s])) else None),
                    n=int((sp == s).sum()))
            for s in SEPS if (sp == s).sum() >= MIN_N}


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    only = os.environ.get('HEADPROBE_MODELS', 'ssl1').split(',')
    print(f"device={dev}  frames/rung={L.N_IMG}  models={only}  ISO={os.environ['LADDER_ISO']}",
          flush=True)
    L.SEPS = SEPS
    data = L.make_images(dev)
    matcher = HungarianMatcher(cost_class=C_CLASS, cost_bbox=C_BBOX, cost_giou=C_GIOU,
                               focal_alpha=0.25).to(dev)
    summary = {}
    for name, ckpt, cfg_file in L.MODELS:
        if name not in only:
            continue
        print(f"\n########## {name} ##########", flush=True)
        t0 = time.time()
        frames = build_cache(ckpt, cfg_file, data, dev)
        tr, va, te = split_frames(frames)
        print(f"  frames train/val/test = {len(tr)}/{len(va)}/{len(te)}  "
              f"(cache {time.time() - t0:.0f}s)", flush=True)

        res = {}
        for arm in ('A', 'B', 'C'):
            print(f"  --- arm {arm}", flush=True)
            head, lr, ep, v = train_arm(arm, tr, va, dev, matcher)
            ev = eval_pixelbased if arm == 'C' else eval_boxbased
            res[arm] = dict(per_sep=per_sep(*ev(head, te, dev)), lr=lr, epoch=ep, val=v)
            if arm == 'A':
                res['A_assign'] = assignment_stats(head, te, matcher, dev)
        print(f"  --- diet sweep (arm A repeated, lr={res['A']['lr']:g})", flush=True)
        res['A_diet'] = dilution_sweep(tr, va, te, dev, matcher, res['A']['lr'])
        for arm, space in (('D', 'px'), ('Dp', 'logit')):
            print(f"  --- arm {arm} (ridge, {space})", flush=True)
            res[arm] = dict(per_sep=per_sep(*ridge_arm(tr, va, te, dev, space)))
        summary[name] = res

    LBL = {'A': 'A detection loss', 'B': 'B oracle assign', 'C': 'C pixel target',
           'D': 'D ridge px', 'Dp': "D' ridge logit"}
    print("\n" + "=" * 100)
    print("  Median |chi-centre error| in px at the ladder tokens, held-out frames")
    print("  (phase Y reference, ssl1: trained head 3.83 px at sep 8; ridge 0.49 px)")
    for nm, r in summary.items():
        print(f"\n  --- {nm}")
        print("  " + f"{'arm':<17s}" + "".join(f"{('sep ' + str(s)):>10s}" for s in SEPS))
        for arm in ('A', 'B', 'C', 'D', 'Dp'):
            row = "".join(
                (f"{r[arm]['per_sep'][s]['chi']:10.2f}" if s in r[arm]['per_sep'] else f"{'-':>10s}")
                for s in SEPS)
            print(f"  {LBL[arm]:<17s}{row}")
        print("\n  " + f"{'height err (px)':<17s}" +
              "".join(f"{('sep ' + str(s)):>10s}" for s in SEPS))
        for arm in ('A', 'B'):
            row = "".join(
                (f"{r[arm]['per_sep'][s]['h']:10.2f}"
                 if s in r[arm]['per_sep'] and r[arm]['per_sep'][s]['h'] is not None
                 else f"{'-':>10s}") for s in SEPS)
            print(f"  {LBL[arm]:<17s}{row}")
        a = r.get('A_assign', {})
        print("\n  " + f"{'arm A assignment':<17s}" +
              "".join(f"{('sep ' + str(s)):>10s}" for s in SEPS))
        print("  " + f"{'  matched own tok':<17s}" +
              "".join((f"{a[s]['hit']:10.2f}" if s in a else f"{'-':>10s}") for s in SEPS))
        print("  " + f"{'  match dist px':<17s}" +
              "".join((f"{a[s]['dist']:10.2f}" if s in a and a[s]['dist'] is not None
                       else f"{'-':>10s}") for s in SEPS))
        d = r.get('A_diet', {})
        if d:
            print("\n  arm A vs close-pair diet (real organic = 0.125 of gaps <5 px; "
                  "phase U clusters = 0.177)")
            print("  " + f"{'close-frame frac':<17s}" +
                  "".join(f"{('sep ' + str(s)):>10s}" for s in SEPS))
            for frac in DILUTIONS:
                ps = d.get(frac) or d.get(str(frac)) or {}
                row = "".join((f"{ps[s]['chi']:10.2f}" if s in ps else f"{'-':>10s}") for s in SEPS)
                print(f"  {frac:<17.2f}{row}")

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for nm, r in summary.items():
        for arm, st in (('A', 'o-'), ('B', 's-'), ('C', '^--'), ('D', 'd:')):
            ss = [s for s in SEPS if s in r[arm]['per_sep']]
            ax.plot(ss, [r[arm]['per_sep'][s]['chi'] for s in ss], st, label=f'{nm}: {LBL[arm]}')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('median |χ-centre error| (px)')
    ax.set_title('Phase Z: which part of the training signal loses the offset?')
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'head_only.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}")
    json.dump(summary, open(os.path.join(OUT, 'head_only_probe.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
