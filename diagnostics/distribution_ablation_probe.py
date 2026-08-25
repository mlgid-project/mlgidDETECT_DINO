"""Phase Z.3 — WHICH property of the real training distribution teaches the merge?

Z.2 established that training this head on real simulated frames reproduces the merge (3.22 px at an
8 px gap) while training it on ladder frames does not (0.31 px), with the trunk frozen at convergence
and every competing loss absent. So the fault is in the training DISTRIBUTION. This asks which part.

Method: neutralise ONE property of the real frames at a time and retrain. Everything else is held at
Z.2's settings — frozen ssl1 trunk, only `enc_out_bbox_embed` moves, the real `interm_outputs` loss,
180 training frames, and THE SAME unmodified ladder test slice for every arm, so all numbers sit on
one scale. Neutralising is done by wrapping `simulate_labels`, which is where the object list is
final (clusters and ring-peaks already added), so the rendered image and its labels stay consistent.

Two anchors bracket every arm: **real = 3.22 px** (Z.2) and **ladder = 0.31 px** (Z). An arm that
falls back toward 0.31 names the property responsible.

ARMS AND THEIR PRE-REGISTERED READINGS. "Drops" = returns toward 0.31.

  real                  the Z.2 baseline, re-run here so every arm shares one eval set.

  fixed intensity       every object set to the ladder's constant 30 (real spans 10-50 for segments,
                        2-50 for rings).
     drops -> BRIGHTNESS VARIATION does it. Plainly: when a bright peak sits next to a faint one, the
     head learns to call the pair one object. Note this property is real -- actual GIWAXS peaks vary
     in intensity -- so the fix would have to be in the LOSS, not the simulator.

  fixed box size        every object replaced by its own class's median box, centre kept, so size
                        VARIATION is removed without changing the class mix.
     drops -> SIZE VARIATION does it. Plainly: when peaks legitimately come in many sizes, "one
     larger box" is a hypothesis the head has been taught is often right. Also a real property, so
     again a loss-side fix rather than a data one.

  no rings              every is_ring object dropped.
     drops -> THE RINGS do it. Plainly: rings are long objects, and learning to stretch a box across a
     long bright feature may be exactly what teaches the head to stretch one across two close peaks.
     This one IS actionable on the data/loss side (class weighting, or separating the two jobs).
     CONFOUND: dropping rings also drops object count, so if this arm wins it must be re-run against
     a random-subset control that removes the same NUMBER of objects. Not pre-run here.

  density -> 16         randomly subsampled to the ladder's 16 objects per frame.
     drops -> CROWDING does it. Plainly: with ~60 objects competing, covering more per box is cheap;
     with 16 it is not. Actionable via loss normalisation (`num_boxes`) or query budget.
     SAME CONFOUND in reverse -- this also changes the class mix.

  all four              every neutralisation at once. THE COMPLETENESS CONTROL.
     drops to ~0.3 while no single arm does -> it is a COMBINATION; no single knob is the fix.
     does NOT drop -> the property list is INCOMPLETE and something else about real frames is
     responsible. That is a real outcome, not a failed experiment, and must be reported as such.

  ladder                the Z baseline anchor.

Every arm also reports the wide rungs (16, 24 px), which bound the domain-gap contribution the same
way Z.2 did: a uniform mismatch degrades every rung, whereas the effect under study is close-pair
specific. Object count per frame is printed for every arm, since two of the neutralisations move it.

GPU, ~60 min. See tmp_diag/run_ablation.sbatch.
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
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, OUT
from diagnostics.head_only_probe import eval_boxbased, per_sep, split_frames, SEPS, \
    C_CLASS, C_BBOX, C_GIOU
from diagnostics.real_frame_head_probe import (cache, train_A, make_ladder_frames, sibling_gap,
                                               N_REAL)

LADDER_INT = 30.0        # the ladder's constant intensity (separation_ladder.INTENS)
LADDER_NOBJ = 16         # the ladder's objects per frame (N_PAIR=8 pairs x 2)
PAIR_PX = 10.0           # a same-q neighbour this close counts as 'paired' and is never thinned
MODES = ['real', 'fixed_intensity', 'fixed_boxsize', 'no_rings', 'drop_count_matched',
         'density16', 'all_four']
if os.environ.get('HEADPROBE_SMOKE') == '1':
    MODES = ['real', 'density16', 'all_four']


def class_median_boxes(dev, n=40):
    """Median box (w, h) per class, measured from the real simulator itself rather than assumed."""
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = True
    seg, ring = [], []
    for k in range(n):
        _sd = 777 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        try:
            _, bx, _, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy()
        r = isr.detach().cpu().numpy().astype(bool)
        wh = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        seg.append(wh[~r]); ring.append(wh[r])
    seg = np.concatenate([x for x in seg if len(x)]) if seg else np.zeros((0, 2))
    ring = np.concatenate([x for x in ring if len(x)]) if ring else np.zeros((0, 2))
    ms = np.median(seg, 0) if len(seg) else np.array([10.6, 8.5])
    mr = np.median(ring, 0) if len(ring) else np.array([10.6, 80.0])
    print(f"  class median box: segment {ms[0]:.1f} x {ms[1]:.1f} px, "
          f"ring {mr[0]:.1f} x {mr[1]:.1f} px", flush=True)
    return ms, mr


def _resize(b, isr, ms, mr):
    """Replace every box by its class median, keeping the centre. Removes size VARIATION only."""
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w = np.where(isr, mr[0], ms[0])
    h = np.where(isr, mr[1], ms[1])
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)


def _groups(b, pair_px=PAIR_PX, qtol=8.0):
    """Objects joined into groups by the same-q / close-in-chi relation, so a pair is one unit."""
    n = len(b)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    q = (b[:, 0] + b[:, 2]) / 2
    c = (b[:, 1] + b[:, 3]) / 2
    for i in range(n):
        for j in range(i + 1, n):
            if abs(q[i] - q[j]) < qtol and abs(c[i] - c[j]) < pair_px:
                a, bb = find(i), find(j)
                if a != bb:
                    par[a] = bb
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [np.array(v, dtype=int) for v in out.values()]


def _thin(b, it, r, target, rng):
    """Cut object count to `target` by dropping whole GROUPS, so pair structure AND composition hold.

    Two wrong ways to do this, both caught before the real run. Subsampling objects uniformly keeps a
    pair only if BOTH members survive -- (16/60)^2 = 0.07 -- so it strips close pairs far faster than
    objects and the arm silently becomes a rarity arm (already refuted three times: phase U, phase Z's
    diet sweep, Z.2's clusters ON/OFF dead heat). Keeping every pair and thinning only isolates
    over-corrects the other way: measured `<5 px` jumped to 0.714 against real frames' 0.200, which
    phase Z's diet curve says is worth about 0.2 px -- small against a 2.9 px effect, but a coupling
    all the same. Sampling whole groups keeps the paired fraction right in expectation, so density is
    the only thing that moves."""
    if len(b) <= target:
        return b, it, r
    gs = _groups(b)
    keep, cnt = [], 0
    for gi in rng.permutation(len(gs)):
        g = gs[gi]
        keep.extend(g.tolist()); cnt += len(g)
        if cnt >= target:
            break
    idx = np.array(sorted(keep), dtype=int)
    return b[idx], it[idx], r[idx]


def make_frames(dev, mode, n, ms, mr):
    """Real frames with ONE property neutralised. The wrap sits on `simulate_labels`, where the
    object list is final, so the rendered image always matches the labels it is given."""
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = True
    _orig = FastSimulation.simulate_labels

    def patched(_self=sim):
        boxes, intens, isr = _orig(_self)
        b = boxes.detach().cpu().numpy()
        it = intens.detach().cpu().numpy()
        r = isr.detach().cpu().numpy().astype(bool)
        if mode in ('no_rings', 'all_four'):
            k = ~r
            if k.sum() >= 2:
                b, it, r = b[k], it[k], r[k]
        if mode in ('density16', 'all_four'):
            b, it, r = _thin(b, it, r, LADDER_NOBJ, np.random.RandomState(len(b)))
        if mode == 'drop_count_matched':
            # the no_rings control: remove the SAME NUMBER of objects, drawn across both classes,
            # so no_rings vs this isolates 'ringness' from 'fewer objects'.
            b, it, r = _thin(b, it, r, max(len(b) - int(r.sum()), 2),
                             np.random.RandomState(1000 + len(b)))
        if mode in ('fixed_intensity', 'all_four'):
            it = np.full(len(b), LADDER_INT, dtype=np.float32)
        if mode in ('fixed_boxsize', 'all_four'):
            b = _resize(b, r, ms, mr)
        return (torch.tensor(b, dtype=torch.float32, device=_self.device),
                torch.tensor(it, dtype=torch.float32, device=_self.device),
                torch.tensor(r, device=_self.device).bool())
    if mode != 'real':
        sim.simulate_labels = patched

    frames, gaps, nobj = [], [], []
    k = 0
    while len(frames) < n and k < n * 4:
        _sd = 4242 + k                            # same seeds as Z.2, so frames are paired across arms
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
        gaps.append(sibling_gap(b)); nobj.append(len(b))
    g = np.concatenate(gaps) if gaps else np.zeros(0)
    fin = g[np.isfinite(g)]
    st = dict(frames=len(frames), obj_per_frame=float(np.mean(nobj)) if nobj else 0.0,
              frac_lt5=float((fin < 5).mean()) if len(fin) else 0.0)
    print(f"  [{mode}] {st['frames']} frames, {st['obj_per_frame']:.1f} obj/frame, "
          f"<5px {st['frac_lt5']:.3f}", flush=True)
    return frames, st


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    print(f"device={dev}  modes={MODES}", flush=True)

    ms, mr = class_median_boxes(dev)
    print("\n--- building frames", flush=True)
    lad = make_ladder_frames(dev, iso=True)
    sets, stats = {}, {}
    for m in MODES:
        sets[m], stats[m] = make_frames(dev, m, N_REAL, ms, mr)
    print(f"  built in {time.time() - t0:.0f}s", flush=True)

    print("\n--- caching through the frozen trunk", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    c_lad = cache(model, a, lad, dev, ladder=True)
    caches = {m: cache(model, a, sets[m], dev, ladder=False) for m in MODES}
    del model; torch.cuda.empty_cache()
    tr_lad, va_lad, te_lad = split_frames(c_lad)
    _g = np.concatenate([sibling_gap(np.asarray(f[1])) for f in lad])
    _g = _g[np.isfinite(_g)]
    stats['ladder'] = dict(obj_per_frame=float(np.mean([len(f[1]) for f in lad])),
                           frac_lt5=float((_g < 5).mean()) if len(_g) else 0.0)
    print(f"  ladder {len(c_lad)} -> {len(tr_lad)}/{len(va_lad)}/{len(te_lad)}; "
          + "  ".join(f"{m} {len(caches[m])}" for m in MODES), flush=True)

    matcher = HungarianMatcher(cost_class=C_CLASS, cost_bbox=C_BBOX, cost_giou=C_GIOU,
                               focal_alpha=0.25).to(dev)
    ARMS = [(m, caches[m][:N_REAL]) for m in MODES] + [('ladder', tr_lad)]
    res = {}
    for tag, tr in ARMS:
        print(f"\n--- arm {tag}  ({len(tr)} frames)", flush=True)
        head, lr, ep = train_A(tr, va_lad, dev, matcher, tag)
        res[tag] = dict(per_sep=per_sep(*eval_boxbased(head, te_lad, dev)), lr=lr, epoch=ep)

    print("\n" + "=" * 100)
    print("  Median |chi-centre error| in px on the ladder test slice, by what was neutralised")
    print("  anchors: real 3.22 (Z.2)   ladder 0.31 (Z)   the deployed head 3.83")
    print("  " + f"{'training frames':<20s}{'obj/fr':>8s}{'<5px':>7s}" +
          "".join(f"{('sep ' + str(s)):>9s}" for s in SEPS))
    for tag, _ in ARMS:
        ps = res[tag]['per_sep']
        st = stats.get(tag, {})
        n = st.get('obj_per_frame', float(LADDER_NOBJ))
        f5 = st.get('frac_lt5', 1.0)
        row = "".join((f"{ps[s]['chi']:9.2f}" if s in ps else f"{'-':>9s}") for s in SEPS)
        print(f"  {tag:<20s}{n:8.1f}{f5:7.3f}{row}")

    json.dump(dict(results=res, stats=stats), open(os.path.join(OUT, 'ablation_probe.json'), 'w'),
              indent=2, default=str)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, _ in ARMS:
        ps = res[tag]['per_sep']
        ss = [s for s in SEPS if s in ps]
        ax.plot(ss, [ps[s]['chi'] for s in ss], 'o-', label=tag, lw=2 if tag in ('real', 'ladder') else 1)
    ax.axhline(3.83, ls=':', c='k', lw=1, label='the deployed head (3.83 px)')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('median |χ-centre error| (px)')
    ax.set_title('Phase Z.3: which property of the real frames teaches the merge?')
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}\nPROBE DONE  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
