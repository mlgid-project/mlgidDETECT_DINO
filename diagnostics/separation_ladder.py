"""
Phase V — synthetic separation ladder: what is the architecture's ACTUAL chi-resolution limit?

Phase U was negative: training on tight azimuthal pairs did not improve close-pair recall at a
matched operating point. That ELIMINATES the training-data explanation but does not establish the
remaining one (stride-8 features cannot separate peaks 3.9 px apart), which is still an a priori
argument -- the same kind phase R used and measurement refuted. Before spending days on a
512x1024 -> 1024x1024 chi-resolution run, measure the limit directly.

Method. Plant EXACTLY TWO peaks at a known chi-separation, several well-separated pairs per frame,
and render through the FULL appearance pipeline (noise, background, detector gaps, HE) by patching
`simulate_labels` and calling the real `simulate_img` -- so the frames are in-domain, not toy
images. Box dimensions are the measured real organic GT medians (q-width 10.6 px, chi-height
8.5 px). One fixed image set, both models, so the only variable is the model.

Per pair the outcome is RESOLVED (both GT boxes matched), MERGED (exactly one matched -- the model
saw the feature but emitted a single box) or MISSED (neither). Compared at a matched operating
point: for each model the score threshold whose total detection count is closest to the true
planted-peak count, so neither model is credited for simply detecting more often (the phase-P
lesson, and the trap phase U's raw table fell into).

Reading the result:
  limit ~8 px, clean above         -> stride-limited; finer chi should halve it; lever justified
  limit >> 8 px                    -> something else binds; resolution will not fix it
  resolves 4 px pairs fine here    -> architecture is NOT the limit, so the real-data failure is the
                                      sim-to-real gap or the labels, and a resolution run is wasted
                                      (compare against clusters_gate.py real <5px recall: ssl1 0.352,
                                      clusters 0.321 at matched count)

RESULT (chi axis, anisotropic stimulus): all three models wall at 16 px, INCLUDING the stride-4
5-scale model. Feature stride is therefore not the mechanism, and neither 512->1024 (which moves a
real 3.9 px gap only to 7.8 px, still under the wall) nor stride-4 buys anything. Both chi-resolution
routes are dead; the mechanism is still open.

AXIS / ISO -- the anisotropy test. The only strongly anisotropic thing in the architecture is the
Swin window: window_size_h=48 (chi) vs window_size_w=6 (q), 8:1, and window_partition confirms h
indexes chi. If the window sets the wall, the q limit should be ~8x finer than the chi limit; if both
axes wall together, the window is not the cause. Run it with LADDER_ISO=1 -- see the SIGMA block
below for why the anisotropic stimulus makes the two axes incomparable.

GPU, ~20 min per configuration. See tmp_diag/run_separation_ladder.sbatch (env: LADDER_AXIS=chi|q,
LADDER_ISO=0|1, LADDER_NMS=1|0).
"""
import os, sys, json, random
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import simulation as S
from simulation import FastSimulation
from util.configuration import Config
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, OUT

_CUR = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation'
# (name, checkpoint, config). The 5-scale entry needs its OWN architecture config:
# return_interm_indices=[0,1,2,3] adds a STRIDE-4 level, which is the variable under test here.
# It shares ssl1's backbone init and recipe, so vs ssl1 the only difference is feature stride.
MODELS = [('ssl1', CKPT_A, CONFIG),
          ('clusters', '/mnt/lustre/work/schreiber/szb389/tmp_diag/clusters_probe_ckpt.pth', CONFIG),
          ('5scale_stride4', f'{_CUR}/detector_runs/dino_5scale_scratch/checkpoint.pth',
           os.path.join(_REPO, 'config/DINO/DINO_5scale_swin_ssl.py'))]
SEPS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]      # separation rungs (px)
# AXIS: 'chi' separates the pair along chi (the failure axis in the real data); 'q' separates it
# along q. The ONLY strongly anisotropic thing in the architecture is the Swin window --
# window_size_h=48 (chi) vs window_size_w=6 (q), an 8:1 ratio, and window_partition confirms h
# indexes chi. If the window sets the resolution limit, the q limit should be ~8x finer than the
# chi limit. If both axes wall at the same separation, the window is NOT the cause.
AXIS = os.environ.get('LADDER_AXIS', 'chi')
N_IMG = 20                                        # frames per rung
N_PAIR = 8                                        # pairs per frame (well separated in q)
INTENS = 30.0                                     # mid-range of seg_intensity_range (10, 50)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]

# ---------------------------------------------------------------------------------------------
# STIMULUS SHAPE. The simulator's box<->Gaussian convention is ANISOTROPIC (simulation.py:816):
#     sigma_q   = box_width  / w_coef     w_coef = 1.0
#     sigma_chi = box_height / a_coef     a_coef = 3.5
# so the real median box (10.6 x 8.5) renders a peak 10.6/2.43 = 4.4x BROADER along q than along
# chi. Planting that box and separating the pair along q therefore presents a far HARDER stimulus
# than separating it along chi, and the first q-axis ladder was confounded by exactly this: read in
# raw pixels q looked worse than chi, read in units of the rendered sigma q looked ~2x better, so
# the comparison decided nothing.
#
# LADDER_ISO=1 plants the box that renders an ISOTROPIC peak (sigma_q == sigma_chi == SIGMA), so
# both axes carry an IDENTICAL stimulus and any surviving difference is the model. SIGMA is the real
# chi width (8.5 / 3.5 = 2.43 px), which leaves the chi profile UNCHANGED from the anisotropic run
# -- so the iso chi ladder doubles as a control: it should reproduce the 16 px chi wall.
# Caveat: the resulting box (2.43 x 8.5) is out-of-distribution for the box REGRESSION target
# (models were trained on ~10.6 x 8.5). That is acceptable here because the hypothesis under test --
# the Swin window's 48(chi):6(q) anisotropy -- lives in the backbone, not the box head.
ISO = os.environ.get('LADDER_ISO', '0') == '1'
SIGMA = 8.5 / S.SimulationConfig.a_coef           # 2.43 px, the real chi Gaussian width
if ISO:
    BOX_W = SIGMA * S.SimulationConfig.w_coef     # 2.43
    BOX_H = SIGMA * S.SimulationConfig.a_coef     # 8.5
else:
    BOX_W, BOX_H = 10.6, 8.5                      # measured real organic GT medians

# NMS CONTROL. Class-aware NMS uses IoU 0.4 for segments and 0.1 for rings, and IoU depends on the
# BOX geometry, which stays anisotropic (2.43 wide x 8.5 tall) even when the RENDER is isotropic.
# Two boxes offset by d along an axis of extent e have IoU = (e-d)/(e+d), so at d=2 px the chi pair
# sits at IoU 0.62 (suppressed) while the q pair sits at 0.10 (kept) -- an axis asymmetry that has
# nothing to do with perception. It only bites at d<=3 px on chi, far below the 16 px wall, but
# LADDER_NMS=0 removes it entirely (IoU thresholds -> 1.0) so the wall can be confirmed on raw
# model output.
NMS = os.environ.get('LADDER_NMS', '1') == '1'
TAG = f"{AXIS}{'_iso' if ISO else ''}{'' if NMS else '_nonms'}"


def plant(sep, rng):
    """Two peaks at separation `sep` along AXIS, N_PAIR times. Pairs are placed at distinct q AND
    distinct chi so they cannot collide whichever axis carries the separation."""
    H, W = S.HEIGHT, S.WIDTH
    m = sep / 2 + 8
    qs = np.linspace(0.15 * W + m, 0.90 * W - m, N_PAIR) + rng.uniform(-8, 8, N_PAIR)
    lo, hi = 0.22 * H + m, 0.78 * H - m          # clear of the angle-limit bands
    cs = np.linspace(lo, hi, N_PAIR) + rng.uniform(-6, 6, N_PAIR)
    boxes = []
    for q, c in zip(qs, cs):
        for d in (-sep / 2, +sep / 2):
            qq, cc = (q, c + d) if AXIS == 'chi' else (q + d, c)
            boxes.append([qq - BOX_W / 2, cc - BOX_H / 2, qq + BOX_W / 2, cc + BOX_H / 2])
    return np.array(boxes, dtype=np.float32)


def make_images(dev):
    """One fixed image set, reused for every model."""
    sim = FastSimulation(device=dev)
    sim.sim_config.prob_single_obj = 0.0          # never truncate to a single object
    data = {}
    for sep in SEPS:
        rng = np.random.RandomState(1234 + sep)
        frames = []
        for k in range(N_IMG):
            _sd = 9000 + sep * 100 + k
            random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)   # np.random is
            # used inside the appearance pipeline; without it the image set drifted between
            # runs and small-separation rungs carried ~+-0.05 noise
            gt = torch.tensor(plant(sep, rng), device=dev)
            inten = torch.full((len(gt),), INTENS, device=dev)
            isr = torch.zeros(len(gt), dtype=torch.bool, device=dev)
            sim.simulate_labels = lambda g=gt, i=inten, r=isr: (g.clone(), i.clone(), r.clone())
            try:
                img, bx, mask, _ = sim.simulate_img()
            except Exception as e:
                print(f"    sep={sep} frame {k} failed: {type(e).__name__}: {e}")
                continue
            # drop pairs sitting in detector-mask / no-data: invisible to ANY model
            m = mask.detach().cpu().numpy().squeeze().astype(bool) if mask is not None \
                else np.ones(img.shape[-2:], bool)
            bb = bx.detach().cpu().numpy()
            keep = []
            for i in range(0, len(bb) - 1, 2):
                ok = True
                for b in (bb[i], bb[i + 1]):
                    x1, y1, x2, y2 = [int(round(v)) for v in b]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(m.shape[1], max(x2, x1 + 1)), min(m.shape[0], max(y2, y1 + 1))
                    if m[y1:y2, x1:x2].mean() < 0.5:
                        ok = False
                if ok:
                    keep += [i, i + 1]
            if len(keep) >= 2:
                frames.append((img.detach().cpu(), bb[keep]))
        data[sep] = frames
        npair = sum(len(f[1]) // 2 for f in frames)
        print(f"  sep={sep:3d}px: {len(frames)} frames, {npair} usable pairs", flush=True)
    return data


def run_model(name, ckpt, cfg_file, data, dev):
    model, a = build_model_from_ckpt(cfg_file, ckpt, dev)
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.05
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    if not NMS:                                   # IoU can never exceed 1.0 -> nothing suppressed
        cfg.POSTPROCESSING_NMSIOU_RING = 1.0
        cfg.POSTPROCESSING_NMSIOU_SEG = 1.0
    cache = {}
    with torch.no_grad():
        for sep, frames in data.items():
            per = []
            for img, gt in frames:
                t = img.to(dev)
                if t.dim() == 2:
                    t = t[None, None]
                elif t.dim() == 3:
                    t = t[None]
                t = t.repeat(1, a.num_channels, 1, 1)
                out = model(t)
                raw = [out['pred_logits'].detach().cpu().numpy(),
                       out['pred_boxes'].detach().cpu().numpy()]

                class _C:
                    pass
                gc = _C()
                onnx_to_xyxy(cfg, gc, raw)
                filter_boxes(cfg, gc)
                per.append((torch.tensor(gt), gc.boxes.clone(), gc.scores.clone()))
            cache[sep] = per
    del model
    torch.cuda.empty_cache()
    return cache


def localisation(cache, thr):
    """Separate the two failure modes the `merged` bucket conflates.

    For each planted pair, count RAW detections in its neighbourhood BEFORE matching:
      n_near == 1  -> the model genuinely emitted ONE box   => query/assignment capacity
      n_near >= 2  -> it emitted two, so ask WHERE they are:
                      their chi-separation ~ the true sep   => localisation is fine, matching failed
                      their chi-separation << the true sep  => both piled near the midpoint
                                                               => regression precision
    These need opposite fixes, which is why the distinction matters.
    """
    out = {}
    for sep, per in cache.items():
        n1 = n2 = n0 = 0
        dchi = []
        for gt, pred, sc in per:
            k = sc > thr
            p, s_ = pred[k], sc[k]
            if len(p):
                pq = ((p[:, 0] + p[:, 2]) / 2).numpy()
                pc = ((p[:, 1] + p[:, 3]) / 2).numpy()
            for i in range(0, len(gt) - 1, 2):
                gq = float((gt[i, 0] + gt[i, 2] + gt[i + 1, 0] + gt[i + 1, 2]) / 4)
                gc_ = float((gt[i, 1] + gt[i, 3] + gt[i + 1, 1] + gt[i + 1, 3]) / 4)
                if not len(p):
                    n0 += 1; continue
                if AXIS == 'chi':
                    near = (np.abs(pq - gq) < 15) & (np.abs(pc - gc_) < sep / 2 + 25)
                    along = pc
                else:
                    near = (np.abs(pc - gc_) < 15) & (np.abs(pq - gq) < sep / 2 + 25)
                    along = pq
                cnt = int(near.sum())
                if cnt == 0:
                    n0 += 1
                elif cnt == 1:
                    n1 += 1
                else:
                    n2 += 1
                    idx = np.nonzero(near)[0]
                    top = idx[np.argsort(-s_.numpy()[idx])[:2]]
                    dchi.append(abs(float(along[top[0]] - along[top[1]])))
        tot = n0 + n1 + n2
        out[sep] = dict(none=n0 / tot if tot else float('nan'),
                        one=n1 / tot if tot else float('nan'),
                        two=n2 / tot if tot else float('nan'),
                        med_dchi=float(np.median(dchi)) if dchi else float('nan'))
    return out


def score(cache, thr):
    matcher = get_matcher('q', min_iou=0.1)
    res = {}
    ndet = ngt = 0
    for sep, per in cache.items():
        r = m1 = m0 = 0
        for gt, pred, sc in per:
            k = sc > thr
            p = pred[k]
            ndet += len(p); ngt += len(gt)
            row = np.array([], int)
            if len(gt) and len(p):
                try:
                    _, row, _ = matcher(gt, p)
                except IndexError:
                    pass
            got = set(row.tolist())
            for i in range(0, len(gt) - 1, 2):
                n = (i in got) + (i + 1 in got)
                r += n == 2; m1 += n == 1; m0 += n == 0
        tot = r + m1 + m0
        res[sep] = dict(n=tot, resolved=r / tot if tot else float('nan'),
                        merged=m1 / tot if tot else float('nan'),
                        missed=m0 / tot if tot else float('nan'))
    return res, ndet, ngt


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  AXIS={AXIS}  ISO={ISO}  NMS={NMS}\n"
          f"stimulus: box {BOX_W:.2f} x {BOX_H:.2f} px  ->  rendered sigma_q={BOX_W / S.SimulationConfig.w_coef:.2f} "
          f"sigma_chi={BOX_H / S.SimulationConfig.a_coef:.2f} px\n"
          f"building the fixed image set (identical for every model)", flush=True)
    data = make_images(dev)

    summary = {}
    for name, ckpt, cfg_file in MODELS:
        print(f"\n########## {name} ##########", flush=True)
        cache = run_model(name, ckpt, cfg_file, data, dev)
        # matched operating point: threshold whose detection count is closest to the true peak count
        best, bestd = None, None
        for t in THRESHOLDS:
            _, nd, ng = score(cache, t)
            if best is None or abs(nd - ng) < bestd:
                best, bestd, bn, bg = t, abs(nd - ng), nd, ng
        res, nd, ng = score(cache, best)
        print(f"  matched operating point: thr={best:.2f} -> {nd} detections vs {ng} planted peaks")
        print(f"  {'sep(px)':>8s} {'pairs':>6s} {'RESOLVED':>9s} {'merged':>8s} {'missed':>8s}")
        for sep in SEPS:
            d = res[sep]
            bar = '#' * int(round(d['resolved'] * 30))
            print(f"  {sep:8d} {d['n']:6d} {d['resolved']:9.3f} {d['merged']:8.3f} "
                  f"{d['missed']:8.3f}  {bar}")
        loc = localisation(cache, best)
        print(f"\n  WHY pairs fail -- raw detections near the pair, BEFORE matching:")
        print(f"  {'sep(px)':>8s} {'no det':>8s} {'ONE box':>9s} {'TWO+ boxes':>11s} "
              f"{'median dchi of the 2':>21s} {'(true sep)':>11s}")
        for sep in SEPS:
            d = loc[sep]
            print(f"  {sep:8d} {d['none']:8.3f} {d['one']:9.3f} {d['two']:11.3f} "
                  f"{d['med_dchi']:21.1f} {sep:11d}")
        summary[name] = dict(thr=best, res={str(k): v for k, v in res.items()},
                             loc={str(k): v for k, v in loc.items()})
        del cache

    names = [m[0] for m in MODELS]
    print("\n" + "=" * 88)
    print("  RESOLVED fraction by model (higher = separates the pair)")
    print(f"  {'sep(px)':>8s}" + "".join(f"{n:>18s}" for n in names))
    for sep in SEPS:
        print(f"  {sep:8d}" + "".join(
            f"{summary[n]['res'][str(sep)]['resolved']:18.3f}" for n in names))
    print("\n  MERGED fraction (model saw the feature but emitted ONE box = resolution failure)")
    print(f"  {'sep(px)':>8s}" + "".join(f"{n:>18s}" for n in names))
    for sep in SEPS:
        print(f"  {sep:8d}" + "".join(
            f"{summary[n]['res'][str(sep)]['merged']:18.3f}" for n in names))
    # where each model crosses 50% resolved = its practical resolution limit
    print("\n  RESOLUTION LIMIT (smallest separation with RESOLVED >= 0.5):")
    for n in names:
        lim = next((s for s in SEPS if summary[n]['res'][str(s)]['resolved'] >= 0.5), None)
        print(f"    {n:18s} {'>64 px' if lim is None else str(lim) + ' px'}")
    print("\n" + "=" * 88)
    print("  DISCRIMINATOR: of pairs the model FAILED to resolve, did it emit ONE box or TWO?")
    print(f"  {'sep(px)':>8s}" + "".join(f"{n + ' 1box/2box':>22s}" for n in names))
    for sep in SEPS:
        row = f"  {sep:8d}"
        for n in names:
            d = summary[n]['loc'][str(sep)]
            row += f"{d['one']:>10.2f} /{d['two']:>10.2f}"
        print(row)
    print("\n  median chi-separation of the two detections when TWO were emitted:")
    print(f"  {'sep(px)':>8s}" + "".join(f"{n:>18s}" for n in names))
    for sep in SEPS:
        print(f"  {sep:8d}" + "".join(
            f"{summary[n]['loc'][str(sep)]['med_dchi']:18.1f}" for n in names))

    print("\n  Real-data reference (clusters_gate.py, matched count, chi-gap <5px):")
    print("    ssl1 0.352   clusters 0.321  -- if synthetic RESOLVED at 4px is far above these,")
    print("    the architecture is not the limit and a chi-resolution run would be wasted.")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, c in zip(names, ('tab:blue', 'tab:red', 'tab:green')):
        y = [summary[name]['res'][str(s)]['resolved'] for s in SEPS]
        ax.plot(SEPS, y, 'o-', color=c, label=f"{name} (thr={summary[name]['thr']:.2f})")
    ax.axvline(8, ls=':', c='k', label='stride-8 cell'); ax.axvline(4, ls=':', c='0.6', label='stride-4 cell')
    ax.axhline(0.352, ls='--', c='gray', lw=1, label='real <5px recall (ssl1)')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel(f'planted {AXIS}-separation (px)'); ax.set_ylabel('fraction of pairs RESOLVED')
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=8)
    ax.set_title(f'Phase V: separation ladder ({TAG}, sigma_q={BOX_W:.2f} sigma_chi={SIGMA:.2f})')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'separation_ladder_{TAG}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nsaved {out}")
    json.dump(summary, open(os.path.join(OUT, f'separation_ladder_{TAG}.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
