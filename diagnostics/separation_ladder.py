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

GPU, ~20 min. See tmp_diag/run_separation_ladder.sbatch.
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
SEPS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]      # chi-separation rungs (px)
N_IMG = 20                                        # frames per rung
N_PAIR = 8                                        # pairs per frame (well separated in q)
BOX_W, BOX_H = 10.6, 8.5                          # measured real organic GT medians
INTENS = 30.0                                     # mid-range of seg_intensity_range (10, 50)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def plant(sep, rng):
    """Two peaks at chi-separation `sep`, N_PAIR times, spread across q. Returns boxes (2N,4)."""
    H, W = S.HEIGHT, S.WIDTH
    qs = np.linspace(0.15 * W, 0.90 * W, N_PAIR) + rng.uniform(-12, 12, N_PAIR)
    # keep the whole pair clear of the angle-limit bands at top/bottom
    lo, hi = 0.22 * H, 0.78 * H
    cs = rng.uniform(lo + sep / 2, hi - sep / 2, N_PAIR)
    boxes = []
    for q, c in zip(qs, cs):
        for s in (-sep / 2, +sep / 2):
            boxes.append([q - BOX_W / 2, c + s - BOX_H / 2, q + BOX_W / 2, c + s + BOX_H / 2])
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
            random.seed(9000 + sep * 100 + k); torch.manual_seed(9000 + sep * 100 + k)
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
    print(f"device={dev}\nbuilding the fixed image set (identical for every model)", flush=True)
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
        summary[name] = dict(thr=best, res={str(k): v for k, v in res.items()})
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
    ax.set_xlabel('planted chi-separation (px)'); ax.set_ylabel('fraction of pairs RESOLVED')
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=8)
    ax.set_title('Phase V: synthetic separation ladder')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'separation_ladder.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nsaved {out}")
    json.dump(summary, open(os.path.join(OUT, 'separation_ladder.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
