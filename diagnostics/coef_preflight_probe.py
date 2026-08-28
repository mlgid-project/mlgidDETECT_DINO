"""Phase AE pre-flight — is changing `a_coef`/`w_coef` really ONE variable?

The claim that motivated AC.4/AC.5 is that `a_coef`/`w_coef` CANCEL between box construction and
rendering, so changing them relabels the same image:
    simulation.py:510-513   box = pos +- widths*w_coef ,  a_pos +- a_widths*a_coef
    simulation.py:842-843   sigma_q = box_w / w_coef ,  sigma_chi = box_h / a_coef
That is exact for a peak that survives to `img_from_labels` untouched. It is NOT exact overall,
because three things in between read the BOX rather than the widths:

  1  filter_peaks_detector_gap (:435, :464, :492) rasterises the box and DROPS the peak if it
     touches a detector gap. A wider box touches more often -> w_coef 1.0 -> 1.30 drops MORE peaks.
  2  filter_dark_area (:438) drops any box whose clamped extent is <= 1.6 px on either axis.
     a_coef 3.5 -> 2.98 shortens every box in chi -> the smallest ones can fall through.
  3  clamp_boxes (:489-493, and inside filter_dark_area) clips boxes to the image, and
     `img_from_labels` then derives sigma from the CLIPPED box -- so an edge peak is rendered
     narrower. Change the coefficients and a different set of peaks gets clipped.

So the honest question is not "is it exactly one variable" (it is not) but "how far from one
variable is it". This probe answers that by simulating the SAME SEEDS under both coefficient
settings and comparing:

  * objects / segments / rings per frame, and the ring:segment ratio  -- class balance drift
  * the fraction of frames where both settings keep the SAME object count -- the filters' footprint
  * box height and width percentiles, against the exact ratios (0.85 in chi, 1.30 in q) that a pure
    relabelling would produce; any deviation is the filters selecting a different population
  * the rendered image difference, restricted to frames where the counts match (once the counts
    diverge the RNG streams diverge too and the images are no longer comparable)

If counts match on nearly every frame and the box ratios land on 0.85/1.30, the change is a
relabelling for practical purposes and the retrain is single-variable. If not, the size of the
drift is what has to be reported alongside the run.

GPU, ~5 min. See tmp_diag/run_coefpre.sbatch.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from simulation import FastSimulation

N = int(os.environ.get('COEFPRE_N', 300))
A_NEW = float(os.environ.get('COEFPRE_A', 2.98))
W_NEW = float(os.environ.get('COEFPRE_W', 1.30))
A_OLD, W_OLD = 3.5, 1.0
PCTS = [10, 25, 50, 75, 90]


def run(a_coef, w_coef, n, dev):
    sim = FastSimulation(device=dev)
    sim.sim_config.a_coef = a_coef
    sim.sim_config.w_coef = w_coef
    per, imgs, hs, ws, nseg, nring, errs = [], {}, [], [], 0, 0, 0
    for k in range(n):
        sd = 90000 + k
        random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
        try:
            img, boxes, _m, isr = sim.simulate_img()
        except Exception as e:
            if not errs:
                print(f"  frame {k} raised {type(e).__name__}: {e}", flush=True)
            errs += 1; per.append(None); continue
        b = boxes.detach().cpu().numpy().astype(np.float64)
        r = isr.detach().cpu().numpy().astype(bool)
        per.append((len(b), int((~r).sum()), int(r.sum())))
        nseg += int((~r).sum()); nring += int(r.sum())
        s = b[~r]
        if len(s):
            hs += list(np.abs(s[:, 3] - s[:, 1])); ws += list(np.abs(s[:, 2] - s[:, 0]))
        if k < 60:
            imgs[k] = img.detach().cpu().numpy().astype(np.float32).squeeze()
    return dict(per=per, imgs=imgs, h=np.asarray(hs), w=np.asarray(ws),
                nseg=nseg, nring=nring, errs=errs, frames=n)


def pct(v):
    return [float(np.percentile(v, p)) for p in PCTS] if len(v) else [float('nan')] * len(PCTS)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  N={N}  OLD=({A_OLD}, {W_OLD})  NEW=({A_NEW}, {W_NEW})", flush=True)
    print(f"a_coef ratio {A_NEW / A_OLD:.4f}   w_coef ratio {W_NEW / W_OLD:.4f}", flush=True)

    old = run(A_OLD, W_OLD, N, dev); print("  old done", flush=True)
    new = run(A_NEW, W_NEW, N, dev); print("  new done", flush=True)

    same = tot = 0
    dseg = []
    for a, b in zip(old['per'], new['per']):
        if a is None or b is None:
            continue
        tot += 1
        if a[0] == b[0]:
            same += 1
        dseg.append(b[1] - a[1])

    print("\n" + "=" * 96)
    print("  1  OBJECT COUNTS AND CLASS BALANCE")
    print("=" * 96)
    for nm, r in [('old (3.50, 1.00)', old), (f'new ({A_NEW:.2f}, {W_NEW:.2f})', new)]:
        f = r['frames'] - r['errs']
        print(f"  {nm:<22s} frames_ok {f:4d}  raised {r['errs']:3d}"
              f"  seg/frame {r['nseg'] / max(f,1):7.3f}  ring/frame {r['nring'] / max(f,1):7.3f}"
              f"  ring:seg {r['nring'] / max(r['nseg'],1):7.4f}")
    print(f"\n  frames with IDENTICAL object count: {same}/{tot} = {same / max(tot,1):.4f}")
    d = np.asarray(dseg, dtype=float)
    print(f"  segment-count delta per frame (new - old): mean {d.mean():+.4f}  "
          f"min {d.min():+.0f}  max {d.max():+.0f}  nonzero {(d != 0).mean():.4f}")

    print("\n" + "=" * 96)
    print("  2  SEGMENT BOX SIZE — measured ratio vs the pure-relabelling prediction")
    print("=" * 96)
    hdr = "".join(f"{('p' + str(p)):>10s}" for p in PCTS)
    for lab, ko, kn, exp in [('box HEIGHT (chi)', old['h'], new['h'], A_NEW / A_OLD),
                             ('box WIDTH  (q)', old['w'], new['w'], W_NEW / W_OLD)]:
        po, pn = pct(ko), pct(kn)
        print(f"\n  {lab}   expected ratio {exp:.4f}")
        print(f"  {'old':<14s}{hdr}");  print(f"  {'':<14s}" + "".join(f"{v:10.2f}" for v in po))
        print(f"  {'new':<14s}{hdr}");  print(f"  {'':<14s}" + "".join(f"{v:10.2f}" for v in pn))
        print(f"  {'ratio':<14s}      " + "".join(
            f"{(b / a if a else float('nan')):10.4f}" for a, b in zip(po, pn)))

    print("\n" + "=" * 96)
    print("  3  RENDERED IMAGE DIFFERENCE  (frames with matching object counts only)")
    print("=" * 96)
    rel, nimg = [], 0
    for k in old['imgs']:
        if k not in new['imgs']:
            continue
        a, b = old['per'][k], new['per'][k]
        if a is None or b is None or a[0] != b[0]:
            continue
        x, y = old['imgs'][k], new['imgs'][k]
        if x.shape != y.shape:
            continue
        nimg += 1
        rel.append(float(np.abs(x - y).mean() / max(x.std(), 1e-9)))
    if rel:
        rel = np.asarray(rel)
        print(f"  frames compared {nimg}   mean|dI| / std(I):  mean {rel.mean():.5f}"
              f"  p50 {np.median(rel):.5f}  max {rel.max():.5f}"
              f"  exactly zero {(rel == 0).mean():.3f}")
    else:
        print("  no comparable frames")

    json.dump(dict(same=same, tot=tot, a_new=A_NEW, w_new=W_NEW,
                   old=dict(nseg=old['nseg'], nring=old['nring'], errs=old['errs']),
                   new=dict(nseg=new['nseg'], nring=new['nring'], errs=new['errs'])),
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/coef_preflight.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
