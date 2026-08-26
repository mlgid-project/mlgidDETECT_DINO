"""Phase AC — the boxes are too tall in chi. Where does that come from?

AB measured, on organic, that the model's boxes are 2.6x taller in chi than the labelled peaks --
and not only for false positives: the TRUE positives are inflated too (TP p50 20.7 px vs GT segment
p50 8.1 px). The deployed matcher runs at IoU 0.1, loose enough that an over-tall box still counts
as a hit, so the defect is invisible in recall and precision and shows up only here. It matters
twice over: an inflated box is a bad starting point for peak fitting, and two peaks 5 px apart get
boxes that overlap enough for duplicate suppression to delete one of them -- which is the exact
shape of the recall hole phases V-AA could not close.

The model emits what it was trained on, so the question is what the SIMULATOR hands it.

Read out of the code first, so the measurement has something to be checked against:
  simulation.py:489-492   box = a_pos +- a_widths * a_coef   (a_coef = 3.5)
  simulation.py:821-822   sigma_chi = box_height / a_coef,  sigma_q = box_width / w_coef (w_coef = 1)
  =>  box_height = 3.5 * sigma_chi  (+-1.75 sigma)  and  box_width = 1.0 * sigma_q  (+-0.5 sigma).
  For a Gaussian, FWHM = 2.355 sigma, so the simulator's own convention predicts
      box_h / FWHM_chi = 1.486     box_w / FWHM_q = 0.425
  The probe measures those two on simulated frames as a SELF-CHECK on the width estimator. If the
  measured sim ratios land on the predicted ones, the estimator works and the numbers it gives for
  organic and 41 -- whose labelling convention nobody has written down -- can be believed.

That separates the two possible causes, which need different fixes:
  CONVENTION  sim and real draw boxes at different multiples of the same peak width. Fix = change
              a_coef / w_coef. One constant, no retraining of anything else.
  PHYSICS     the simulated peaks really are broader in chi than real ones. Fix = the a_width
              sampling distribution, a bigger change.

FOUR blocks:
  1  segments-only box height and width, organic / 41 / sim clusters OFF / sim clusters ON.
     Rings excluded -- AA.4 showed they dominate the marginal and are what made 42.4 px look
     comparable to 8.5 px.
  2  measured peak widths (FWHM in chi and in q) for the same four sets, and the box-over-FWHM
     ratios, which is the convention question stated as a number.
  3  per-matched-pair pred/GT size ratio on organic and 41 -- how much the model actually inflates,
     pair by pair rather than as a difference of medians.
  4  matched-pair IoU, and what it becomes if predicted boxes are shrunk in chi by a single global
     factor. Says whether shrinking is free or costs matches.

Single model ssl1, score > 0.3, deployed postprocessing, matcher q at IoU 0.1 -- identical to AB.
Intensity caveat: sim images come out of apply_log/apply_he/apply_clip/kernel/normalize and real
images out of contrast_correction; both are log-then-equalise but not byte-identical, so an FWHM
measured on one is not exactly an FWHM measured on the other. The estimator self-check bounds how
much that matters.

GPU, ~10 min. See tmp_diag/run_boxsize.sbatch.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from simulation import FastSimulation
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG, ST
from diagnostics.ring_count_fix import classify

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM = 200
PCTS = [10, 25, 50, 75, 90]
PAD = 2.0          # window around a box, in box heights, used to find the local baseline
SHRINK = [1.0, 0.75, 0.5, 0.4, 0.3]


def pcts(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return {p: float(np.percentile(v, p)) for p in PCTS} if len(v) else {p: float('nan') for p in PCTS}


def _fwhm_1d(prof):
    """Full width at half maximum above a local baseline, in samples. NaN if no usable peak."""
    prof = np.asarray(prof, dtype=float)
    if len(prof) < 5 or not np.isfinite(prof).any():
        return float('nan')
    base = np.nanpercentile(prof, 10)
    pk = np.nanmax(prof)
    if not np.isfinite(pk) or not np.isfinite(base) or pk - base <= 1e-9:
        return float('nan')
    half = base + 0.5 * (pk - base)
    i = int(np.nanargmax(prof))
    lo = i
    while lo > 0 and np.isfinite(prof[lo - 1]) and prof[lo - 1] >= half:
        lo -= 1
    hi = i
    while hi < len(prof) - 1 and np.isfinite(prof[hi + 1]) and prof[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == len(prof) - 1:
        return float('nan')          # clipped by the window -- not a measurement
    return float(hi - lo + 1)


def box_widths(img, b):
    """(FWHM in chi, FWHM in q) for one box, measured on the image inside a padded window."""
    H, W = img.shape
    q0f, q1f = float(min(b[0], b[2])), float(max(b[0], b[2]))
    c0f, c1f = float(min(b[1], b[3])), float(max(b[1], b[3]))
    bh, bw = max(c1f - c0f, 1.0), max(q1f - q0f, 1.0)

    q0 = int(np.clip(round(q0f), 0, W - 1)); q1 = int(np.clip(round(q1f), q0 + 1, W))
    c0 = int(np.clip(round(c0f), 0, H - 1)); c1 = int(np.clip(round(c1f), c0 + 1, H))

    r0 = int(np.clip(round(c0f - PAD * bh), 0, H - 1)); r1 = int(np.clip(round(c1f + PAD * bh), r0 + 1, H))
    s0 = int(np.clip(round(q0f - PAD * bw), 0, W - 1)); s1 = int(np.clip(round(q1f + PAD * bw), s0 + 1, W))

    with np.errstate(invalid='ignore'):
        wc = img[r0:r1, q0:q1]
        wq = img[c0:c1, s0:s1]
        pc = np.nanmean(np.where(wc > 0, wc, np.nan), axis=1)   # collapse q -> profile along chi
        pq = np.nanmean(np.where(wq > 0, wq, np.nan), axis=0)   # collapse chi -> profile along q
    return _fwhm_1d(pc), _fwhm_1d(pq)


def collect(img, boxes, is_ring):
    """Per-segment box height, box width, and the two measured FWHMs."""
    h = np.abs(boxes[:, 3] - boxes[:, 1]); w = np.abs(boxes[:, 2] - boxes[:, 0])
    out = dict(h=[], w=[], fc=[], fq=[])
    for i in range(len(boxes)):
        if is_ring[i]:
            continue
        fc, fq = box_widths(img, boxes[i])
        out['h'].append(h[i]); out['w'].append(w[i]); out['fc'].append(fc); out['fq'].append(fq)
    return out


def merge(dst, src):
    for k in dst:
        dst[k] += src[k]


def summarise(name, acc, frames, nseg):
    h = np.asarray(acc['h']); w = np.asarray(acc['w'])
    fc = np.asarray(acc['fc']); fq = np.asarray(acc['fq'])
    ok_c = np.isfinite(fc) & (fc > 0); ok_q = np.isfinite(fq) & (fq > 0)
    return dict(name=name, frames=frames, segs=nseg,
                h=pcts(h), w=pcts(w), fc=pcts(fc[ok_c]), fq=pcts(fq[ok_q]),
                r_c=pcts(h[ok_c] / fc[ok_c]), r_q=pcts(w[ok_q] / fq[ok_q]),
                n_c=int(ok_c.sum()), n_q=int(ok_q.sum()))


def open_ds(path):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                               buffer_size=5))
    return cfg, ds


def iou_xyxy(a, b):
    x0 = np.maximum(a[:, 0], b[:, 0]); y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2]); y1 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = np.abs(a[:, 2] - a[:, 0]) * np.abs(a[:, 3] - a[:, 1])
    bb = np.abs(b[:, 2] - b[:, 0]) * np.abs(b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + bb - inter, 1e-9)


def shrink_chi(p, f):
    cy = (p[:, 1] + p[:, 3]) / 2; hh = np.abs(p[:, 3] - p[:, 1]) / 2 * f
    o = p.copy(); o[:, 1] = cy - hh; o[:, 3] = cy + hh
    return o


def do_real(name, path, model, a, dev):
    cfg, ds = open_ds(path)
    matcher = get_matcher('q', min_iou=0.1)
    acc = dict(h=[], w=[], fc=[], fq=[])
    n_fr = nseg = 0
    M = dict(rh=[], rw=[], gh=[], gw=[], ph=[], pw=[], iou=[], seg=[])
    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]
            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            if not len(b):
                continue
            _h, _sp, is_ring = classify(img_np, b)
            n_fr += 1; nseg += int((~is_ring).sum())
            merge(acc, collect(img_np, b, is_ring))

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, a.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
            gc2 = filter_boxes(cfg, onnx_to_xyxy(cfg, gc, raw))
            pred, sc = gc2.boxes, gc2.scores
            keep = (sc > ST).cpu().numpy() if hasattr(sc, 'cpu') else (sc > ST)
            pred = pred[keep]
            P = pred.numpy() if hasattr(pred, 'numpy') else np.asarray(pred)
            if not len(P):
                continue
            try:
                _, row, col = matcher(torch.tensor(b, dtype=torch.float32), pred)
            except IndexError:
                continue
            row = np.asarray(row, dtype=int); col = np.asarray(col, dtype=int)
            if not len(row):
                continue
            G = b[row]; Q = P[col].astype(np.float64)
            gh = np.abs(G[:, 3] - G[:, 1]); gw = np.abs(G[:, 2] - G[:, 0])
            ph = np.abs(Q[:, 3] - Q[:, 1]); pw = np.abs(Q[:, 2] - Q[:, 0])
            M['gh'] += list(gh); M['gw'] += list(gw); M['ph'] += list(ph); M['pw'] += list(pw)
            M['rh'] += list(ph / np.maximum(gh, 1e-6)); M['rw'] += list(pw / np.maximum(gw, 1e-6))
            M['iou'] += list(iou_xyxy(Q, G))
            M['seg'] += list(~is_ring[row])
    if hasattr(ds, 'close'):
        ds.close()

    sh = {}
    for f in SHRINK:
        Q = np.zeros((len(M['ph']), 4))
        Q[:, 0] = 0; Q[:, 2] = np.asarray(M['pw']); Q[:, 1] = -np.asarray(M['ph']) * f / 2
        Q[:, 3] = np.asarray(M['ph']) * f / 2
        Gb = np.zeros_like(Q)
        Gb[:, 0] = (np.asarray(M['pw']) - np.asarray(M['gw'])) / 2
        Gb[:, 2] = Gb[:, 0] + np.asarray(M['gw'])
        Gb[:, 1] = -np.asarray(M['gh']) / 2; Gb[:, 3] = np.asarray(M['gh']) / 2
        v = iou_xyxy(Q, Gb)
        sh[f] = dict(med=float(np.median(v)), ge10=float(np.mean(v >= 0.1)),
                     ge30=float(np.mean(v >= 0.3)), ge50=float(np.mean(v >= 0.5)))

    segm = np.asarray(M['seg'], dtype=bool)
    rh = np.asarray(M['rh']); rw = np.asarray(M['rw'])
    return (summarise(name, acc, n_fr, nseg),
            dict(name=name, n=len(rh), n_seg=int(segm.sum()),
                 rh=pcts(rh), rw=pcts(rw), rh_seg=pcts(rh[segm]), rw_seg=pcts(rw[segm]),
                 rh_ring=pcts(rh[~segm]), iou=pcts(M['iou']), shrink=sh))


def do_sim(clusters, n, dev):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    acc = dict(h=[], w=[], fc=[], fq=[])
    n_fr = nseg = 0
    for k in range(n):
        sd = 90000 + k
        random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
        try:
            img, bx, _m, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy().astype(np.float64)
        if not len(b):
            continue
        im = img.detach().cpu().numpy()
        im = im[0, 0] if im.ndim == 4 else (im[0] if im.ndim == 3 else im)
        r = isr.detach().cpu().numpy().astype(bool)
        n_fr += 1; nseg += int((~r).sum())
        merge(acc, collect(im, b, r))
    return summarise(f"sim clusters {'ON' if clusters else 'OFF'}", acc, n_fr, nseg)


def prow(label, d, key):
    print(f"  {label:<24s}" + "".join(f"{d[key].get(p, float('nan')):10.2f}" for p in PCTS))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("###### phase AC: box height vs peak width ######", flush=True)
    print(f"device={dev}  SINGLE MODEL ssl1  PAD={PAD}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    model.eval()

    real, match = [], []
    for nm, pth in DSETS:
        s, m = do_real(nm, pth, model, a, dev)
        real.append(s); match.append(m)
    sims = [do_sim(False, N_SIM, dev), do_sim(True, N_SIM, dev)]
    rows = real + sims
    hdr = "".join(f"{('p' + str(p)):>10s}" for p in PCTS)

    print("\n" + "=" * 100)
    print("  1  SEGMENT BOX SIZE (px)  — rings excluded from every row")
    print("=" * 100)
    print(f"  {'set':<24s}{'frames':>8s}{'segments':>10s}")
    for r in rows:
        print(f"  {r['name']:<24s}{r['frames']:8d}{r['segs']:10d}")
    print(f"\n  box HEIGHT in chi        {hdr}")
    for r in rows:
        prow(r['name'], r, 'h')
    print(f"\n  box WIDTH in q           {hdr}")
    for r in rows:
        prow(r['name'], r, 'w')

    print("\n" + "=" * 100)
    print("  2  MEASURED PEAK WIDTH (FWHM, px) and BOX / FWHM  — the convention question")
    print("=" * 100)
    print(f"  simulator's own convention, read from the code: box_h/FWHM_chi = 1.486, "
          f"box_w/FWHM_q = 0.425")
    print(f"\n  FWHM in chi              {hdr}{'usable':>10s}")
    for r in rows:
        print(f"  {r['name']:<24s}" + "".join(f"{r['fc'].get(p, float('nan')):10.2f}" for p in PCTS)
              + f"{r['n_c']:10d}")
    print(f"\n  FWHM in q                {hdr}{'usable':>10s}")
    for r in rows:
        print(f"  {r['name']:<24s}" + "".join(f"{r['fq'].get(p, float('nan')):10.2f}" for p in PCTS)
              + f"{r['n_q']:10d}")
    print(f"\n  box_h / FWHM_chi         {hdr}")
    for r in rows:
        prow(r['name'], r, 'r_c')
    print(f"\n  box_w / FWHM_q           {hdr}")
    for r in rows:
        prow(r['name'], r, 'r_q')

    print("\n" + "=" * 100)
    print("  3  PREDICTED / GROUND-TRUTH SIZE, per matched pair  (model vs the gate it is graded on)")
    print("=" * 100)
    print(f"  {'gate':<24s}{'pairs':>7s}{'segs':>7s}")
    for m in match:
        print(f"  {m['name']:<24s}{m['n']:7d}{m['n_seg']:7d}")
    for key, lab in [('rh', 'pred_h / gt_h  (all)'), ('rh_seg', 'pred_h / gt_h  (segments)'),
                     ('rh_ring', 'pred_h / gt_h  (rings)'), ('rw_seg', 'pred_w / gt_w  (segments)')]:
        print(f"\n  {lab:<24s}{hdr}")
        for m in match:
            prow(m['name'], m, key)

    print("\n" + "=" * 100)
    print("  4  MATCHED-PAIR IoU, and what a global chi-shrink does to it")
    print("=" * 100)
    print(f"  matched IoU as deployed  {hdr}")
    for m in match:
        prow(m['name'], m, 'iou')
    print(f"\n  {'gate':<24s}{'shrink':>8s}{'IoU med':>10s}{'>=0.1':>8s}{'>=0.3':>8s}{'>=0.5':>8s}")
    for m in match:
        for f in SHRINK:
            s = m['shrink'][f]
            print(f"  {m['name']:<24s}{f:8.2f}{s['med']:10.3f}{s['ge10']:8.3f}"
                  f"{s['ge30']:8.3f}{s['ge50']:8.3f}")
    print("  (block 4 aligns each pair at its centre, so it isolates SIZE from position error)")

    json.dump(dict(sets=rows, match=match),
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/box_size.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
