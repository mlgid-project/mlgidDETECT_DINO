"""Phase AA.3 — every number on BOTH eval gates, plus the simulated training distribution.

The close-pair investigation (V-AA) reported almost exclusively on organic. 41.h5 (perovskite) is the
second gate and the uncontaminated one, and the precision axis now under consideration has never been
characterised on either. This consolidates them side by side, with the simulated training
distribution as the third column, so composition mismatches are visible rather than inferred.

WHY THIS RUN EXISTS. Phase AA.2 measured the peak-on-ring dose and found the generator's own
`add_peaks_on_rings` yields peaks in only **0.7% of frames** (it declines on 90% by its first line,
then `max_a_width < 100` and a `randint(0, 4)` draw remove most of the rest). It also turned up an
unlooked-for composition gap on organic: simulated frames carry 29.7 segments and 15.6 rings, real
ones 99.1 and 3.0 — a ring:segment ratio off by 17x. Both findings need the 41 side before they can
carry a training run.

It also FIXES a criterion error from AA.2. "On-ring" in the phase-Q false-positive analysis never
meant "inside an annotated ring box" — it meant a detection at the SAME RADIUS as a real peak where
the azimuthally-integrated intensity is high. Measuring segments inside labelled ring boxes returned
0.00% on real organic, which is uninformative rather than evidence. The structural statistic used
here instead is **the share of labelled peaks that share their radius (within 8 px in q) with another
labelled object** — non-circular, identical on all three columns, and exactly what an on-ring false
positive is: right radius, wrong angle.

REPORTED PER GATE (single model ssl1, score>0.3, deployed postprocessing):
  recall, precision, TP/FP; segments and rings per frame; the share of peaks with a same-radius
  neighbour; the GT-to-GT χ-gap distribution; recall stratified by χ-gap; and for the false
  positives — the on-ring share, their q-distance, and their χ-distance to the nearest same-radius
  peak, with the high-confidence subset split out.

REPORTED FOR THE SIMULATOR (clusters off = the ssl1 training config, and on): the same composition and
structure statistics, plus what `add_peaks_on_rings` actually produced.

SINGLE MODEL ssl1 throughout — not the deployed ensemble. GPU, ~12 min.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

import simulation as S
from simulation import FastSimulation
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG, ST, HI, \
    ONRING_PX, HIQ_PCT

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM = 300
QTOL = 8.0
BINS = [(0, 5), (5, 10), (10, 20), (20, 33), (33, 1e9)]


def q_c(b):
    return (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2


def sameq_stats(b):
    """For every object: distance in χ to its nearest same-radius neighbour (inf if it has none)."""
    if len(b) < 2:
        return np.full(len(b), np.inf)
    q, c = q_c(b)
    out = np.full(len(b), np.inf)
    for i in range(len(b)):
        m = np.abs(q - q[i]) < QTOL
        m[i] = False
        if m.any():
            out[i] = np.min(np.abs(c[m] - c[i]))
    return out


def summarise_gaps(g):
    fin = g[np.isfinite(g)]
    if not len(fin):
        return dict(has_neighbour=0.0, med=float('nan'), lt5=0.0, lt10=0.0, lt20=0.0)
    return dict(has_neighbour=float(np.isfinite(g).mean()), med=float(np.median(fin)),
                lt5=float((fin < 5).mean()), lt10=float((fin < 10).mean()),
                lt20=float((fin < 20).mean()))


def run_gate(name, path, model, a, dev):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = 0.1
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                               buffer_size=5))
    matcher = get_matcher('q', min_iou=0.1)
    n_fr = n_gt = n_seg = n_ring = tp = fp_on = fp_off = 0
    gaps, fp_q, fp_chi, fp_chi_hi, on_chi = [], [], [], [], []
    bucket_n = {b: 0 for b in BINS}
    bucket_hit = {b: 0 for b in BINS}
    seg_in_ring = 0

    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]
            valid = img_np > 1e-6
            den = valid.sum(0); den[den == 0] = 1
            Iq = (img_np * valid).sum(0) / den
            Iq_pct = np.argsort(np.argsort(Iq)) / len(Iq)

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, a.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
            gc2 = filter_boxes(cfg, onnx_to_xyxy(cfg, gc, raw))
            pred, sc = gc2.boxes, gc2.scores
            keep = sc > ST; pred, sc = pred[keep], sc[keep]

            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            if not len(b):
                continue
            n_fr += 1; n_gt += len(b)
            isr = np.array(list(L.is_ring), dtype=bool) if getattr(L, 'is_ring', None) is not None \
                and len(list(L.is_ring)) == len(b) else np.zeros(len(b), bool)
            n_seg += int((~isr).sum()); n_ring += int(isr.sum())

            gq, gcc = q_c(b)
            if isr.any() and (~isr).any():
                rb = b[isr]
                sq, scc = gq[~isr], gcc[~isr]
                hit = np.zeros(len(sq), bool)
                for r in rb:
                    hit |= (sq >= r[0]) & (sq <= r[2]) & (scc >= r[1]) & (scc <= r[3])
                seg_in_ring += int(hit.sum())

            g = sameq_stats(b)
            gaps.append(g)

            gt_t = torch.tensor(b, dtype=torch.float32)
            row = np.array([], int); col = np.array([], int)
            if len(pred):
                try:
                    _, row, col = matcher(gt_t, pred)
                except IndexError:
                    pass
            hitset = set(row.tolist()); cset = set(col.tolist())
            tp += len(cset)
            for i in range(len(b)):
                gi = g[i] if np.isfinite(g[i]) else 1e9
                for bk in BINS:
                    if bk[0] <= gi < bk[1]:
                        bucket_n[bk] += 1
                        if i in hitset:
                            bucket_hit[bk] += 1
                        break
            for j in range(len(pred)):
                if j in cset:
                    continue
                q = float((pred[j, 0] + pred[j, 2]) / 2)
                c = float((pred[j, 1] + pred[j, 3]) / 2)
                qd = float(np.min(np.abs(gq - q)))
                m = np.abs(gq - q) < QTOL
                cd = float(np.min(np.abs(gcc[m] - c))) if m.any() else np.inf
                fp_q.append(qd); fp_chi.append(cd)
                onring = (qd < ONRING_PX) and (Iq_pct[int(np.clip(q, 0, 1023))] > HIQ_PCT)
                if onring:
                    fp_on += 1; on_chi.append(cd)
                else:
                    fp_off += 1
                if float(sc[j]) > HI:
                    fp_chi_hi.append(cd)
    if hasattr(ds, 'close'):
        ds.close()

    fp = fp_on + fp_off
    g_all = np.concatenate(gaps) if gaps else np.zeros(0)
    return dict(name=name, frames=n_fr, gt=n_gt,
                seg_per_frame=n_seg / max(n_fr, 1), ring_per_frame=n_ring / max(n_fr, 1),
                ring_seg_ratio=n_ring / max(n_seg, 1),
                seg_in_ring_frac=seg_in_ring / max(n_seg, 1),
                recall=len(set()) if False else None,
                tp=tp, fp=fp, fp_on=fp_on, fp_off=fp_off,
                precision=tp / max(tp + fp, 1),
                struct=summarise_gaps(g_all),
                buckets={f"{b[0]}-{b[1] if b[1] < 1e8 else 'inf'}":
                         dict(n=bucket_n[b], recall=bucket_hit[b] / max(bucket_n[b], 1))
                         for b in BINS},
                fp_q_med=float(np.median(fp_q)) if fp_q else float('nan'),
                fp_chi=summarise_gaps(np.asarray(fp_chi)) if fp_chi else None,
                fp_chi_on=summarise_gaps(np.asarray(on_chi)) if on_chi else None,
                fp_chi_hi=summarise_gaps(np.asarray(fp_chi_hi)) if fp_chi_hi else None,
                recall_total=sum(bucket_hit.values()) / max(sum(bucket_n.values()), 1))


def run_sim(dev, clusters, n):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    _orig = FastSimulation.add_peaks_on_rings
    rec = {}

    def wrapped(self, x_position, widths, boxes, ring_intensities):
        out = _orig(self, x_position, widths, boxes, ring_intensities)
        rec['n'] = 0 if out[0] is None else int(len(out[0]))
        return out
    FastSimulation.add_peaks_on_rings = wrapped

    n_fr = n_seg = n_ring = tagged = fr_tag = 0
    gaps = []
    for k in range(n):
        _sd = 90000 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        rec['n'] = 0
        try:
            _img, bx, _m, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy().astype(np.float64)
        r = isr.detach().cpu().numpy().astype(bool)
        if not len(b):
            continue
        n_fr += 1
        n_seg += int((~r).sum()); n_ring += int(r.sum())
        tagged += rec['n']; fr_tag += int(rec['n'] > 0)
        gaps.append(sameq_stats(b))
    FastSimulation.add_peaks_on_rings = _orig
    g = np.concatenate(gaps) if gaps else np.zeros(0)
    return dict(name=f"sim clusters {'ON' if clusters else 'OFF'}", frames=n_fr,
                seg_per_frame=n_seg / max(n_fr, 1), ring_per_frame=n_ring / max(n_fr, 1),
                ring_seg_ratio=n_ring / max(n_seg, 1),
                struct=summarise_gaps(g),
                tagged_onring=tagged, tagged_frac_of_seg=tagged / max(n_seg, 1),
                frames_with_tagged=fr_tag / max(n_fr, 1))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    gates = [run_gate(n, p, model, a, dev) for n, p in DSETS]
    del model; torch.cuda.empty_cache()
    sims = [run_sim(dev, False, N_SIM), run_sim(dev, True, N_SIM)]

    print("\n" + "=" * 104)
    print("  DETECTION — single model ssl1, score>0.3, deployed postprocessing")
    print("=" * 104)
    print(f"  {'gate':<18s}{'frames':>7s}{'GT':>7s}{'TP':>6s}{'FP':>6s}"
          f"{'recall':>9s}{'precision':>11s}{'FP on-ring':>12s}")
    for g in gates:
        print(f"  {g['name']:<18s}{g['frames']:7d}{g['gt']:7d}{g['tp']:6d}{g['fp']:6d}"
              f"{g['recall_total']:9.3f}{g['precision']:11.3f}"
              f"{g['fp_on'] / max(g['fp'], 1):12.3f}")

    print("\n" + "=" * 104)
    print("  COMPOSITION — what a frame contains")
    print("=" * 104)
    print(f"  {'set':<18s}{'segs/frame':>12s}{'rings/frame':>13s}{'ring:seg':>10s}"
          f"{'segs in ring box':>18s}")
    for g in gates:
        print(f"  {g['name']:<18s}{g['seg_per_frame']:12.1f}{g['ring_per_frame']:13.1f}"
              f"{g['ring_seg_ratio']:10.3f}{g['seg_in_ring_frac']:18.3f}")
    for s in sims:
        print(f"  {s['name']:<18s}{s['seg_per_frame']:12.1f}{s['ring_per_frame']:13.1f}"
              f"{s['ring_seg_ratio']:10.3f}{'-':>18s}")

    print("\n" + "=" * 104)
    print("  STRUCTURE — share of labelled objects sharing a RADIUS (<8 px in q) with another")
    print("=" * 104)
    print(f"  {'set':<18s}{'has neighbour':>15s}{'median gap':>12s}{'<5px':>8s}{'<10px':>8s}{'<20px':>8s}")
    for x in gates + sims:
        st = x['struct']
        print(f"  {x['name']:<18s}{st['has_neighbour']:15.3f}{st['med']:12.1f}"
              f"{st['lt5']:8.3f}{st['lt10']:8.3f}{st['lt20']:8.3f}")
    print("\n  peaks-on-rings the generator actually produced:")
    for s in sims:
        print(f"    {s['name']:<16s} {s['tagged_onring']} tagged = "
              f"{100 * s['tagged_frac_of_seg']:.2f}% of segments, in "
              f"{100 * s['frames_with_tagged']:.1f}% of frames")

    print("\n" + "=" * 104)
    print("  RECALL BY χ-GAP TO NEAREST SAME-RADIUS NEIGHBOUR")
    print("=" * 104)
    keys = list(gates[0]['buckets'].keys())
    print(f"  {'gate':<18s}" + "".join(f"{k:>16s}" for k in keys))
    for g in gates:
        print(f"  {g['name']:<18s}" +
              "".join(f"{g['buckets'][k]['recall']:8.3f} (n{g['buckets'][k]['n']:>4d})"
                      for k in keys))

    print("\n" + "=" * 104)
    print("  FALSE POSITIVES — distance to the nearest labelled peak")
    print("=" * 104)
    print(f"  {'gate':<18s}{'q-dist med':>12s}{'χ-dist med':>12s}{'χ<10px':>9s}"
          f"{'on-ring χ med':>15s}{'hi-conf χ med':>15s}")
    for g in gates:
        f_all, f_on, f_hi = g['fp_chi'], g['fp_chi_on'], g['fp_chi_hi']
        print(f"  {g['name']:<18s}{g['fp_q_med']:12.1f}"
              f"{(f_all['med'] if f_all else float('nan')):12.1f}"
              f"{(f_all['lt10'] if f_all else float('nan')):9.3f}"
              f"{(f_on['med'] if f_on else float('nan')):15.1f}"
              f"{(f_hi['med'] if f_hi else float('nan')):15.1f}")

    json.dump(dict(gates=gates, sims=sims),
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/both_gates.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
