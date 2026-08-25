"""Phase AA.4 — ring counts, done properly. AA.3's ring column for 41 was WRONG.

AA.3 reported 0.0 rings/frame on 41.h5. The user flagged that as impossible — the perovskite set has
many rings. Cause, confirmed in the code: `util/labeleddataset.py:138-145` (`create_boxes`) copies
boxes / radii / widths / angles / angles_std / confidences / intensities / img_nr onto `polar_labels`
but NEVER `is_ring`. The H5 path sets it only on `reciprocal_labels` (line 170). 41.h5 loads through
`H5GIWAXSDataset`, so `polar_labels.is_ring` was empty, AA.3's length check failed, and it silently
defaulted to all-False. Organic loads through `PyGIDDataset`, which does populate the flag
(`pygidloader.py:169`) — so that column had a real flag behind it, but it is re-checked here too.

Primary criterion is the user's, and it needs no flag: **a ring spans the entire non-NaN extent of the
frame at its radius.** For each labelled box, the valid χ span is measured from the image itself
(rows with data in the box's q columns) and the box counts as a ring if it covers most of it.

Three readings per dataset so none is taken on trust:
  - FLAG    `polar_labels.is_ring`, then `reciprocal_labels.is_ring` as fallback (length-checked, and
            it says so when the lengths disagree rather than silently defaulting — the exact failure
            that produced the wrong number)
  - SPAN    the user's criterion, box χ-extent / valid χ span at that q >= RING_FRAC
  - HEIGHT  the raw box χ-extent distribution, printed as percentiles, so the split is visible and
            the SPAN threshold can be judged rather than believed

Applied identically to organic, 41, and the simulator.

No model. ~4 min. See tmp_diag/run_ringfix.sbatch.
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

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM = 200
RING_FRAC = 0.7          # box χ-extent / valid χ span at that radius, above which it is a ring
PCTS = [10, 25, 50, 75, 90]


def valid_span_at(img, b):
    """Rows carrying data in this box's q columns — the 'non-NaN frame' extent at that radius."""
    H, W = img.shape
    q0 = int(np.clip(min(b[0], b[2]), 0, W - 1))
    q1 = int(np.clip(max(b[0], b[2]), 0, W - 1))
    if q1 <= q0:
        q1 = min(q0 + 1, W)
    col = img[:, q0:q1] > 1e-6
    rows = np.where(col.any(1))[0]
    return float(rows[-1] - rows[0] + 1) if len(rows) else float(H)


def classify(img, boxes):
    h = np.abs(boxes[:, 3] - boxes[:, 1])
    span = np.array([valid_span_at(img, b) for b in boxes])
    return h, span, (h >= RING_FRAC * np.maximum(span, 1.0))


def do_dataset(name, path):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                               buffer_size=5))
    n_fr = n_obj = flag_ring = span_ring = 0
    flag_state = 'polar_labels.is_ring'
    heights, spans = [], []
    for gc in ds.iter_images():
        L = gc.polar_labels
        b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
        if not len(b):
            continue
        img = gc.converted_polar_image[0, 0]
        n_fr += 1; n_obj += len(b)

        fl = list(getattr(L, 'is_ring', []) or [])
        if len(fl) != len(b):
            fl2 = list(getattr(getattr(gc, 'reciprocal_labels', None), 'is_ring', []) or [])
            if len(fl2) == len(b):
                fl, flag_state = fl2, 'reciprocal_labels.is_ring (polar was empty/mismatched)'
            else:
                fl, flag_state = [], f'UNAVAILABLE (polar {len(fl)}, reciprocal {len(fl2)}, boxes {len(b)})'
        if len(fl) == len(b):
            flag_ring += int(np.array(fl, dtype=bool).sum())

        h, sp, isr = classify(img, b)
        span_ring += int(isr.sum())
        heights.append(h); spans.append(sp)
    if hasattr(ds, 'close'):
        ds.close()
    H = np.concatenate(heights) if heights else np.zeros(0)
    SP = np.concatenate(spans) if spans else np.zeros(0)
    return dict(name=name, frames=n_fr, objects=n_obj, flag_state=flag_state,
                flag_ring=flag_ring, span_ring=span_ring,
                flag_per_frame=flag_ring / max(n_fr, 1), span_per_frame=span_ring / max(n_fr, 1),
                seg_per_frame=(n_obj - span_ring) / max(n_fr, 1),
                ring_seg=span_ring / max(n_obj - span_ring, 1),
                h_pct={p: float(np.percentile(H, p)) for p in PCTS} if len(H) else {},
                span_med=float(np.median(SP)) if len(SP) else float('nan'))


def do_sim(clusters, n, dev):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    n_fr = n_obj = flag_ring = span_ring = 0
    heights, spans = [], []
    for k in range(n):
        _sd = 90000 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        try:
            img, bx, _m, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy().astype(np.float64)
        if not len(b):
            continue
        im = img.detach().cpu().numpy()
        im = im[0, 0] if im.ndim == 4 else (im[0] if im.ndim == 3 else im)
        n_fr += 1; n_obj += len(b)
        flag_ring += int(isr.detach().cpu().numpy().astype(bool).sum())
        h, sp, r = classify(im, b)
        span_ring += int(r.sum())
        heights.append(h); spans.append(sp)
    H = np.concatenate(heights) if heights else np.zeros(0)
    SP = np.concatenate(spans) if spans else np.zeros(0)
    return dict(name=f"sim clusters {'ON' if clusters else 'OFF'}", frames=n_fr, objects=n_obj,
                flag_state='simulator is_ring (authoritative)',
                flag_ring=flag_ring, span_ring=span_ring,
                flag_per_frame=flag_ring / max(n_fr, 1), span_per_frame=span_ring / max(n_fr, 1),
                seg_per_frame=(n_obj - span_ring) / max(n_fr, 1),
                ring_seg=span_ring / max(n_obj - span_ring, 1),
                h_pct={p: float(np.percentile(H, p)) for p in PCTS} if len(H) else {},
                span_med=float(np.median(SP)) if len(SP) else float('nan'))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  RING_FRAC={RING_FRAC}", flush=True)
    rows = [do_dataset(n, p) for n, p in DSETS]
    rows += [do_sim(False, N_SIM, dev), do_sim(True, N_SIM, dev)]

    print("\n" + "=" * 104)
    print("  RINGS PER FRAME — corrected.  AA.3 reported 41 as 0.0 rings/frame; that was a probe bug.")
    print("=" * 104)
    print(f"  {'set':<20s}{'frames':>7s}{'objects':>9s}{'FLAG rings/fr':>15s}"
          f"{'SPAN rings/fr':>15s}{'segs/fr':>10s}{'ring:seg':>10s}")
    for r in rows:
        print(f"  {r['name']:<20s}{r['frames']:7d}{r['objects']:9d}{r['flag_per_frame']:15.1f}"
              f"{r['span_per_frame']:15.1f}{r['seg_per_frame']:10.1f}{r['ring_seg']:10.3f}")
    print("\n  where the flag came from:")
    for r in rows:
        print(f"    {r['name']:<20s} {r['flag_state']}")

    print("\n" + "=" * 104)
    print("  BOX χ-EXTENT percentiles (px) — so the SPAN threshold can be judged, not believed")
    print("=" * 104)
    print(f"  {'set':<20s}" + "".join(f"{('p' + str(p)):>10s}" for p in PCTS) + f"{'valid span':>13s}")
    for r in rows:
        print(f"  {r['name']:<20s}" +
              "".join(f"{r['h_pct'].get(p, float('nan')):10.1f}" for p in PCTS) +
              f"{r['span_med']:13.1f}")
    json.dump(rows, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/ring_count_fix.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
