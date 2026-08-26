"""Phase AC.1 — the aspect-ratio floor, counted exactly.

AC found the simulator's segments are ~3.3x taller in chi and ~4.3x narrower in q than organic's.
`simulation.py:204` says why, structurally:

    if is_segment:
        a_widths = torch.maximum(a_widths, widths * (torch.rand_like(widths) + 1.0))

so sigma_chi >= sigma_q * U(1,2) for EVERY simulated segment, and with a_coef=3.5 / w_coef=1.0 the
box aspect box_h/box_w = 3.5 * (sigma_chi/sigma_q) has a hard floor of 3.5 and a typical value of
5.25. A simulated segment can never be wider in q than it is tall in chi.

This counts how often the real gates ARE. No model, no simulation of images -- labels only for the
real sets, `simulate_labels` for the sim, so it runs in a couple of minutes.
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
from diagnostics.ring_count_fix import classify

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41 (perovskite)", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM = 200
PCTS = [10, 25, 50, 75, 90]
EDGES = [0.0, 0.5, 1.0, 2.0, 3.5, 7.0, 1e9]
NAMES = ["<0.5", "0.5-1", "1-2", "2-3.5", "3.5-7", ">7"]


def report(name, ar):
    ar = np.asarray([x for x in ar if np.isfinite(x) and x > 0])
    frac = [float(np.mean((ar >= EDGES[i]) & (ar < EDGES[i + 1]))) for i in range(len(NAMES))]
    return dict(name=name, n=len(ar), wider_than_tall=float(np.mean(ar < 1.0)),
                below_sim_floor=float(np.mean(ar < 3.5)),
                pct={p: float(np.percentile(ar, p)) for p in PCTS}, hist=frac)


def do_real(name, path):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                               buffer_size=5))
    ar = []
    for gc in ds.iter_images():
        L = gc.polar_labels
        b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
        if not len(b):
            continue
        _h, _sp, is_ring = classify(gc.converted_polar_image[0, 0], b)
        s = b[~is_ring]
        if len(s):
            ar += list(np.abs(s[:, 3] - s[:, 1]) / np.maximum(np.abs(s[:, 2] - s[:, 0]), 1e-6))
    if hasattr(ds, 'close'):
        ds.close()
    return report(name, ar)


def do_sim(clusters, n, dev):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    ar, errs = [], []
    for k in range(n):
        sd = 90000 + k
        random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
        try:
            # full simulate_img, as in ring_count_fix: simulate_labels alone leaves the detector
            # mask / dark-area state unset and every call raises into the except below.
            _img, boxes, _m, isr = sim.simulate_img()
        except Exception as e:
            if not errs:
                print(f"  sim frame {k} raised {type(e).__name__}: {e}", flush=True)
            errs.append(1)
            continue
        b = boxes.detach().cpu().numpy().astype(np.float64)
        r = isr.detach().cpu().numpy().astype(bool)
        if not len(b):
            continue
        s = b[~r]
        if len(s):
            ar += list(np.abs(s[:, 3] - s[:, 1]) / np.maximum(np.abs(s[:, 2] - s[:, 0]), 1e-6))
    print(f"  sim clusters {'ON' if clusters else 'OFF'}: {len(ar)} segments, "
          f"{len(errs)} frames raised", flush=True)
    return report(f"sim clusters {'ON' if clusters else 'OFF'}", ar)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}", flush=True)
    rows = [do_real(n, p) for n, p in DSETS] + [do_sim(False, N_SIM, dev), do_sim(True, N_SIM, dev)]

    print("\n" + "=" * 100)
    print("  SEGMENT BOX ASPECT  box_h(chi) / box_w(q)   — simulator's structural floor is 3.5")
    print("=" * 100)
    print(f"  {'set':<24s}{'segs':>8s}{'wider than tall':>17s}{'below sim floor 3.5':>22s}")
    for r in rows:
        print(f"  {r['name']:<24s}{r['n']:8d}{r['wider_than_tall']:17.3f}{r['below_sim_floor']:22.3f}")
    print(f"\n  aspect percentiles      " + "".join(f"{('p' + str(p)):>10s}" for p in PCTS))
    for r in rows:
        print(f"  {r['name']:<24s}" + "".join(f"{r['pct'][p]:10.2f}" for p in PCTS))
    print(f"\n  share by aspect band    " + "".join(f"{n:>10s}" for n in NAMES))
    for r in rows:
        print(f"  {r['name']:<24s}" + "".join(f"{v:10.3f}" for v in r['hist']))
    json.dump(rows, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/aspect.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
