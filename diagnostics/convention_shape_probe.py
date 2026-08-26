"""Phase AC.3 — is the chi boxing convention a property of the DATASET or of the PEAK?

AC.2 settled the measurement and produced a problem. Box half-extent in units of the fitted sigma,
on raw polar data where the peak is physical and nothing needs calibrating:

    organic     k_chi 0.92   k_q 1.05     (isotropic: the box IS the peak's shape)
    41          k_chi 1.71   k_q 0.88     (2x anisotropic)
    simulator   k_chi 1.75   k_q 0.50     (3.5x anisotropic, by construction)

The gates disagree in chi by 1.9x, so NO single `a_coef` satisfies both -- if the convention is a
property of the dataset. But the two gates also hold different PEAK SHAPES: organic's peaks are
compact spots (sigma_chi/sigma_q = 0.67) and 41's are arcs (3.13). So the disagreement has two
possible readings, and they demand different fixes:

  DATASET-DRIVEN   two labelling pipelines with different habits. The model cannot tell which set a
                   frame came from, so no simulator setting serves both and `a_coef` becomes a
                   compromise that is wrong everywhere.
  SHAPE-DRIVEN     both pipelines box an ARC generously in chi and a SPOT tightly, and organic reads
                   low only because it is nearly all spots. Then the convention is a function of the
                   peak, the image tells the model which regime it is in, and the simulator should
                   implement k_chi(shape) rather than a constant -- serving both gates at once.

THE TEST, and it is a clean one: bin peaks by their own fitted aspect sigma_chi/sigma_q and compare
k_chi ACROSS THE TWO GATES WITHIN EACH BIN. Same shape, same convention => shape-driven. Same shape,
different convention => dataset-driven. The gates overlap in the 1-4 aspect range, which is where the
comparison is made; bins carrying fewer than MIN_BIN peaks are printed but flagged, never averaged
into a verdict.

Also binned by absolute sigma_chi, because a floor on how small a box a human will draw would produce
the same correlation for a duller reason -- small peaks getting relatively larger boxes.

Reuses AC.2's fitter and raw-polar loader unchanged. Neighbours masked (AC.2 showed masking changes
nothing, so this is for consistency, not correction). CPU-bound, ~6 min.
"""
import os, sys, json

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from diagnostics.gauss_fit_probe import DSETS, fit_peak, real_frames

ASPECT_EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
ASPECT_NAMES = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", ">8"]
SIG_EDGES = [0.0, 2.0, 4.0, 8.0, 16.0, 1e9]
SIG_NAMES = ["<2", "2-4", "4-8", "8-16", ">16"]
MIN_BIN = 15


def collect(name, path):
    rows = []
    for img, b, is_ring in real_frames(path, True)():
        for i in np.where(~is_ring)[0]:
            res, _why = fit_peak(img, b[i], np.delete(b, i, axis=0), True)
            if res is None:
                continue
            sq, sc = res
            rows.append(dict(sq=sq, sc=sc, aspect=sc / sq,
                             kq=abs(b[i][2] - b[i][0]) / 2 / sq,
                             kc=abs(b[i][3] - b[i][1]) / 2 / sc))
    print(f"  {name}: {len(rows)} fitted peaks", flush=True)
    return dict(name=name, rows=rows)


def binned(rows, key, edges, names, val):
    v = np.array([r[val] for r in rows]); x = np.array([r[key] for r in rows])
    out = []
    for i in range(len(names)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        out.append((int(m.sum()), float(np.median(v[m])) if m.sum() else float('nan')))
    return out


def show(title, sets, key, edges, names, val):
    print(f"\n  {title}")
    print(f"  {'gate':<20s}" + "".join(f"{n:>14s}" for n in names))
    for s in sets:
        b = binned(s['rows'], key, edges, names, val)
        cells = []
        for n, m in b:
            cells.append("      -  (0)" if n == 0 else
                         (f"{m:8.2f} ({n:3d})" + ("*" if n < MIN_BIN else " ")))
        print(f"  {s['name']:<20s}" + "".join(f"{c:>14s}" for c in cells))
    print(f"  (* fewer than {MIN_BIN} peaks — shown, not concluded from)")


def main():
    sets = [collect(n, p) for n, p in DSETS]

    print("\n" + "=" * 104)
    print("  THE TEST: k_chi at MATCHED PEAK SHAPE.  Same value across gates => shape-driven"
          " convention.")
    print("=" * 104)
    show("k_chi = (box_h/2) / sigma_chi,  binned by the peak's own aspect sigma_chi/sigma_q",
         sets, 'aspect', ASPECT_EDGES, ASPECT_NAMES, 'kc')
    show("k_q = (box_w/2) / sigma_q,  same bins",
         sets, 'aspect', ASPECT_EDGES, ASPECT_NAMES, 'kq')

    print("\n" + "=" * 104)
    print("  CONTROL: the same, binned by ABSOLUTE sigma_chi (a floor on box size would mimic it)")
    print("=" * 104)
    show("k_chi binned by sigma_chi (px)", sets, 'sc', SIG_EDGES, SIG_NAMES, 'kc')

    print("\n" + "=" * 104)
    print("  WHERE THE GATES OVERLAP IN SHAPE")
    print("=" * 104)
    print(f"  {'gate':<20s}{'peaks':>8s}{'aspect p10':>12s}{'p50':>8s}{'p90':>8s}"
          f"{'k_chi p50':>11s}{'k_q p50':>10s}")
    for s in sets:
        a = np.array([r['aspect'] for r in s['rows']])
        kc = np.array([r['kc'] for r in s['rows']]); kq = np.array([r['kq'] for r in s['rows']])
        print(f"  {s['name']:<20s}{len(a):8d}{np.percentile(a, 10):12.2f}"
              f"{np.percentile(a, 50):8.2f}{np.percentile(a, 90):8.2f}"
              f"{np.median(kc):11.2f}{np.median(kq):10.2f}")

    json.dump([{'name': s['name'],
                'aspect_kc': binned(s['rows'], 'aspect', ASPECT_EDGES, ASPECT_NAMES, 'kc'),
                'aspect_kq': binned(s['rows'], 'aspect', ASPECT_EDGES, ASPECT_NAMES, 'kq'),
                'sig_kc': binned(s['rows'], 'sc', SIG_EDGES, SIG_NAMES, 'kc'),
                'n': len(s['rows'])} for s in sets],
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/convention_shape.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
