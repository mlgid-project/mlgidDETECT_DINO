"""Compare a pygidsim peak bank against the real labeled peaks of the eval sets.

Phase 0 gate for the pygidSIM track. The declined phase-P run used a bank that was 98.5%
perovskite; this asks, quantitatively, whether a bank's (q, chi, intensity) statistics resemble
the real labeled peaks the detector must find.

Both eval sets store labels in their own format, and NEITHER stores what phase O measured:
  * organic_labeled.h5 (pyGID/NeXus) -- `fitted_peaks` with `radius` already in A^-1, `angle` in
    degrees from the q_xy axis, and `visibility` 0-3. Its `amplitude`, `q_xy`, `q_z` and `is_ring`
    columns are ALL ZERO, so no intensity is available on this side.
  * 41.h5 (roi_data) -- `radius` in reciprocal-image PIXELS (converted here with
    GEO_PIXELPERANGSTROEM = 500, util/configuration.py:32), `angle` in degrees, `peak height` as an
    image-domain amplitude, and `confidence_level` as the GT confidence.

So the q and chi distributions are directly comparable across real and bank; the intensity ratio
is NOT strictly comparable between them, because real `peak height` is an image amplitude while a
bank carries structure-factor intensity. Use the intensity column to compare BANKS WITH EACH
OTHER, and the q/chi columns to compare banks with real.

NOTE on the 14.3 / 1.97 / 0.70 figures in MODIFICATIONS.md phase O: those were image-domain peak
contrast against local background after log+HE, from a script that no longer exists. Do not
compare them numerically with anything printed here.

Usage:
  python diagnostics/bank_stats.py --bank <bank.npz> [--bank <other.npz>]
"""
import argparse
import os

import h5py
import numpy as np

ORGANIC_H5 = '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
H5_41 = '/mnt/lustre/work/schreiber/szb389/datasets/41.h5'
PIXEL_PER_ANGSTROEM = 500.0      # util/configuration.py:32, converts 41's radius px -> A^-1
Q_SPLIT = 1.5          # A^-1; boundary between the "low q" and "high q" halves
MIN_Q = 0.05


def _stats(q, chi, inten, n_entries=None, per_entry=None):
    """Scale-free summary of one peak population; `inten` may be None."""
    keep = np.isfinite(q) & (q > MIN_Q)
    if inten is not None:
        keep &= np.isfinite(inten) & (inten > 0)
    q = q[keep]
    inten = inten[keep] if inten is not None else None
    chi = chi[keep] if chi is not None else None
    if inten is None:
        ratio = float('nan')
    else:
        lo, hi = inten[q < Q_SPLIT], inten[q >= Q_SPLIT]
        ratio = (np.median(hi) / np.median(lo)) if len(lo) and len(hi) else float('nan')
    out = dict(
        n_peaks=len(q),
        q_median=float(np.median(q)),
        q_p10=float(np.percentile(q, 10)),
        q_p90=float(np.percentile(q, 90)),
        frac_high_q=float((q >= Q_SPLIT).mean()),
        hi_lo_ratio=float(ratio),
    )
    if chi is not None:
        real_chi = chi[chi >= 0]
        out['chi_median'] = float(np.median(real_chi)) if len(real_chi) else float('nan')
    if n_entries:
        out['n_entries'] = n_entries
    if per_entry is not None and len(per_entry):
        out['peaks_per_entry'] = float(np.median(per_entry))
    return out


def organic_peaks(path, visibility_min=1):
    """Labeled peaks from the pyGID/NeXus organic set -> (q, chi, None, n_frames)."""
    qs, chis, n_frames = [], [], 0

    def visit(name, obj):
        nonlocal n_frames
        if not name.endswith('fitted_peaks'):
            return
        d = obj[()]
        d = d[d['visibility'] >= visibility_min]
        if not len(d):
            return
        n_frames += 1
        qs.append(d['radius'].astype(float))        # already A^-1
        chis.append(d['angle'].astype(float))       # degrees from the q_xy axis

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    if not qs:
        return None
    return np.concatenate(qs), np.concatenate(chis), None, n_frames


def peaks_41(path, min_confidence=0.1):
    """Labeled peaks from the roi_data set -> (q, chi, peak height, n_frames)."""
    qs, chis, amps, n_frames = [], [], [], 0

    def visit(name, obj):
        nonlocal n_frames
        if not name.endswith('roi_data') or 'radius' not in obj:
            return
        conf = obj['confidence_level'][()].astype(float)
        keep = conf >= min_confidence
        if not keep.any():
            return
        n_frames += 1
        qs.append(obj['radius'][()].astype(float)[keep] / PIXEL_PER_ANGSTROEM)
        chis.append(obj['angle'][()].astype(float)[keep])
        amps.append(obj['peak height'][()].astype(float)[keep])

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    if not qs:
        return None
    return np.concatenate(qs), np.concatenate(chis), np.concatenate(amps), n_frames


def bank_peaks(path):
    d = np.load(path, allow_pickle=True)
    return d['q'].astype(float), d['chi'].astype(float), d['intensity'].astype(float), \
        len(d['entry_start']), d['entry_count']


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic: max gap between the two ECDFs, in [0,1].

    Used instead of a p-value on purpose -- with millions of bank peaks any distributional
    difference is 'significant', so what matters is the EFFECT SIZE. 0 = identical shape.
    """
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float('nan')
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, 'right') / len(a)
    cb = np.searchsorted(b, grid, 'right') / len(b)
    return float(np.max(np.abs(ca - cb)))


def _row(label, s):
    return (f'{label:<28} {s["n_peaks"]:>9,} {s["q_median"]:>8.2f} '
            f'{s["q_p10"]:>7.2f} {s["q_p90"]:>7.2f} {s["frac_high_q"]:>9.2f} '
            f'{s["hi_lo_ratio"]:>10.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bank', action='append', default=[],
                    help='bank .npz; repeatable, compared side by side')
    ap.add_argument('--q-split', type=float, default=Q_SPLIT)
    ap.add_argument('--visibility-min', type=int, default=1)
    args = ap.parse_args()
    globals()['Q_SPLIT'] = args.q_split

    print(f'q split at {Q_SPLIT} A^-1; hi_lo = median(I | q>=split) / median(I | q<split)\n')
    print(f'{"source":<28} {"peaks":>9} {"q_med":>8} {"q_p10":>7} {"q_p90":>7} '
          f'{"frac_hiq":>9} {"hi/lo":>10}')
    print('-' * 84)

    real = {}
    for label, reader in (('REAL organic', organic_peaks), ('REAL 41', peaks_41)):
        path = ORGANIC_H5 if reader is organic_peaks else H5_41
        got = reader(path, args.visibility_min if reader is organic_peaks else 0.1)
        if got is None:
            print(f'{label:<28} (no labels found)')
            continue
        q, chi, amp, nfr = got
        real[label] = (q, chi)
        s_ = _stats(q, chi, amp)
        print(_row(f'{label} ({nfr} frames)', s_))
        print(f'{"":<28} chi median {s_.get("chi_median", float("nan")):.1f} deg'
              + ('' if amp is not None else '   [no intensity in this file]'))

    for path in args.bank:
        if not os.path.isfile(path):
            print(f'{os.path.basename(path):<28} MISSING')
            continue
        q, chi, inten, n_entries, counts = bank_peaks(path)
        s = _stats(q, chi, inten, n_entries, counts)
        print(_row(f'BANK {os.path.basename(path)}', s))
        print(f'{"":<28} {n_entries:,} entries, median {s["peaks_per_entry"]:.0f} peaks/entry, '
              f'chi median {s.get("chi_median", float("nan")):.1f} deg')
        bank_chi = chi[chi >= 0]
        for label, (rq, rchi) in real.items():
            print(f'{"":<28} vs {label:<14} KS(q) {ks(q, rq):.3f}   '
                  f'KS(chi) {ks(bank_chi, rchi):.3f}')


if __name__ == '__main__':
    main()
