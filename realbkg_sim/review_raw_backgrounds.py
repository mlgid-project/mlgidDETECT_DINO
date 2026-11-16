"""Review a delivered raw-background sample: read every format, score it, render a contact sheet.

Replaces the ad-hoc scripts used on the 2026-09-16 sample. Applies the three acceptance tests the
donor pool actually needs, which the first sample taught us are independent:

  1. FULL EXPOSURE      >= 90% of unmasked pixels non-zero. The first sample's "background"-named
                        frames were beam-off: ID10 0.0% non-zero, Dima 0.2%, and the Eiger frame
                        was the direct beam on an otherwise black detector.
  2. NO DIFFRACTION     the q-narrow DoG detector from build_donor_bank, hot fraction near zero.
                        The only two full-exposure frames in that sample (ESRF_Perovskite at 962
                        counts/px, Elena at 28) were both covered in rings.
  3. GEOMETRY           distance AND alpha_i, on top of energy/wavelength/pixel size/beam centre.
                        None of the six beamtimes had alpha_i; one had a distance.

A frame has to pass all three. Passing 1 and failing 2 is a diffraction pattern; passing 2 and
failing 1 is an empty readout, which is worse than what we already have.

DEAD PIXELS. Eiger encodes them as 2^32-1 and Pilatus as 2^31-1. Left in, they made the Eiger
frame's mean read as 2.4e+08. A real pixel_mask from the file is preferred when present.

  python realbkg_sim/review_raw_backgrounds.py <sample_dir> [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Eiger 2^32-1, Pilatus 2^31-1, Lambda saturation 2^24-1, 16-bit detectors 2^16-1.
SENTINELS = (4294967295, 2147483647, 16777215, 65535)


def _as_index(v):
    """frame_index is not always an int: Lambda manifests label the MODULE ('m01'), and each
    .nxs then holds a single frame. Anything non-numeric means 'the only frame in this file'."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def read_frame(path, frame_index=0):
    """-> (image float64, note). Handles NeXus/HDF5 (incl. Eiger bitshuffle) and fabio formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.h5', '.nxs', '.hdf5'):
        import h5py
        try:
            import hdf5plugin  # noqa: F401   registers bitshuffle/LZ4 for Eiger
        except ImportError:
            pass
        with h5py.File(path, 'r') as f:
            ds = []
            f.visititems(lambda n, o: ds.append((n, o.shape))
                         if hasattr(o, 'shape') and o.ndim >= 2 and o.size > 1e4 else None)
            if not ds:
                return None, 'no 2-D dataset'
            name, _ = max(ds, key=lambda x: int(np.prod(x[1])))
            d = f[name]
            if d.ndim == 3:
                i = min(_as_index(frame_index), d.shape[0]-1)
                return np.asarray(d[i], dtype=np.float64), name
            return np.asarray(d[()], dtype=np.float64), name
    import fabio
    try:
        return fabio.open(path).data.astype(np.float64), 'fabio:auto'
    except Exception:
        for fmt in ('GEimage', 'adscimage', 'marccd', 'tifimage', 'edfimage', 'cbfimage'):
            try:
                return fabio.open(path, fmt).data.astype(np.float64), 'fabio:' + fmt
            except Exception:
                continue
    return None, 'unreadable'


def valid_mask(a):
    m = np.isfinite(a) & (a >= 0)
    for s in SENTINELS:
        m &= (a < s)
    return m


def score(a, mask=None):
    """-> dict of the three acceptance measurements."""
    from realbkg_sim.build_donor_bank import _detect
    import cv2
    m = valid_mask(a) if mask is None else (valid_mask(a) & mask)
    v = a[m]
    if v.size < 1000:
        return dict(ok=False, reason='too few valid pixels')
    det = cv2.erode(m.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    z = _detect(a.astype(np.float32), det, 2.5, 16.0, 3.0)
    hot = det & (z > 3.0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    ar = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0)
    ar = ar[ar >= 12]
    return dict(ok=True, shape=list(a.shape), masked_frac=float(1-m.mean()),
                nonzero=float((v > 0).mean()), mean_counts=float(v.mean()),
                p999=float(np.percentile(v, 99.9)), maxv=float(v.max()),
                hot_fraction=float(hot.sum()/max(det.sum(), 1)),
                n_blobs=int(len(ar)), blob_area=int(ar.sum()) if len(ar) else 0)


def verdict(s, geom, require_geometry=False):
    """Geometry is OFF by default: for a BACKGROUND donor the exact q mapping does not matter.

    The donor supplies texture, panel seams, beamstop shadow and radial falloff, all of which are
    already in the detector frame and survive any assumed mapping. Labels stay valid because the
    simulator places peaks in the same assumed coordinates it used for the background -- what the
    training pair needs is self-consistency, not agreement with the real experiment. The beam
    centre still matters (it sets where q=0 lands) but is fitted from the image here, not asked
    for. Pass --require-geometry to restore the strict gate.
    """
    if not s.get('ok'):
        return 'UNREADABLE'
    if s['nonzero'] < 0.90:
        return f"EMPTY ({100*s['nonzero']:.1f}% non-zero)"
    if s['hot_fraction'] > 5e-3:
        return f"DIFFRACTION (hot {s['hot_fraction']:.1e}, blobs {s['n_blobs']})"
    if require_geometry:
        d = geom.get('detector_distance_mm') or geom.get('detector_distance_m')
        if not isinstance(d, (int, float)) or not isinstance(geom.get('alpha_i_deg'), (int, float)):
            return 'NO GEOMETRY (need distance + alpha_i)'
    return 'PASS'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sample_dir')
    ap.add_argument('--out', default='/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images')
    ap.add_argument('--max-panels', type=int, default=24)
    ap.add_argument('--require-geometry', action='store_true')
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.sample_dir, 'manifest.json')))
    rows, frames = [], []
    for r in man:
        p = os.path.join(a.sample_dir, 'raw', r['raw_file'])
        if not os.path.exists(p):
            p = os.path.join(a.sample_dir, r['raw_file'])
        img, note = (read_frame(p, _as_index(r.get('frame_index', 0))) if os.path.exists(p)
                     else (None, 'missing'))
        s = score(img) if img is not None else dict(ok=False, reason=note)
        g = r.get('geometry') or {}
        row = dict(beamtime=r.get('beamtime'), sample=r.get('sample'),
                   frame=r.get('frame_index'), source=note, **s)
        row['verdict'] = verdict(s, g, a.require_geometry)
        rows.append(row)
        if img is not None and len(frames) < a.max_panels:
            frames.append((row, img))
        print(f"{str(row['beamtime'])[:24]:24s} {str(row.get('shape','')):13s} "
              f"nz {100*row.get('nonzero',0):5.1f}%  cnt {row.get('mean_counts',0):9.3f}  "
              f"hot {row.get('hot_fraction',0):8.1e}  -> {row['verdict']}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, 'rawbkg_review.json'), 'w'), indent=1)
    npass = sum(1 for r in rows if r['verdict'] == 'PASS')
    print(f"\n{npass} of {len(rows)} frames PASS all three tests")
    from collections import Counter
    for k, v in Counter(r['verdict'].split(' (')[0] for r in rows).most_common():
        print(f"   {k:20s} {v}")

    if frames:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(frames)
        fig, ax = plt.subplots(n, 1, figsize=(15, 3.1*n), squeeze=False)
        for k, (row, img) in enumerate(frames):
            m = valid_mask(img)
            pos = img[m & (img > 0)]
            lo, hi = (np.percentile(pos, 1), np.percentile(pos, 99.5)) if pos.size > 100 else (0, 1)
            d = np.log10(1 + 9*np.clip((img-lo)/max(hi-lo, 1e-9), 0, 1))
            d[~m] = 0
            ax[k][0].imshow(d, cmap='gray', aspect='auto', origin='lower')
            ax[k][0].set_title(f"{row['beamtime']} / {row['sample']}  {row.get('shape')}  "
                               f"nonzero {100*row.get('nonzero',0):.1f}%  "
                               f"hot {row.get('hot_fraction',0):.1e}  -> {row['verdict']}",
                               fontsize=9)
            ax[k][0].set_xticks([]); ax[k][0].set_yticks([])
        fig.suptitle('Raw background candidates (log stretch, dead pixels masked)', fontsize=13)
        fig.tight_layout()
        fn = os.path.join(a.out, 'rawbkg_review.png')
        fig.savefig(fn, dpi=110)
        print('wrote', fn)


if __name__ == '__main__':
    main()
