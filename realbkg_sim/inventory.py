"""Inventory the real-frame corpus used as background donors, and leak-check it against eval.

Writes `inventory.json` (one row per hdf5 entry: path, entry, frame count, q axes), which every
later step consumes, and prints the azimuthal-I(q) cosine of every donor frame against every
`organic_labeled.h5` frame.

RESULT WHEN BUILT (2026-09-15): 46 entries, 4,040 linear frames; max cosine to any eval frame
**0.888**, against a 0.949 same-material baseline and a 0.999 same-frame value. No eval frame is
in the donor pool. The matcher is the one validated in the SSL corpus leak check -- q-calibrated,
chi-averaged I(q) -- because 2-D pixel cosine is unreliable across conversion pipelines.

    python realbkg_sim/inventory.py
"""
import glob
import json

import h5py
import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK   = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
DATA   = f'{WORK}/datasets'
DONOR_SRC = os.environ.get('DONOR_SRC', f'{DATA}/ekaterina_aftermlgidFIT')
CACHE  = os.environ.get('REALBKG_CACHE', f'{WORK}/tmp_diag/sim2')
INVENTORY = f'{CACHE}/inventory.json'

ROOT = DONOR_SRC
fs = sorted(glob.glob(f'{ROOT}/**/*.h5', recursive=True))
fs = [p for p in fs if '/backup/' not in p]
rows = []
for p in fs:
    try:
        with h5py.File(p, 'r') as f:
            for ent in f:
                g = f[ent]
                if 'data/img_gid_q' not in g: continue
                d = g['data/img_gid_q']
                rows.append(dict(path=p, entry=ent, n=d.shape[0], shape=list(d.shape[1:]),
                                 qz=float(g['data/q_z'][-1]), qxy=float(g['data/q_xy'][-1])))
    except Exception as e:
        print('ERR', p, e)
tot = sum(r['n'] for r in rows)
print(f'{len(rows)} entries, {tot} linear frames')
from collections import Counter
print('shapes:', Counter(tuple(r['shape']) for r in rows).most_common(6))
print('n per entry:', Counter(r['n'] for r in rows).most_common(8))
json.dump(rows, open(INVENTORY,'w'), indent=1)

# ---- leak check: azimuthal I(q) against the two eval sets ----
def iq_profiles(path, limit=None):
    out = []
    with h5py.File(path, 'r') as f:
        for ent in f:
            g = f[ent]
            if 'data/img_gid_q' not in g: continue
            qz = np.asarray(g['data/q_z']); qxy = np.asarray(g['data/q_xy'])
            n = g['data/img_gid_q'].shape[0]
            idx = range(n) if limit is None else range(0, n, max(1, n//limit))
            for i in idx:
                img = np.nan_to_num(np.asarray(g['data/img_gid_q'][i], dtype=np.float32))
                if img.shape != (len(qz), len(qxy)):
                    continue
                Z, XY = np.meshgrid(qz, qxy, indexing='ij')
                r = np.hypot(Z, XY)
                b = np.linspace(0, 3.0, 128)
                w = np.digitize(r.ravel(), b)
                s = np.bincount(w, img.ravel(), minlength=130)[:130]
                c = np.bincount(w, minlength=130)[:130]
                pr = s/np.maximum(c, 1)
                pr = pr/max(pr.max(), 1e-9)
                out.append((f'{os.path.basename(path)}:{ent}:{i}', pr))
    return out

ek = []
for r in rows:
    ek += iq_profiles(r['path'], limit=3) if r['n'] > 3 else iq_profiles(r['path'])
    if len(ek) > 400: break
ev = iq_profiles(f'{DATA}/organic_labeled.h5')
print('organic eval profiles:', len(ev))
print(f'\nleak check: {len(ek)} ekaterina profiles vs {len(ev)} organic-eval profiles')
A = np.array([p for _, p in ek]); B = np.array([p for _, p in ev])
A = A/np.linalg.norm(A, axis=1, keepdims=True); B = B/np.linalg.norm(B, axis=1, keepdims=True)
M = A @ B.T
print(f'  max cosine {M.max():.4f}   (same-frame would be ~1.000)')
bad = np.argwhere(M > 0.97)
seen = set()
for i, j in bad:
    k = ek[i][0].split(':')[0]
    if k not in seen:
        seen.add(k); print(f'  >0.97: {ek[i][0]}  vs organic frame {j}  cos {M[i,j]:.4f}')
print('  files to exclude:', sorted(seen) if seen else 'none')
