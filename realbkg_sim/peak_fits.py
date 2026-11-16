"""What real peaks look like: mlgidFIT's own 2-D Gaussian fits, pooled over the donor corpus.

`fitted_peaks/parameters_peak` stores (amp, x0, y0, sigma_x, sigma_y, theta, offset) per peak, in
the same 512x1024 polar pixel units the detector works in. These are real peaks in real frames,
fitted by the pipeline this detector feeds -- not anything estimated from a 1-D cut.

RESULT (1,926 peaks, 57 frames). The two numbers that reshaped the simulator:
    sigma_x (q)     p10 1.52   p50 3.81   p90 8.92   px
    sigma_y (chi)   p10 4.54   p50 22.5   p90 83.2   px
i.e. real peaks are ARCS, not blobs. And they are faint -- see amp_calibration.py.
    peaks/frame     p10 23     p50 35     p90 43
    amp/amp_max     p10 8e-4   p50 4.8e-3 p90 0.065      (within-frame dynamic range)
    type            1 = ring (5), 2 = peak (916)          -> rings are ~0.5% of labels here

    python realbkg_sim/peak_fits.py
"""
import json

import h5py
import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK  = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
DATA  = f'{WORK}/datasets'
CACHE = os.environ.get('REALBKG_CACHE', f'{WORK}/tmp_diag/sim2')
INVENTORY = f'{CACHE}/inventory.json'

rows = json.load(open(INVENTORY))
amp, sx, sy, off, rad, ang, rw, aw, typ, ratio, npf = [], [], [], [], [], [], [], [], [], [], []
x0, y0 = [], []
nfile = 0
for r in rows:
    try:
        with h5py.File(r['path'], 'r') as f:
            g = f[r['entry']]
            if 'data/analysis' not in g: continue
            nfile += 1
            for fr in g['data/analysis']:
                a = g[f'data/analysis/{fr}']
                if 'fitted_peaks' not in a: continue
                fp = a['fitted_peaks']
                if 'parameters_peak' not in fp: continue
                P = fp['parameters_peak'][()]
                if len(P) == 0: continue
                amp += list(P['amp']); sx += list(P['sigma_x']); sy += list(P['sigma_y'])
                off += list(P['offset']); x0 += list(P['x0']); y0 += list(P['y0'])
                npf.append(len(P))
                m = float(np.max(P['amp']))
                if m > 0: ratio += list(np.asarray(P['amp'])/m)
                if 'radius' in fp: rad += list(fp['radius'][()])
                if 'angle' in fp: ang += list(fp['angle'][()])
                if 'radius_width' in fp: rw += list(fp['radius_width'][()])
                if 'angle_width' in fp: aw += list(fp['angle_width'][()])
                if 'type' in fp: typ += list(fp['type'][()])
    except Exception as e:
        print('ERR', r['path'], repr(e)[:60])
A = lambda v: np.asarray(v, dtype=float)
def q(v, nm, ps=(1,10,50,90,99)):
    v = A(v); v = v[np.isfinite(v)]
    if not len(v): print(f'{nm:<16} empty'); return
    print(f'{nm:<16} n={len(v):>7}  ' + '  '.join(f'p{p}={np.percentile(v,p):.3g}' for p in ps))
print(f'{nfile} entries with analysis, {len(npf)} frames with fits')
q(npf,'peaks/frame'); q(amp,'amplitude'); q(off,'offset(local bkg)')
q(A(amp)/np.maximum(A(off),1e-9),'amp/offset')
q(sx,'sigma_x(px?)'); q(sy,'sigma_y(px?)'); q(x0,'x0'); q(y0,'y0')
q(rad,'radius(q)'); q(ang,'angle(chi)'); q(rw,'radius_width'); q(aw,'angle_width')
q(ratio,'amp/amp_max', ps=(1,5,10,25,50,75,90))
t = A(typ)
if len(t): print('type values:', {int(k): int(v) for k, v in zip(*np.unique(t, return_counts=True))})
np.savez(f'{CACHE}/peakstats.npz',
         amp=A(amp), sx=A(sx), sy=A(sy), off=A(off), ratio=A(ratio), npf=A(npf),
         rad=A(rad), ang=A(ang), rw=A(rw), aw=A(aw), typ=A(typ), x0=A(x0), y0=A(y0))
