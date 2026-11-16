"""How bright a real peak is RELATIVE TO ITS OWN FRAME's local noise.

This is what sets simulated peak brightness. The fitted `offset` is not usable as a background
estimate (it is ~1 in normalised units while amplitudes run to 1e5), so the local background and
noise are measured directly: suppress the frame's features, smooth what is left, and take the
robust sd of the residual in a 65x65 patch around each fitted peak.

RESULT (214 fitted peaks over 11 real frames):
    amp / local noise    p10 1.49   p50 3.12   p90 16.3
    amp / local bkg      p50 0.90
    local bkg / noise    p50 4.11                 <- the donor-selection target in realbkg_simulation
    sigma_chi/sigma_q    p50 4.40

Real peaks are FAINT -- a median peak is three times the local noise. Anything much brighter across
the board reads as synthetic immediately, which is what the old simulator got wrong.

    python realbkg_sim/amp_calibration.py     ->  ampcal.npy
"""
import json

import cv2
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

from build_donor_bank import HEIGHT, WIDTH, suppress, to_polar

rows = json.load(open(INVENTORY))
recs = []
done = 0
for r in rows:
    if done >= 60: break
    try:
        with h5py.File(r['path'], 'r') as f:
            g = f[r['entry']]
            if 'data/analysis' not in g: continue
            frames = sorted(g['data/analysis'].keys())
            qz, qxy = np.asarray(g['data/q_z']), np.asarray(g['data/q_xy'])
            for fr in frames[:3]:
                i = int(''.join(ch for ch in fr if ch.isdigit()))
                if i >= r['n']: continue
                a = g[f'data/analysis/{fr}/fitted_peaks']
                if 'parameters_peak' not in a: continue
                P = a['parameters_peak'][()]
                if len(P) < 5: continue
                raw = np.asarray(g['data/img_gid_q'][i], dtype=np.float32)
                if raw.shape != (len(qz), len(qxy)): continue
                pol, m, qmax = to_polar(raw, qz, qxy)
                if m.mean() < 0.25: continue
                bkg, rm, _ = suppress(pol, m)
                keep = m & ~rm
                sm = cv2.GaussianBlur(np.where(keep, bkg, 0).astype(np.float32), (0,0), 16)
                wn = cv2.GaussianBlur(keep.astype(np.float32), (0,0), 16)
                sm = sm/np.maximum(wn, 1e-6)
                res = np.where(keep, bkg - sm, np.nan)
                amax = float(np.max(P['amp']))
                for p in P:
                    x, y = int(round(p['x0'])), int(round(p['y0']))
                    if not (10 < x < WIDTH-11 and 10 < y < HEIGHT-11): continue
                    sub = res[max(y-32,0):y+33, max(x-32,0):x+33]
                    sub = sub[np.isfinite(sub)]
                    if sub.size < 200: continue
                    noise = 1.4826*np.median(np.abs(sub - np.median(sub)))
                    B = float(sm[y, x])
                    if noise <= 0 or B <= 0: continue
                    recs.append((float(p['amp']), B, noise, float(p['amp'])/amax,
                                 float(p['sigma_x']), float(p['sigma_y'])))
                done += 1
                print(f'  {os.path.basename(r["path"])[:24]:<26} {len(P):>3} peaks  '
                      f'B={B:.3g} noise={noise:.3g}', flush=True)
    except Exception as e:
        print('ERR', repr(e)[:70], flush=True)
R = np.array(recs)
print(f'\n{len(R)} fitted peaks over {done} real frames')
def q(v, nm, ps=(1,10,25,50,75,90,99)):
    v = np.asarray(v); v = v[np.isfinite(v) & (v > 0)]
    print(f'{nm:<22} n={len(v):>6} ' + '  '.join(f'p{p}={np.percentile(v,p):.4g}' for p in ps))
q(R[:,0]/R[:,2], 'amp / local noise')
q(R[:,0]/R[:,1], 'amp / local bkg')
q(R[:,1]/R[:,2], 'local bkg / noise')
q(R[:,3], 'amp / amp_max')
q(R[:,4], 'sigma_q (px)'); q(R[:,5], 'sigma_chi (px)')
q(R[:,5]/np.maximum(R[:,4],1e-9), 'sigma_chi/sigma_q')
np.save(f'{CACHE}/ampcal.npy', R)
