"""How peak widths vary BETWEEN frames and WITHIN one frame.

The simulator draws a per-frame median sigma_q and sigma_chi from the between-frame lognormals,
then scatters each peak around it by the within-frame spread. Both come from here.

RESULT (57 real frames with >= 6 fitted peaks):
    between-frame  median sigma_q    exp(mu) 3.89 px   sd(log) 0.40
    between-frame  median sigma_chi  exp(mu) 20.29 px  sd(log) 0.71
    within-frame   sd(log sigma_q)   0.53
    within-frame   sd(log sigma_chi) 0.83
    per-peak corr(log sigma_q, log sigma_chi) = 0.00      -> the two axes are INDEPENDENT

The within-frame spread is large: a real frame contains both sharp and broad peaks. That
contradicts the "widths stable within 10% per image" rule the earlier notebook used, and the data
wins.

    python realbkg_sim/width_stats.py     ->  widthstats.npz
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
frames = []
for r in rows:
    try:
        with h5py.File(r['path'], 'r') as f:
            g = f[r['entry']]
            if 'data/analysis' not in g: continue
            for fr in g['data/analysis']:
                a = g[f'data/analysis/{fr}']
                if 'fitted_peaks/parameters_peak' not in a: continue
                P = a['fitted_peaks/parameters_peak'][()]
                if len(P) < 6: continue
                frames.append(dict(sq=np.asarray(P['sigma_x'], float),
                                   sc=np.asarray(P['sigma_y'], float),
                                   amp=np.asarray(P['amp'], float)))
    except Exception:
        pass
print(f'{len(frames)} real frames with >=6 fitted peaks')
lq = np.array([np.median(np.log(f['sq'][f['sq'] > 0])) for f in frames])
lc = np.array([np.median(np.log(f['sc'][f['sc'] > 0])) for f in frames])
wq = np.array([np.std(np.log(f['sq'][f['sq'] > 0])) for f in frames])
wc = np.array([np.std(np.log(f['sc'][f['sc'] > 0])) for f in frames])
print(f'\nbetween-frame  median sigma_q  : exp(mean) {np.exp(lq.mean()):.2f} px, sd(log) {lq.std():.2f}')
print(f'between-frame  median sigma_chi: exp(mean) {np.exp(lc.mean()):.2f} px, sd(log) {lc.std():.2f}')
print(f'within-frame   sd(log sigma_q)  : median {np.median(wq):.2f}')
print(f'within-frame   sd(log sigma_chi): median {np.median(wc):.2f}')
rho = np.corrcoef(lq, lc)[0,1]
print(f'corr(log median sigma_q, log median sigma_chi) across frames = {rho:.2f}')
# per-peak correlation within frames
allq = np.concatenate([np.log(f['sq'][(f['sq']>0)&(f['sc']>0)]) for f in frames])
allc = np.concatenate([np.log(f['sc'][(f['sq']>0)&(f['sc']>0)]) for f in frames])
print(f'per-peak corr(log sigma_q, log sigma_chi) = {np.corrcoef(allq, allc)[0,1]:.2f}')
# within-frame dynamic range
rat = np.concatenate([f['amp']/max(f['amp'].max(),1e-9) for f in frames])
print(f'\namp/amp_max: ' + '  '.join(f'p{p}={np.percentile(rat,p):.4g}' for p in (1,10,25,50,75,90)))
npf = np.array([len(f['amp']) for f in frames])
print(f'peaks/frame: ' + '  '.join(f'p{p}={np.percentile(npf,p):.0f}' for p in (10,25,50,75,90)))
np.savez(f'{CACHE}/widthstats.npz',
         log_sq=lq, log_sc=lc, w_sq=wq, w_sc=wc, ratio=rat, npf=npf,
         all_sq=np.exp(allq), all_sc=np.exp(allc))
