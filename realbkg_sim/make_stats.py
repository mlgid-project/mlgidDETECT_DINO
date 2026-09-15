"""Consolidate the measured real statistics into the single file the simulator loads.

`realbkg_simulation.py` and the config point at `sim_real_stats.npz`; this is what writes it.
Run width_stats.py and amp_calibration.py first.

    python realbkg_sim/make_stats.py     ->  $DATA/sim_real_stats.npz

VALUES AS BUILT (2026-09-15):
    log_sq_mu 1.3595  log_sq_sd 0.3951     (per-frame median sigma_q,   exp -> 3.89 px)
    log_sc_mu 3.0101  log_sc_sd 0.7099     (per-frame median sigma_chi, exp -> 20.29 px)
    w_sq      0.5305  w_sc      0.8321     (within-frame sd of log sigma)
    log_an_mu 1.3635  log_an_sd 1.0023     (amp / local noise, n = 214)
    npf n=57                               (labelled peaks per frame, sampled directly)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK  = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
DATA  = f'{WORK}/datasets'
CACHE = os.environ.get('REALBKG_CACHE', f'{WORK}/tmp_diag/sim2')
INVENTORY = f'{CACHE}/inventory.json'

import numpy as np

W = np.load(f'{CACHE}/widthstats.npz')
A = np.load(f'{CACHE}/ampcal.npy')          # cols: amp, local_bkg, local_noise, amp/amp_max, sq, sc
an = A[:, 0]/A[:, 2]

out = dict(
    log_sq_mu=float(np.mean(W['log_sq'])), log_sq_sd=float(np.std(W['log_sq'])),
    log_sc_mu=float(np.mean(W['log_sc'])), log_sc_sd=float(np.std(W['log_sc'])),
    w_sq=float(np.median(W['w_sq'])),      w_sc=float(np.median(W['w_sc'])),
    log_an_mu=float(np.mean(np.log(an))),  log_an_sd=float(np.std(np.log(an))),
    npf=W['npf'], amp_noise=an, all_sq=W['all_sq'], all_sc=W['all_sc'], ratio=W['ratio'],
)
np.savez(f'{DATA}/sim_real_stats.npz', **out)
print('wrote', f'{DATA}/sim_real_stats.npz')
for k in ('log_sq_mu', 'log_sq_sd', 'log_sc_mu', 'log_sc_sd',
          'w_sq', 'w_sc', 'log_an_mu', 'log_an_sd'):
    print(f'  {k} = {out[k]:.4f}')
print(f'  npf n={len(out["npf"])}  amp_noise n={len(an)}')
