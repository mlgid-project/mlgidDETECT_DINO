"""Eyeball the donor bank: best, p25, median and near-worst by leftover score.

`leftover` is the 99.9th percentile of the feature-detector z-score AFTER suppression, so a clean
donor scores near the detection threshold (~3) and one with a surviving arc scores in the tens.
Use this to see what a given score actually looks like before trusting the ranking.

    python realbkg_sim/check_bank.py     ->  bankcheck.png
"""
import json

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK  = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
DATA  = f'{WORK}/datasets'
CACHE = os.environ.get('REALBKG_CACHE', f'{WORK}/tmp_diag/sim2')
INVENTORY = f'{CACHE}/inventory.json'

import matplotlib.pyplot as plt

from util.exp_preprocess import apply_contrast

CHAIN = {'clip': (5, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}
CM = plt.cm.viridis.copy(); CM.set_bad('black')
f = h5py.File(os.environ.get('BANK_OUT', f'{DATA}/sim_background_bank4.h5'),'r')
meta=[json.loads(s) for s in f['meta'][()].astype(str)]
lv = np.array([m['leftover'] for m in meta])
order = np.argsort(lv)
sel = [order[0], order[len(order)//4], order[len(order)//2], order[-5]]
fig, axes = plt.subplots(len(sel),1, figsize=(15, len(sel)*3.2))
for ax, i in zip(axes, sel):
    b = f['background'][i].astype(np.float64); m = f['mask'][i].astype(bool)
    ax.imshow(np.where(m, apply_contrast(b, m, CHAIN), np.nan), cmap=CM, origin='lower',
              aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"z={meta[i]['leftover']:.0f} rm={meta[i]['removed']:.2f}\n{os.path.basename(meta[i]['path'])[:16]}", fontsize=6)
fig.suptitle('donor backgrounds sorted by leftover score: best, p25, median, near-worst', fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(f'{CACHE}/bankcheck.png', dpi=100)
print('saved; leftover percentiles', np.round(np.percentile(lv,[10,50,90]),1))
