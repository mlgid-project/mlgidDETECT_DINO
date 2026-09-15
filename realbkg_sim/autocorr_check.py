"""Does the feature removal preserve the background's pixel-to-pixel correlation?

It is the diagnostic that caught the worst bug in this pipeline. Real backgrounds are strongly
correlated -- unprocessed donor frames measure lag-1 autocorrelation 0.53, real organic eval frames
0.31 -- and a WHITE background is exactly what makes a simulated frame look synthetic. With an iid
fill the suppression destroyed it:

    raw 0.876 | 1 round 0.465 | 3 rounds 0.015 | untouched tiles of the same frame 0.431

After switching the fill to a chi-shifted copy of the frame's own residual:

    raw 0.876 | 1 round 0.676 | 3 rounds 0.650

Run this after any change to `suppress()`.

    python realbkg_sim/autocorr_check.py
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

from build_donor_bank import suppress, to_polar

def ac1(b, m, T=48):
    out=[]
    for r in range(0,512-T,T):
        for c in range(0,1024-T,T):
            if not m[r:r+T,c:c+T].all(): continue
            t=b[r:r+T,c:c+T].astype(float); d=t-t.mean(); v=(d*d).mean()
            if v>0: out.append((d[:,:-1]*d[:,1:]).mean()/v)
    return float(np.median(out)) if out else np.nan
rows=json.load(open(INVENTORY))
for r in rows[:5]:
    with h5py.File(r['path'],'r') as f:
        g=f[r['entry']]; qz=np.asarray(g['data/q_z']); qxy=np.asarray(g['data/q_xy'])
        raw=np.asarray(g['data/img_gid_q'][r['n']//2],dtype=np.float32)
    if raw.shape!=(len(qz),len(qxy)):
        if raw.shape==(len(qxy),len(qz)): raw,qz,qxy=raw.T,qxy,qz
        else: continue
    pol,m,_=to_polar(raw,qz,qxy)
    a0=ac1(pol,m)
    b1,rm1,_=suppress(pol,m,rounds=1)
    b3,rm3,_=suppress(pol,m,rounds=3)
    # ac on ONLY the untouched pixels: zero out removed and measure tiles with no removal
    def ac_clean(b, m, rm, T=48):
        out=[]
        for rr in range(0,512-T,T):
            for cc in range(0,1024-T,T):
                if not m[rr:rr+T,cc:cc+T].all(): continue
                if rm[rr:rr+T,cc:cc+T].any(): continue
                t=b[rr:rr+T,cc:cc+T].astype(float); d=t-t.mean(); v=(d*d).mean()
                if v>0: out.append((d[:,:-1]*d[:,1:]).mean()/v)
        return float(np.median(out)) if out else np.nan
    print(f'{os.path.basename(r["path"])[:22]:<24} raw {a0:.3f} | 1 round {ac1(b1,m):.3f} '
          f'(rm {rm1.mean():.2f}) | 3 rounds {ac1(b3,m):.3f} (rm {rm3.mean():.2f}) | '
          f'untouched-tiles-only {ac_clean(b3,m,rm3):.3f}')
