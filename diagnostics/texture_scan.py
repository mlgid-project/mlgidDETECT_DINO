"""Does giving realbkg rings 41-like texture along chi make them foreign to a realbkg model?

transplant_probe.py: on the processed image realbkg r7 rings have patch-scale variation along chi
(CV after a 61-row detrend) of 0.003, legacy 0.031, real 41 0.106. conv3 is confident on r7 rings
(0.95) and fails on legacy (0.36) and 41 (0.30) rings. If texture is the feature, conv3's ring score
should fall as r7 rings get 41-like texture -- and the fix is to train on textured rings.

Same r7 frames (all seeds fixed), only `realbkg_ring_chi_texture` varies.
  python diagnostics/texture_scan.py name=/ckpt.pth [...]
"""
import os, sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
import peak_shape_compare as PS
from cross_sim_probe import score
from transplant_probe import patchiness

N = int(os.environ.get('TS_FRAMES', 60))
R7 = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'
BASE = dict(realbkg_ring_min_gap=8.0, realbkg_rings_cap=(2, 60), realbkg_mosaic_seed=1234)
SDS = [float(v) for v in os.environ.get('TS_SDS', '0,0.15,0.3,0.6,1.0').split(',')]
CORR = tuple(float(v) for v in os.environ.get('TS_CORR', '3,20').split(','))

if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    sets = {}
    for sd in SDS:
        random.seed(11); np.random.seed(11); torch.manual_seed(11)
        sets[sd] = list(PS.realbkg6(N, config=R7, overrides=dict(BASE, realbkg_ring_chi_texture=sd,
                                                                       realbkg_ring_chi_corr=CORR)))
        p, n = patchiness(sets[sd])
        print(f'texture sd {sd:4.2f} corr {CORR}: ring patchiness p50 {p:.3f} (n {n})   [real 41 0.106, legacy 0.031]',
              flush=True)
    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep})', flush=True)
        for sd in SDS:
            print(f'  texture sd {sd:4.2f}  {score(model, args, sets[sd])}', flush=True)
        del model
