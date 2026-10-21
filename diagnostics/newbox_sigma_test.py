"""Does box_sigma_halfwidth = (1, 1) really give boxes of +-1 sigma, for what the simulators DRAW?

2026-10-21: the validation boxes are centre +- stored_width/2 with stored width = 2 sigma, i.e. +-sigma.
main.py turns `box_sigma_halfwidth` (sigma units) into the simulators' a_coef = w_coef = 2.0. This checks:
  1. the new config merges as intended (box_sigma_halfwidth set, box_coef_override None, eval_files = visfix)
  2. LEGACY sim: build main.SimulationDataset's sim_config the way main.py does, take boxes of known size,
     render them with FastSimulation.img_from_labels, measure the drawn peak's sigma along q and chi
     (second moment of the profile) and compare with the box half-width -> ratio must be 1.00
  3. REALBKG sim: RealBkgSimulation._boxes at coef 2.0: half-width must equal s_q / s_c (the sigma that
     _render draws with)
CPU only, no bank.   python diagnostics/newbox_sigma_test.py
"""
import os, sys, random, types
import numpy as np
import torch
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
from util.slconfig import SLConfig

cfg = SLConfig.fromfile('config/DINO/DINO_4scale_swin_ssl_nb.py')
print('1. config: box_sigma_halfwidth', cfg.box_sigma_halfwidth, '| box_coef_override', cfg.box_coef_override)
print('   eval_files', dict(cfg.eval_files))
assert cfg.box_coef_override is None and tuple(cfg.box_sigma_halfwidth) == (1.0, 1.0)
for p in cfg.eval_files.values():
    assert os.path.exists(p), p

# the conversion exactly as main.SimulationDataset does it
_sig = cfg.box_sigma_halfwidth
a_coef, w_coef = 2.0 * float(_sig[0]), 2.0 * float(_sig[1])
print(f'   -> simulators get a_coef={a_coef} w_coef={w_coef}')

import simulation as SIM
sc = SIM.SimulationConfig(); sc.a_coef, sc.w_coef = a_coef, w_coef
sim = SIM.FastSimulation(sim_config=sc, device='cpu')
print('2. legacy sim: drawn sigma vs box half-width (profile second moment, one isolated peak)')
worst = 0.0
for hw, hh in ((3.0, 4.0), (5.0, 12.0), (8.0, 20.0), (2.0, 6.0)):
    x0, y0 = 400.0, 250.0
    boxes = torch.tensor([[x0 - hw, y0 - hh, x0 + hw, y0 + hh]])
    random.seed(0)                       # img_from_labels picks the profile power by random.random() > 0.1 -> 2
    while True:
        st = random.getstate()
        if random.random() > 0.1:
            random.setstate(st); break
    img = sim.img_from_labels(boxes, torch.tensor([1.0]), torch.tensor([False]))
    img = img.reshape(SIM.HEIGHT, SIM.WIDTH).numpy().astype(np.float64)
    prof_q = img[int(y0), :]; prof_c = img[:, int(x0)]
    def sigma(p, c):
        idx = np.arange(len(p)); w = np.clip(p, 0, None)
        return float(np.sqrt((w * (idx - c) ** 2).sum() / w.sum()))
    sq, sc_ = sigma(prof_q, x0), sigma(prof_c, y0)
    print(f'   box half-width q {hw:5.1f} chi {hh:5.1f} | drawn sigma q {sq:6.2f} chi {sc_:6.2f} | ratio q {sq/hw:.3f} chi {sc_/hh:.3f}')
    worst = max(worst, abs(sq / hw - 1), abs(sc_ / hh - 1))
assert worst < 0.03, worst
print('   legacy OK (max deviation %.3f)' % worst)

print('3. realbkg sim: _boxes half-width vs the sigma _render draws with')
import realbkg_simulation as RS
stub = types.SimpleNamespace(w_coef=w_coef, a_coef=a_coef, ring_w_coef=None, ring_box_from_mask=True)
s_q = np.array([2.0, 3.9, 7.0]); s_c = np.array([5.0, 20.3, 40.0])
x = np.array([100.0, 500.0, 900.0]); y = np.array([60.0, 250.0, 400.0])
bx = RS.RealBkgSimulation._boxes(stub, x, y, s_q, s_c, np.zeros(3, bool), np.ones((512, 1024), bool))
print('   half-width q', (bx[:, 2] - bx[:, 0]) / 2, 'vs s_q', s_q, '| chi', (bx[:, 3] - bx[:, 1]) / 2, 'vs s_c', s_c)
assert np.allclose((bx[:, 2] - bx[:, 0]) / 2, s_q) and np.allclose((bx[:, 3] - bx[:, 1]) / 2, s_c)
print('   realbkg OK')
print('NEWBOX SIGMA TEST OK')
