"""Ring vs segment BOX HEIGHT in the simulator boxconv1 trained on.

Real 41 ring boxes track the valid chi span (p50 446 px, only 0.6% full height); the realbkg
simulator makes every ring exactly 512 px tall. This asks whether the legacy simulator -- the one
behind every 0.74+ score on 41 -- had the same flaw, which decides whether full-height ring boxes
are a REGRESSION in conv1/conv2 or a pre-existing error that cost nothing.
"""
import importlib.util, json, random, os
import numpy as np, torch

SNAP = ('/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/'
        'detector_runs/dino_boxconv1/simulation.py')
spec = importlib.util.spec_from_file_location('bcs', SNAP)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = json.load(open(os.path.join(os.path.dirname(SNAP), 'config_args_all.json')))

random.seed(11); np.random.seed(11); torch.manual_seed(11)
sc = m.SimulationConfig(); sc.a_coef, sc.w_coef = a['box_coef_override']
sim = m.FastSimulation(sim_config=sc, device='cpu')

rh, sh, n = [], [], 0
while n < 120:
    try:
        out = sim.simulate_img()
    except Exception:
        continue
    if out is None:
        continue
    img, bx, *rest = out
    b = np.asarray(bx, float)
    if len(b) == 0:
        continue
    rg = None
    for r in rest:
        v = np.asarray(r)
        if v.dtype == bool and v.ndim == 1 and len(v) == len(b):
            rg = v
            break
    if rg is None:
        continue
    n += 1
    h = b[:, 3] - b[:, 1]
    rh += list(h[rg]); sh += list(h[~rg])
    if n % 25 == 0:
        print(f'  {n} frames...', flush=True)

rh, sh = np.array(rh), np.array(sh)
p = lambda x, q: np.percentile(x, q) if len(x) else float('nan')
print(f'\nboxconv1 legacy sim, {n} frames: {len(rh)} ring boxes, {len(sh)} segment boxes')
print(f'  RING box height px   p10 {p(rh,10):6.0f}  p50 {p(rh,50):6.0f}  '
      f'p90 {p(rh,90):6.0f}  max {rh.max():6.0f}')
print(f'  ring height / 512    p10 {p(rh/512,10):6.2f}  p50 {p(rh/512,50):6.2f}  '
      f'p90 {p(rh/512,90):6.2f}')
print(f'  FULL height (>=505): {100*np.mean(rh>=505):.1f}%')
print(f'  SEG  box height px   p10 {p(sh,10):6.1f}  p50 {p(sh,50):6.1f}  p90 {p(sh,90):6.1f}')
print('\nreal 41:      ring p50 446 px, 0.6% full height, height/valid-span 0.98')
print('real organic: ring p50 512 px,  80% full height')
print('realbkg conv1/conv2: 100% of rings exactly 512 px')
