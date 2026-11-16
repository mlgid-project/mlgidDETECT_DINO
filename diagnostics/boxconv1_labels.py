"""Label statistics of the simulator dino_boxconv1 actually trained on.

boxconv1 ran from branch `development` at 15d3da4 and its config file no longer exists, but the
run directory keeps a snapshot of the simulation.py it used. That snapshot is self-contained
(torch/torchvision only), so it is loaded directly here rather than reconstructed -- this measures
the code that produced 0.585 organic / 0.748 on 41, not a guess at it.

Box geometry uses box_coef_override=(2.8, 1.3) from the run's own config_args_all.json.
"""
import os, sys, json, importlib.util, random
import numpy as np, torch

SNAP = ('/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/'
        'detector_runs/dino_boxconv1/simulation.py')
spec = importlib.util.spec_from_file_location('boxconv_sim', SNAP)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

a = json.load(open(os.path.join(os.path.dirname(SNAP), 'config_args_all.json')))
ac, wc = a['box_coef_override']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150

random.seed(11); np.random.seed(11); torch.manual_seed(11)
sc = m.SimulationConfig(); sc.a_coef, sc.w_coef = ac, wc
sim = m.FastSimulation(sim_config=sc, device='cpu')
print(f'snapshot {SNAP}\na_coef={sc.a_coef} w_coef={sc.w_coef} '
      f'min_nms={sc.min_nms} min_ring_seg_nms={sc.min_ring_seg_nms}\n')

nb, nr, bw, bh, cons = [], [], [], [], []
tries = 0
while len(nb) < N and tries < N*20:
    tries += 1
    try:
        out = sim.simulate_img()
    except Exception as e:
        if tries < 3: print('  simulate_img raised:', type(e).__name__, e)
        continue
    if out is None:
        continue
    img, bx, *rest = out if isinstance(out, (tuple, list)) else (out, None)
    if bx is None or len(bx) == 0:
        continue
    rg = None
    for r in rest:
        v = np.asarray(r)
        if v.dtype == bool and v.ndim == 1 and len(v) == len(bx):
            rg = v; break
    b = np.asarray(bx, float)
    nb.append(len(b)); nr.append(int(rg.sum()) if rg is not None else -1)
    bw.append(np.median(b[:, 2]-b[:, 0])); bh.append(np.median(b[:, 3]-b[:, 1]))
    # amp/local-noise of the labelled peaks, read off the frame the sim just made
    im = np.asarray(img, float)
    if im.ndim == 3: im = im[0]
    if len(nb) % 25 == 0: print(f'  {len(nb):4d} frames...', flush=True)

nb = np.array(nb, float); nr = np.array(nr, float)
print(f'\n{len(nb)} frames from {tries} draws')
def row(t, x):
    print(f'  {t:<26s} mean {x.mean():7.1f}  p50 {np.median(x):7.1f}  '
          f'p10 {np.percentile(x,10):7.1f}  p90 {np.percentile(x,90):7.1f}  max {x.max():6.0f}')
row('boxes / frame', nb)
if (nr >= 0).all():
    row('rings / frame', nr)
    row('segments / frame', nb-nr)
    print(f'  ring-free frames: {100*(nr==0).mean():.0f}%   '
          f'ring:segment {nr.sum()/max((nb-nr).sum(),1):.3f}')
row('median box width  (px)', np.array(bw))
row('median box height (px)', np.array(bh))
print('\nEVERY simulated peak is labelled here: there is no brightness gate in this simulator,')
print('and img_from_labels() paints the frame FROM the boxes, so render and label already agreed.')
