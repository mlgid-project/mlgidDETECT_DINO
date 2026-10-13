"""Frames of the real-background simulator in the dump_dets pickle format (img, mask, gt, gt_ring), so that ring_compare.py / probe tools can read them. Needs GPU + ~13 GB RAM.

  python heatmap/sim_pickle.py --out <file>.pkl --name <set name> --n 150 --cfg "key=val;key=val" [--init "mask_keep=[0,3,6,7];ring_x_share=(...)"]
cfg = attributes set after construction (as sheet_job.py), init = constructor options."""
import os, sys, types, argparse, ast, pickle, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--name', default='sim'); p.add_argument('--n', type=int, default=150)
    p.add_argument('--cfg', default=''); p.add_argument('--init', default=''); p.add_argument('--seed', type=int, default=5); p.add_argument('--preset', default='r6')
    a = p.parse_args()
    from realsim.factory import make_realbkg_sim
    init = {k: ast.literal_eval(v) for k, v in (x.split('=', 1) for x in a.init.split(';') if x)}
    sim = make_realbkg_sim(device='cuda', preset=a.preset, overrides=init)
    for item in [x for x in a.cfg.split(';') if x]:
        k, v = item.split('=', 1)
        if k.startswith('R_'):
            sim.R[k[2:]] = ast.literal_eval(v)
        else:
            setattr(sim, k, ast.literal_eval(v))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    fr = []
    while len(fr) < a.n:
        try:
            out = sim.simulate_img()
        except Exception:
            continue
        if out is None:
            continue
        img, boxes, mask, is_ring = out
        img = img if img.dim() == 3 else img[None]
        fr.append(dict(img=img[0].detach().cpu().numpy().astype(np.float16), mask=mask.reshape(img.shape[-2:]).cpu().numpy().astype(bool),
                       gt=boxes.float().cpu().numpy(), gt_ring=is_ring.cpu().numpy().astype(bool), gtconf=np.ones(len(boxes), np.float32),
                       boxes=np.zeros((0, 4), np.float32), scores=np.zeros(0, np.float32), cls=np.zeros(0, int)))
    pickle.dump(dict(run=a.name, epoch=-1, ckpt='', sets={a.name: fr}), open(a.out, 'wb'))
    print(f'wrote {a.out}: {a.name} {len(fr)} frames', flush=True)


if __name__ == '__main__':
    main()
