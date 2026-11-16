"""Why does the ring brightness in the PROCESSED image not follow the ring amplitude put into the simulator? Instruments the simulator's contrast step.

  python heatmap/debug_ring_chain.py [--gains 1 0.25 0.05] [--n 6] [--mode pygid|fitted]
For ring-only frames it captures the RAW composed image (before clip / log / histogram equalisation) and the processed image, and for the first rings of each frame prints:
  raw contrast   (ring-column mean - flank median) / flank sd, measured on the raw image over the valid rows
  raw / clip     the ring column's raw level relative to the 99.5th percentile of the frame (the clip level of the contrast chain)
  frac clipped   fraction of the ring column's rows at or above that clip level  (these become exactly white)
  processed      ring amplitude in the processed image (profile peak - flank median), and the share of rows with processed value >= 0.99
Needs the simulator (13 GB RAM, GPU small)."""
import os, sys, types, argparse, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gains', type=float, nargs='+', default=[1.0, 0.25, 0.05]); p.add_argument('--n', type=int, default=6)
    p.add_argument('--mode', default='fitted'); p.add_argument('--seed', type=int, default=0); p.add_argument('--contrast_min', type=float, default=0.3)
    a = p.parse_args()
    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    import realsim.realbkg_simulation as RS
    from realsim.factory import make_realbkg_sim
    cap = {}
    orig = RS.apply_contrast
    def wrapped(total, mask, chain):
        cap['raw'], cap['mask'] = np.array(total, copy=True), np.array(mask, copy=True)
        return orig(total, mask, chain)
    RS.apply_contrast = wrapped
    sim = make_realbkg_sim(device='cuda', overrides=dict(ring_sq_scale=0.5, ring_w_coef=4.0, ring_chi_texture=0.45, ring_chi_corr=(1.0, 64.0)))
    sim.amplitude_mode = a.mode; sim.single_structure = True; sim.p_ring = 1.0; sim.ring_contrast_min = a.contrast_min
    for g in a.gains:
        sim.ring_gain = g
        print(f'\n=== mode {a.mode}, ring_gain {g}', flush=True)
        got = 0
        while got < a.n:
            out = sim.simulate_img()
            if out is None:
                continue
            img, boxes, mask, is_ring = out
            img = img.float().cpu().numpy(); boxes = boxes.float().cpu().numpy(); ring = is_ring.cpu().numpy().astype(bool); m = cap['mask'].astype(bool); raw = cap['raw']
            got += 1
            clip = np.percentile(raw[m], 99.5)
            rows_out = []
            for b in boxes[ring][:5]:
                xc = int(round((b[0] + b[2]) / 2))
                if xc < 45 or xc > 978:
                    continue
                rows = m[:, xc - 40:xc + 41][:, 36:45].all(1)
                if rows.sum() < 40:
                    continue
                col = raw[rows][:, xc - 2:xc + 3].mean(1); flank = np.concatenate([raw[rows][:, xc - 40:xc - 25].ravel(), raw[rows][:, xc + 25:xc + 41].ravel()])
                rc = (col.mean() - np.median(flank)) / max(flank.std(), 1e-9)
                pr = img[rows][:, xc - 2:xc + 3].mean(1); pf = np.median(np.concatenate([img[rows][:, xc - 40:xc - 25].ravel(), img[rows][:, xc + 25:xc + 41].ravel()]))
                rows_out.append((rc, col.mean() / clip, float((col >= clip).mean()), float(pr.mean() - pf), float((pr >= 0.99).mean())))
            if rows_out:
                r = np.array(rows_out)
                print(f'  frame {got}: {len(boxes[ring])} rings; first {len(r)}: raw contrast {np.round(r[:, 0], 2).tolist()} | raw/clip {np.round(r[:, 1], 2).tolist()} | frac clipped {np.round(r[:, 2], 2).tolist()} | processed amp {np.round(r[:, 3], 2).tolist()} | share >=0.99 {np.round(r[:, 4], 2).tolist()}', flush=True)


if __name__ == '__main__':
    main()
