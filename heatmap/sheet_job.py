"""Contact sheets WITHOUT colorbox data: real 41 frames (read from 41.h5, ground-truth boxes) next to frames of the real-background simulator, for several settings, in one process.

  python heatmap/sheet_job.py --out <dir> --cfg "name:key=val;key=val" [--cfg ...] [--real_frames 30 23 15 34 12 20] [--n 6] [--seed 5] [--scale 0.5]
Setting syntax as heatmap/calibrate_sim.py (attributes of RealBkgSimulation, R_<key> for the measured-statistics dict). Writes <dir>/sheet_<name>.png (left: real 41, right: simulated, rows = pairs;
green = segment box, magenta = ring box) and <dir>/<name>_stats.txt (boxes/frame etc.). Needs GPU (the simulator renders with torch) and ~13 GB RAM."""
import os, sys, types, argparse, ast, copy, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import cv2
import torch
from heatmap import evaluation as E


def draw_frame(img, mask, boxes, ring, text):
    im = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    im[~mask] = (40, 0, 0)
    for b, r in zip(boxes, ring):
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255) if r else (0, 200, 0), 1)
    cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--cfg', action='append', required=True); p.add_argument('--real_frames', type=int, nargs='+', default=[30, 23, 15, 34, 12, 20])
    p.add_argument('--n', type=int, default=6); p.add_argument('--seed', type=int, default=5); p.add_argument('--scale', type=float, default=0.5); p.add_argument('--preset', default='r6')
    p.add_argument('--init', default='', help="constructor options that cannot be set afterwards, e.g. \"mask_keep=[0,3,6,7]\"")
    a = p.parse_args()
    import random
    os.makedirs(a.out, exist_ok=True)
    real = []
    wanted = set(a.real_frames)
    for i, (cfg, ic) in enumerate(E.iter_frames(E.DATASETS['41'])):
        if i in wanted:
            g = E.gt_of(ic)
            mask = g['mask']; gt = g['gt']
            real.append((i, np.asarray(ic.converted_polar_image[0, 0], np.float32), mask, gt, E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool)))
    real.sort(key=lambda r: a.real_frames.index(r[0]))
    from realsim.factory import make_realbkg_sim
    init = {k: ast.literal_eval(v) for k, v in (x.split('=', 1) for x in a.init.split(';') if x)}
    sim = make_realbkg_sim(device='cuda', preset=a.preset, overrides=init)
    base = {k: copy.deepcopy(getattr(sim, k)) for k in dir(sim) if not k.startswith('_') and isinstance(getattr(sim, k), (int, float, str, bool, tuple, type(None), dict)) and k not in ('device',)}
    for spec in a.cfg:
        name, _, kv = spec.partition(':')
        for k, v in base.items():
            setattr(sim, k, copy.deepcopy(v))
        for item in [x for x in kv.split(';') if x]:
            k, v = item.split('=', 1)
            if k.startswith('R_'):
                sim.R[k[2:]] = ast.literal_eval(v)
            else:
                setattr(sim, k, ast.literal_eval(v))
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        rows, nb, nr = [], [], []
        while len(rows) < min(a.n, len(real)):
            try:
                out = sim.simulate_img()
            except Exception:
                continue
            if out is None:
                continue
            img, boxes, mask, is_ring = out
            img = img.float().cpu().numpy(); mask = mask.cpu().numpy().astype(bool); boxes = boxes.float().cpu().numpy(); ring = is_ring.cpu().numpy().astype(bool)
            nb.append(len(boxes)); nr.append(int(ring.sum()))
            ri, rimg, rmask, rgt, rring = real[len(rows)]
            L = draw_frame(rimg, rmask, rgt, rring, f'REAL 41 f{ri}: {int(rring.sum())} rings, {int((~rring).sum())} segments')
            R = draw_frame(img, mask, boxes, ring, f'{name} sim: {int(ring.sum())} rings, {int((~ring).sum())} segments')
            rows.append(np.hstack([L, R]))
        sheet = np.vstack(rows)
        if a.scale != 1.0:
            sheet = cv2.resize(sheet, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(a.out, f'sheet_{name}.png'), sheet)
        open(os.path.join(a.out, f'{name}_stats.txt'), 'w').write(f'{name}: {kv}\nframes drawn {len(nb)}, boxes/frame {np.mean(nb):.1f}, ring boxes/frame {np.mean(nr):.1f}\n')
        print(f'wrote sheet_{name}.png ({sheet.shape}) boxes/frame {np.mean(nb):.1f} rings/frame {np.mean(nr):.1f}', flush=True)


if __name__ == '__main__':
    main()
