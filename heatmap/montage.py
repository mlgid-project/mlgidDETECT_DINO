"""Contact sheet for judging a simulator by eye: real labelled frames (left column) next to simulated frames (right column), same processed image, ground-truth boxes drawn
(green segments, magenta rings). CPU only.

  python heatmap/montage.py --real <dump.pkl>:41 --real_frames 3 9 17 22 31 38 --sim <bench dir>/sim_frames.npz --sim_frames 0 1 2 3 4 5 --out sheet.png [--label "text"] [--scale 0.5]
--real: a dump_dets / probe_frames pickle and the set name (e.g. hm_runs/plots/dumps/dets_he_tf32_seed42.pkl:41); --sim: the sim_frames.npz written by heatmap/bench_sim.py --save N.
One row per pair, so the two lists must have the same length."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import cv2
from heatmap import evaluation as E


def draw(img, mask, boxes, ring, text):
    im = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    im[~mask] = (40, 0, 0)
    for b, r in zip(boxes, ring):
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255) if r else (0, 200, 0), 1)
    cv2.putText(im, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real', required=True); p.add_argument('--real_frames', type=int, nargs='+', required=True)
    p.add_argument('--sim', required=True); p.add_argument('--sim_frames', type=int, nargs='+', required=True)
    p.add_argument('--out', required=True); p.add_argument('--label', default='simulated'); p.add_argument('--scale', type=float, default=0.5)
    a = p.parse_args()
    assert len(a.real_frames) == len(a.sim_frames), 'the two frame lists must have the same length'
    path, ds = a.real.rsplit(':', 1)
    frames = pickle.load(open(path, 'rb'))['sets'][ds]
    z = np.load(a.sim)
    rows = []
    for rf, sf in zip(a.real_frames, a.sim_frames):
        f = frames[rf]
        img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool); gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
        ring = E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool)
        L = draw(img, mask, gt, ring, f'REAL {ds} f{rf}: {int(ring.sum())} rings, {int((~ring).sum())} segments')
        simg = z[f'{sf}_img']; sm = z[f'{sf}_mask'].astype(bool); sb = z[f'{sf}_boxes']; sr = z[f'{sf}_ring'].astype(bool)
        R = draw(simg, sm, sb, sr, f'{a.label} f{sf}: {int(sr.sum())} rings, {int((~sr).sum())} segments')
        rows.append(np.hstack([L, R]))
    sheet = np.vstack(rows)
    if a.scale != 1.0:
        sheet = cv2.resize(sheet, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, sheet); print('wrote', a.out, sheet.shape, flush=True)


if __name__ == '__main__':
    main()
