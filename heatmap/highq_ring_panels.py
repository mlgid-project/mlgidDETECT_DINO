"""Look at the high-q rings: right part of the frame (q column >= --x0) with ground truth (green segments, magenta rings) and, for real 41, the model's detections (red segments, yellow rings).
CPU only; input = dump_dets / sim_pickle pickles.
  python heatmap/highq_ring_panels.py --real <dump.pkl>:41 --out <dir> [--sim name=<pkl>:<set> ...] [--n 6] [--x0 560] [--thr 0.3] [--scale 0.8]
Real 41: the frames with the most unmatched GT rings at q column >= 781 (score > thr); one image per frame: left = image + GT, right = image + detections. Sim: the frames with the most rings at q column >= 781, GT only."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import cv2
import torch
from heatmap import evaluation as E


def panel(img, mask, boxes, ring, x0, scale, colors, label):
    base = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    base[~mask] = (60, 0, 0)
    for b, r in zip(boxes, ring):
        cv2.rectangle(base, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colors[int(bool(r))], 1)
    crop = base[:, x0:]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return crop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--real', required=True); p.add_argument('--out', required=True); p.add_argument('--sim', action='append', default=[])
    p.add_argument('--n', type=int, default=6); p.add_argument('--x0', type=int, default=560); p.add_argument('--thr', type=float, default=0.3); p.add_argument('--scale', type=float, default=0.8)
    p.add_argument('--xhi', type=float, default=781)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path, ds = a.real.rsplit(':', 1)
    frames = pickle.load(open(path, 'rb'))['sets'][ds]
    score = []
    for fi, f in enumerate(frames):
        gt = np.asarray(f['gt'], np.float32).reshape(-1, 4); mask = np.asarray(f['mask'], bool)
        if not len(gt):
            score.append(0); continue
        k = f['scores'] > a.thr; b = f['boxes'][k]
        hit = np.zeros(len(gt), bool)
        if len(b):
            _, ri, _ = E.matcher(torch.tensor(gt).float(), torch.tensor(b).float()); hit[np.asarray(ri, int)] = True
        ring = E.is_ring_geom(gt, mask); xc = (gt[:, 0] + gt[:, 2]) / 2
        score.append(int(((~hit) & ring & (xc >= a.xhi)).sum()))
    order = np.argsort(-np.array(score))[:a.n]
    print('real 41 frames with the most missed high-q rings:', [(int(i), score[i]) for i in order])
    for fi in order:
        f = frames[fi]; img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool); gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
        ring = E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool)
        k = f['scores'] > a.thr
        left = panel(img, mask, gt, ring, a.x0, a.scale, ((0, 200, 0), (255, 0, 255)), f'real 41 f{fi}: ground truth (magenta = ring)')
        right = panel(img, mask, f['boxes'][k], f['cls'][k] == 1, a.x0, a.scale, ((0, 0, 255), (0, 255, 255)), f'detections > {a.thr} (yellow = ring)')
        cv2.imwrite(os.path.join(a.out, f'real41_f{fi}.png'), np.concatenate([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right], 1))
    for s in a.sim:
        name, rest = s.split('=', 1); path, ds = rest.rsplit(':', 1)
        fr = pickle.load(open(path, 'rb'))['sets'][ds]
        sc = []
        for f in fr:
            gt = np.asarray(f['gt'], np.float32).reshape(-1, 4); mask = np.asarray(f['mask'], bool)
            ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else (E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool))
            xc = (gt[:, 0] + gt[:, 2]) / 2
            sc.append(int((ring & (xc >= a.xhi)).sum()) if len(gt) else 0)
        order = np.argsort(-np.array(sc))[:a.n]
        print(f'{name}: frames with most high-q rings:', [(int(i), sc[i]) for i in order])
        for fi in order:
            f = fr[fi]; img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool); gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
            ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else (E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool))
            cv2.imwrite(os.path.join(a.out, f'{name}_f{fi}.png'), panel(img, mask, gt, ring, a.x0, a.scale, ((0, 200, 0), (255, 0, 255)), f'{name} f{fi}: ground truth'))


if __name__ == '__main__':
    main()
