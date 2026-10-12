"""Pictures for looking: ground truth and a model's detections on the same frame, from a dump_dets / probe_frames pickle. CPU only.

  python heatmap/overlay_dump.py <dump.pkl> --set 41 --frames 30 23 15 13 34 --out <dir> [--thr 0.3]
Writes <dir>/<run>_<set>_f<N>.png: left panel = image + ground truth (green segments, magenta rings), right panel = image + detections with score > thr
(red = segment class, yellow = ring class). The image is the processed (HE) image the network sees, 1024 x 512, both panels stacked vertically."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import cv2
from heatmap import evaluation as E


def main():
    p = argparse.ArgumentParser(); p.add_argument('dump'); p.add_argument('--set', required=True); p.add_argument('--frames', type=int, nargs='+', required=True)
    p.add_argument('--out', required=True); p.add_argument('--thr', type=float, default=0.3)
    a = p.parse_args()
    d = pickle.load(open(a.dump, 'rb')); os.makedirs(a.out, exist_ok=True)
    for fi in a.frames:
        f = d['sets'][a.set][fi]
        img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool)
        base = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        base[~mask] = (40, 0, 0)
        gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
        ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else (E.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool))
        L = base.copy(); R = base.copy()
        for b, r in zip(gt, ring):
            cv2.rectangle(L, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255) if r else (0, 200, 0), 1)
        sel = f['scores'] > a.thr
        for b, c, s in zip(f['boxes'][sel], f['cls'][sel], f['scores'][sel]):
            cv2.rectangle(R, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255) if c == 1 else (0, 0, 255), 1)
        cv2.putText(L, f'{d["run"]} {a.set} f{fi}: ground truth ({int(ring.sum())} rings magenta, {int((~ring).sum())} segments green)', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(R, f'detections score>{a.thr}: {int((f["cls"][sel] == 1).sum())} ring class (yellow), {int((f["cls"][sel] == 0).sum())} segment class (red)', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out = os.path.join(a.out, f'{d["run"]}_{a.set}_f{fi}.png')
        cv2.imwrite(out, np.vstack([L, R])); print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
