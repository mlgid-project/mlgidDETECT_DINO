#!/usr/bin/env python3
"""One frame of an h5 file through a heatmap checkpoint with a chosen contrast, for the labelling agent (mlgidAGENT/mlgid_tools.py run_detector).

  python heatmap/predict_frame.py --file <h5> --entry <group> --ckpt <final_checkpoint.pth> --out <boxes.json>
         [--clip LO HI | --clip none] [--log 0|1] [--gamma G] [--he 0|1] [--clahe LIMIT TILE_H TILE_W] [--floor 0.1] [--num_select 225] [--device cuda|cpu]

Contrast = util.exp_preprocess.apply_contrast on the raw polar image, the same pipeline as the deployed default: clip (5, 99.5), log, histogram equalisation.
Without contrast options the input is the one heatmap/predict.py feeds, and the output is identical to its boxes. Decode as predict.py: top-num_select, class-aware NMS, score floor.
Writes <out>: {"q_max", "default_diff", "boxes": [{radius, angle, radius_width, angle_width, score, is_ring}]} with q in 1/A and chi in deg (the units of detected_peaks).
default_diff is the largest difference between apply_contrast with the deployed settings and the deployed input, a check that the contrast code is the real one."""
import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'heatmap'))
import predict as P                                  # noqa: E402  (sets up the models package stub, imports the loaders)
from heatmap import evaluation as E                  # noqa: E402
from heatmap.visualize import load                   # noqa: E402
from models.heatmap_head import decode               # noqa: E402
from util.exp_preprocess import apply_contrast       # noqa: E402

POLAR_H, POLAR_W = E.POLAR
DEPLOYED = dict(clip=(5.0, 99.5), log=True, gamma=None, he=True, clahe=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True); p.add_argument('--entry', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--clip', nargs='+', default=None); p.add_argument('--log', type=int, default=1); p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--he', type=int, default=1); p.add_argument('--clahe', nargs=3, type=float, default=None)
    p.add_argument('--floor', type=float, default=0.1); p.add_argument('--num_select', type=int, default=225)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    s = dict(clip=DEPLOYED['clip'] if a.clip is None else None if a.clip == ['none'] else (float(a.clip[0]), float(a.clip[1])),
             log=bool(a.log), gamma=a.gamma, he=bool(a.he), clahe=None if a.clahe is None else (a.clahe[0], int(a.clahe[1]), int(a.clahe[2])))
    model, _ = load(a.ckpt, a.device)
    for cfg, ic in P.iter_unlabeled(a.file):
        if getattr(ic, 'h5_group', '').strip('/') != a.entry:
            continue
        cfg.POSTPROCESSING_SCORE = a.floor
        raw = np.asarray(ic.raw_polar_image, np.float32).reshape(POLAR_H, POLAR_W)
        mask = np.asarray(ic.converted_mask).reshape(POLAR_H, POLAR_W).astype(bool)
        default_diff = float(np.abs(apply_contrast(raw, mask, DEPLOYED) - np.asarray(ic.converted_polar_image[0, 0])).max())
        img = apply_contrast(raw, mask, s)
        x = torch.tensor(img)[None, None].to(a.device)
        with torch.no_grad():
            o = model(x, E.frame_mask(ic, a.device))
        b, sc, cl = P.detect(cfg, decode(o, model.out_stride, a.num_select)[0])
        import h5py
        with h5py.File(a.file, 'r') as h:
            d = h[a.entry]['data']
            q_max = float(np.hypot(d['q_xy'][()].max(), d['q_z'][()].max()))
        boxes = [dict(radius=float((x1 + x2) / 2 / POLAR_W * q_max), radius_width=float((x2 - x1) / POLAR_W * q_max), angle=float((y1 + y2) / 2 * 90 / POLAR_H),
                      angle_width=float((y2 - y1) * 90 / POLAR_H), score=float(v), is_ring=bool(c == 1)) for (x1, y1, x2, y2), v, c in zip(b, sc, cl)]
        json.dump(dict(q_max=q_max, default_diff=default_diff, contrast={k: v for k, v in s.items()}, boxes=boxes), open(a.out, 'w'))
        print(f'{a.entry}: {len(boxes)} boxes, {sum(bx["score"] > 0.3 for bx in boxes)} above 0.3, deployed-contrast check {default_diff:.2e}', flush=True)
        return
    sys.exit(f'no entry {a.entry} in {a.file}')


if __name__ == '__main__':
    main()
