"""Predict boxes for UNLABELED h5 files from a heatmap checkpoint. No ground truth, no AP.

  python heatmap/predict.py --ckpt <run_dir>/final_checkpoint.pth --out <dir> file1.h5 [file2.h5 ...]
         [--floor 0.1] [--num_select 225] [--images] [--img_thr 0.3] [--max_frames N] [--device cuda|cpu]

Same decode as the headline evaluation (+nms): top-225 peaks, shared class-aware NMS, score floor 0.1 (--floor).
Input files: pyGID/NeXus layout (data/img_gid_q ...) loaded WITHOUT labels. roi_data-style files go through the
labeled loader (which reads the labels but they are ignored here).
Env as visualize.py: HM_BB_PATH (SimMIM weights if the run used them).

Writes <out>/<file stem>/
  boxes.csv   one row per box: frame (running index in the file), group (h5 group), nr (frame number in the group),
              x1,y1,x2,y2 (polar-image pixels, 512 x 1024: x = q axis 0..1024, y = chi axis 0..512), score, class (segment|ring)
  boxes.json  same content, grouped per frame, plus the checkpoint/floor settings
  images/frame_XXXX.png  (only with --images) polar image with boxes whose score > --img_thr (default 0.3)
Run name, checkpoint epoch and settings are repeated in <out>/README.txt."""
import os, sys, json, types, argparse, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from models.heatmap_head import decode
from heatmap import evaluation as E
from heatmap.visualize import load
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import filter_boxes

CLASSES = ['segment', 'ring']


def iter_unlabeled(path):
    cfg = E.make_config(path)
    if detect_dataset_type(path) == 'pygid':
        ds = PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=False)
    else:
        ds = H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5)
    try:
        for ic in ds.iter_images():
            yield cfg, ic
    finally:
        if callable(getattr(ds, 'close', None)):
            ds.close()


def detect(cfg, per_image):
    """top-225 peaks -> shared class-aware NMS + score floor; keeps the class label (E.heatmap_dets drops it)."""
    boxes, scores, cls = (t.cpu() for t in per_image)
    class _P: pass
    c = _P(); c.boxes, c.scores, c.pred_labels = boxes, scores, cls
    c = filter_boxes(cfg, c)
    return (np.asarray(c.boxes, np.float32).reshape(-1, 4), np.asarray(c.scores, np.float32),
            np.asarray(c.pred_labels, int).reshape(-1))


def draw(path, img, b, s, c, thr, title):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from heatmap.visualize import add_boxes, TP_C, FP_C, SURFACE, INK
    k = s > thr
    fig, ax = plt.subplots(figsize=(15, 7.5), facecolor=SURFACE)
    ax.imshow(img, cmap='gray', aspect='equal')
    add_boxes(ax, b[k & (c == 0)], TP_C, 1.1); add_boxes(ax, b[k & (c == 1)], FP_C, 1.1)
    ax.set_title(f'{title} | score > {thr}: {int((k & (c == 0)).sum())} segments (blue), {int((k & (c == 1)).sum())} rings (red)',
                 loc='left', color=INK, fontsize=10)
    ax.set_xlabel('q pixel'); ax.set_ylabel('chi pixel')
    fig.tight_layout(); fig.savefig(path, dpi=100, facecolor=SURFACE); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+'); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--floor', type=float, default=0.1); p.add_argument('--num_select', type=int, default=225)
    p.add_argument('--images', action='store_true'); p.add_argument('--img_thr', type=float, default=0.3)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, ep = load(a.ckpt, a.device)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, 'README.txt'), 'w').write(
        __doc__ + f'\n\nRun: {run} (epoch {ep}), ckpt {a.ckpt}\nDecode: top-{a.num_select} + class-aware NMS, score floor {a.floor}.\n')
    for path in a.files:
        stem = os.path.splitext(os.path.basename(path))[0]
        od = os.path.join(a.out, stem); os.makedirs(od, exist_ok=True)
        if a.images:
            os.makedirs(os.path.join(od, 'images'), exist_ok=True)
        rows, frames = [], []
        for fi, (cfg, ic) in enumerate(iter_unlabeled(path)):
            if a.max_frames and fi >= a.max_frames:
                break
            cfg.POSTPROCESSING_SCORE = a.floor
            with torch.no_grad():
                x = (E.frame_inputs(ic, a.device) if model.chan_mode == 'he'
                     else E.frame_inputs(ic, a.device, chan_mode=model.chan_mode))
                o = model(x, E.frame_mask(ic, a.device))
            b, s, c = detect(cfg, decode(o, model.out_stride, a.num_select)[0])
            grp, nr = getattr(ic, 'h5_group', ''), getattr(ic, 'nr', fi)
            for bb, ss, cc in zip(b, s, c):
                rows.append([fi, grp, nr, *[round(float(v), 2) for v in bb], round(float(ss), 4), CLASSES[cc]])
            frames.append(dict(frame=fi, group=grp, nr=int(nr), boxes=[dict(x1=r[3], y1=r[4], x2=r[5], y2=r[6], score=r[7], cls=r[8]) for r in rows if r[0] == fi]))
            if a.images:
                draw(os.path.join(od, 'images', f'frame_{fi:04d}.png'), np.asarray(ic.converted_polar_image[0, 0]), b, s, c, a.img_thr,
                     f'{run} (epoch {ep}) | {stem} frame {fi}')
            print(f'{stem} frame {fi}: {len(b)} boxes (score > {a.floor}), {int((s > 0.3).sum())} above 0.3', flush=True)
        with open(os.path.join(od, 'boxes.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['frame', 'group', 'nr', 'x1', 'y1', 'x2', 'y2', 'score', 'class']); w.writerows(rows)
        json.dump(dict(run=run, epoch=ep, ckpt=a.ckpt, floor=a.floor, num_select=a.num_select, frames=frames),
                  open(os.path.join(od, 'boxes.json'), 'w'))
        print(f'{stem}: {len(frames)} frames, {len(rows)} boxes -> {od}', flush=True)


if __name__ == '__main__':
    main()
