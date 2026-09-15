"""Does the physics sim RENDER peaks it does not LABEL?

On its own simulator, where GT is exact, physics4 keeps 96.3 boxes for 57.6 GT objects --
precision 0.507, ~39 false positives per frame against ground truth the simulator itself wrote.
lr4e5 on its own sim does not do this (46.8 boxes for 44.7 GT, precision 0.954). Either the model
is badly wrong, or the simulator renders signal it leaves unlabeled -- which would teach the model
to fire on unlabeled structure, and that habit would transfer to real data as the near-miss spray.

For every predicted box, measures mean image intensity inside it, against a CONTROL box of the
same size at the same RADIUS but a random angle (intensity falls off strongly with q, so the
control has to be radius-matched). Buckets: matched TPs, unmatched FPs, control.

If FP intensity ~ control, the model is firing on empty detector -- a model problem.
If FP intensity >> control and approaches TP intensity, the sim is under-labeling -- a sim bug.

Writes the worst offenders as PNGs: GT in lime, unmatched high-intensity predictions in red.

RESULT 2026-09-15: REJECTED, the physics sim does not under-label. Unmatched FPs
touching no GT have median intensity 0.633, BELOW a mask-aware radius-matched control at 0.699
(matched TPs 0.755). The own-sim FP split is dup 4.0 / near-miss 33.4 / background 8.0 per frame --
74% near-misses, the same duplicate spray seen on real data, so the hedging is present on the
training distribution itself.

CAUTION, and the reason the control is mask-aware: the first version sampled the control angle over
the full frame height. That lands in the dark-area wedge ~45% of the time, deflated the control to
0.542 and produced a FALSE POSITIVE (FP 0.834 > TP 0.772 > control 0.542) that the rendered frames
then contradicted. Always mask-restrict a background control in polar frames.
"""
import os, sys
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, _eval_config, _P
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

OUT = os.environ.get('UNDERLABEL_OUT', '/mnt/lustre/work/schreiber/szb389/tmp_diag/underlabel')
N = int(os.environ.get('FRAMES', '120'))
NPNG = int(os.environ.get('NPNG', '6'))
matcher = get_matcher('q', min_iou=0.1)


def box_mean(img, b):
    H, W = img.shape[-2:]
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, x1 = max(0, min(x0, W - 1)), max(1, min(x1, W))
    y0, y1 = max(0, min(y0, H - 1)), max(1, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return np.nan
    return float(img[y0:y1, x0:x1].mean())


def control_mean(img, b, rng, valid):
    """Same size, same radius (x), random angle (y) -- but only among angles that are INSIDE the
    valid mask at that radius. Sampling y uniformly over the full height lands in the dark-area
    wedge ~45% of the time and deflates the control to meaninglessness."""
    h = max(1, int(round(b[3] - b[1])))
    xc = int(np.clip((b[0] + b[2]) / 2, 0, valid.shape[1] - 1))
    rows = np.flatnonzero(valid[:, xc])
    if len(rows) <= h:
        return np.nan
    y0 = int(rng.choice(rows[:len(rows) - h]))
    return box_mean(img, [b[0], y0, b[2], y0 + h])


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    name, ckpt = sys.argv[1].split('=', 1)
    model, args, epoch = load_model(ckpt)
    from main import SimulationDataset
    ds = SimulationDataset(args)
    sim = ds.physics if ds.physics is not None else ds.simulation
    config = _eval_config('/dev/null')
    rng = np.random.default_rng(0)

    tp_v, fp_v, ct_v, bg_v, frames = [], [], [], [], []
    split = {'dup': 0, 'near': 0, 'bg': 0}
    for _ in range(N):
        while True:
            try:
                img, gt, mask, is_ring = sim.simulate_img()
                if img is not None and torch.isfinite(img).all() and len(gt):
                    break
            except Exception:
                pass
        x = img.repeat(args.num_channels, 1, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            o = model(x)
        c = filter_boxes(config, onnx_to_xyxy(config, _P(),
              [o['pred_logits'].cpu().numpy(), o['pred_boxes'].cpu().numpy()], num_select=225))
        t = torch.tensor(gt.cpu().numpy()).float()
        _, row, col = matcher(t, c.boxes)
        im = img.cpu().numpy()
        im = im[0] if im.ndim == 3 else im
        pb = c.boxes.cpu().numpy(); sc = np.asarray(c.scores)
        ismatched = np.zeros(len(pb), bool); ismatched[np.asarray(col, int)] = True
        vals = np.array([box_mean(im, b) for b in pb])
        vmask = mask.cpu().numpy()
        vmask = vmask[0] if vmask.ndim == 3 else vmask
        vmask = vmask.astype(bool)
        ctrl = np.array([control_mean(im, b, rng, vmask) for b in pb])
        tp_v += list(vals[ismatched]); fp_v += list(vals[~ismatched]); ct_v += list(ctrl)
        from torchvision.ops import box_iou
        if (~ismatched).any():
            iou = box_iou(torch.tensor(pb[~ismatched]).float(), t).max(1).values.numpy()
            split['dup'] += int((iou > 0.3).sum())
            split['near'] += int(((iou > 0) & (iou <= 0.3)).sum())
            split['bg'] += int((iou == 0).sum())
            bgv = vals[~ismatched][iou == 0]
            bg_v.extend(bgv[np.isfinite(bgv)])
        # rank frames by how many BRIGHT unmatched boxes they carry
        thr = np.nanmedian(vals[ismatched]) if ismatched.any() else np.inf
        bright = (~ismatched) & (vals >= thr) & (sc > 0.3)
        frames.append((int(bright.sum()), im, gt.cpu().numpy(), pb[bright], sc[bright]))

    n_fr = len(frames)
    print(f'  FP split: dup(IoU>0.3) {split["dup"]/n_fr:.1f}/fr   '
          f'near(0<IoU<=0.3) {split["near"]/n_fr:.1f}/fr   bg(IoU=0) {split["bg"]/n_fr:.1f}/fr', flush=True)
    for tag, v in (('matched TP', tp_v), ('unmatched FP', fp_v),
                   ('  of those, bg-only FP', bg_v), ('control (mask-aware)', ct_v)):
        v = np.array(v); v = v[np.isfinite(v)]
        print(f'  {tag:>26}: n={len(v):6d}  median {np.median(v):.4f}  '
              f'p25 {np.percentile(v,25):.4f}  p75 {np.percentile(v,75):.4f}', flush=True)

    frames.sort(key=lambda f: -f[0])
    for i, (nb, im, gt, pb, sc) in enumerate(frames[:NPNG]):
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.imshow(im, cmap='gray', origin='lower', aspect='auto')
        for b in gt:
            ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, ec='lime', lw=1.0))
        for b, s in zip(pb, sc):
            ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                   fill=False, ec='red', lw=1.2))
            ax.text(b[0], b[3], f'{s:.2f}', color='red', fontsize=6)
        ax.set_title(f'{name} on its own sim -- lime = GT ({len(gt)}), '
                     f'red = unmatched prediction, score>0.3, brighter than the median TP ({nb})')
        fig.savefig(f'{OUT}/{name}_frame{i}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
    print(f'  wrote {min(NPNG, len(frames))} PNGs to {OUT}', flush=True)
