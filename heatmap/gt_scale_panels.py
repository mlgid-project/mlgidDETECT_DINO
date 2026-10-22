"""Images of the GT box scaling (heatmap/evaluation.py): real labeled frames with the raw GT (+-1 sigma) and the GT scaled by 1.25 (box_coef 2.5) and 1.5 (box_coef 3.0).
  python heatmap/gt_scale_panels.py --out <dir> [--frames 41:30,23 organic:3,10] ; env HM_DATA_DIR as for evaluation.py
Per set one figure: rows = frames, columns = full frame (raw GT green, x1.5 red) | zoom raw | zoom x1.25 | zoom x1.5 (zoom centred on the segment with most neighbours, ring boxes x only)."""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from heatmap import evaluation as E


def draw(ax, img, mask, boxes, color, lw, xlim=None, ylim=None, title=''):
    show = np.where(mask, img, np.nan)
    ax.imshow(show, cmap='gray', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_facecolor('#000040')
    for b in boxes:
        ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw))
    if xlim: ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--frames', nargs='+', default=['41:auto', 'organic:auto'], help="'set:i,j' frame indices or 'set:auto' = the frame with most segments and the one with most rings")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for spec in a.frames:
        ds, fl = spec.split(':')
        if fl == 'auto':
            cnt = []
            for fi, (cfg, ic) in enumerate(E.iter_frames(E.DATASETS[ds])):
                b = np.asarray(ic.polar_labels.boxes, np.float32); m = np.asarray(ic.converted_mask).reshape(*E.POLAR).astype(bool)
                rg = E.is_ring_geom(b, m) if len(b) else np.zeros(0, bool)
                cnt.append((int((~rg).sum()), int(rg.sum())))
            seg_order = np.argsort([-c[0] for c in cnt]); ring_order = np.argsort([-c[1] for c in cnt])
            want = [int(seg_order[0])] + [int(i) for i in ring_order if i != seg_order[0]][:1]
            print(ds, 'auto frames', want, [cnt[i] for i in want], '(segments, rings)', flush=True)
        else:
            want = [int(v) for v in fl.split(',')]
        rows = []
        for fi, (cfg, ic) in enumerate(E.iter_frames(E.DATASETS[ds])):
            if fi in want:
                img = np.asarray(ic.converted_polar_image[0, 0]); mask = np.asarray(ic.converted_mask).reshape(*E.POLAR).astype(bool)
                raw = np.asarray(ic.polar_labels.boxes, np.float32)
                sc = {}
                for name, c in (('x1.25 (box_coef 2.5)', 2.5), ('x1.5 (box_coef 3.0)', 3.0)):
                    E.set_gt_box_coef(c, c); sc[name] = E._scale_gt(raw, mask)
                E.set_gt_box_coef(None, None)
                rows.append((fi, img, mask, raw, sc))
            if len(rows) == len(want):
                break
        rows.sort(key=lambda r: want.index(r[0]))
        fig, axs = plt.subplots(len(rows), 4, figsize=(20, 4.6 * len(rows)), squeeze=False)
        for r, (fi, img, mask, raw, sc) in enumerate(rows):
            ring = E.is_ring_geom(raw, mask)
            cx = (raw[:, 0] + raw[:, 2]) / 2; cy = (raw[:, 1] + raw[:, 3]) / 2
            seg = np.where(~ring)[0]
            if len(seg):
                d = np.hypot(cx[seg][:, None] - cx[seg][None], cy[seg][:, None] - cy[seg][None])
                k = seg[(d < 40).sum(1).argmax()]; x0, y0 = cx[k], cy[k]
            else:
                x0, y0 = img.shape[1] / 2, img.shape[0] / 2
            xl = (max(x0 - 90, 0), min(x0 + 90, img.shape[1])); yl = (min(y0 + 70, img.shape[0]), max(y0 - 70, 0))
            s15 = sc['x1.5 (box_coef 3.0)']
            ax = axs[r][0]; draw(ax, img, mask, raw, '#1baf7a', 0.6, title=f'{ds} frame {fi}: raw GT green (+-1 sigma), x1.5 red')
            for b in s15: ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#e34948', lw=0.4))
            ax.add_patch(mp.Rectangle((xl[0], yl[1]), xl[1] - xl[0], yl[0] - yl[1], fill=False, ec='#eda100', lw=1.5))
            draw(axs[r][1], img, mask, raw, '#1baf7a', 1.2, xl, yl, 'raw GT (+-1 sigma, box_coef 2,2)')
            draw(axs[r][2], img, mask, sc['x1.25 (box_coef 2.5)'], '#2a78d6', 1.2, xl, yl, 'GT x1.25 (trained at 2.5,2.5)')
            draw(axs[r][3], img, mask, s15, '#e34948', 1.2, xl, yl, 'GT x1.5 (trained at 3.0,3.0)')
        fig.tight_layout(); fn = os.path.join(a.out, f'gt_scale_{ds}.png'); fig.savefig(fn, dpi=80); plt.close(fig); print('wrote', fn, flush=True)


if __name__ == '__main__':
    main()
