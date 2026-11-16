"""Side-by-side figures of several detectors on the same frames, in polar space (the network image, 1024 x 512) and in reciprocal space (raw q_xy / q_z image). CPU only.

  python heatmap/compare_figs.py --out <dir> --baseline "<label>=<dump.pkl>" [--baseline ...] --model "<label>=<dump.pkl>" [--model ...] [--sets organic 41] [--thr 0.3] [--top 5] [--frames organic:3 41:12 ...]
One figure per (frame, --model): columns = all --baseline models, then that model (the --model runs are NOT compared with each other); <out>/<model label>/<set>_frameNN.png + frame_stats.csv.
Each dump is a dump_dets.py pickle ({'sets': {name: [frame]}}, frame = img, mask, gt (already scaled to that model's training box convention), boxes, scores, cls; boxes in polar px, x = q axis (1024), y = angle 0..90 deg (512)).
Frames are in dataset order, so they are matched to the h5 file by index. Panel = image + that model's own (scaled) ground truth + detections with score > thr, coloured by correctness: GREEN solid = correct detection (TP: matched to a GT box of the same class, IoU >= 0.1), RED solid = wrong detection (FP), CYAN dashed = GT box that was missed (FN). The title gives TP / FP / FN counts.
Reciprocal panel: the polar box (x1..x2, y1..y2) is the annular sector r in [x1, x2] / 1024 * r_max, angle in [y1, y2] / 512 * 90 deg around the origin q = 0 (same transform the polar image was built with).
Frame selection: per frame and model a greedy one-to-one match (highest score first, same class, IoU>=0.1 in polar px; an approximation of the evaluation matching) of detections (score > thr) to the model's own GT gives P / R / F1; the discrepancy of a frame is the F1 spread (max - min) over the columns of that figure; the --top most discrepant frames of each set are drawn
(plus --frames given explicitly). <out>/frame_stats.csv has the per-frame numbers of all frames.
Env: HM_DATA_DIR (default ~/Documents/datasets/validationsSets/newbox_visfix)."""
import os, sys, argparse, pickle, csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA = os.environ.get('HM_DATA_DIR', os.path.expanduser('~/Documents/datasets/validationsSets/newbox_visfix'))
FILES = {'organic': 'organic_labeled.h5', '41': '41.h5'}
PW, PH = 1024, 512


def load_recip(ds):
    """-> list of (raw reciprocal image [N,N], q_max_axis_xy, q_max_axis_z) in dataset order (frame = one entry per key/frame)."""
    out = []
    with h5py.File(os.path.join(DATA, FILES[ds]), 'r') as f:
        for key in f.keys():
            d = f[key]['data']
            for i in range(len(d['img_gid_q'])):
                out.append((np.nan_to_num(d['img_gid_q'][i]).astype(np.float32), float(d['q_xy'][-1]), float(d['q_z'][-1])))
    return out


def is_ring_box(b, mask):
    rows = np.where(mask.any(axis=1))[0]
    span = (rows.max() - rows.min() + 1) if len(rows) else PH
    return (b[3] - b[1]) >= 0.7 * span


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(fr, thr):
    """greedy one-to-one match (highest score first, same class, IoU >= 0.1 in polar px) of detections with score > thr to the GT.
    -> (kept detection boxes, their classes, tp flags [n_det], missed-GT boxes [n_fn,4], n_gt)"""
    keep = np.asarray(fr['scores']) > thr
    b, s, c = np.asarray(fr['boxes'])[keep].reshape(-1, 4), np.asarray(fr['scores'])[keep], np.asarray(fr['cls'])[keep]
    g = np.asarray(fr['gt']).reshape(-1, 4)
    gr = np.array([is_ring_box(x, fr['mask']) for x in g], bool) if len(g) else np.zeros(0, bool)
    tp = np.zeros(len(b), bool); hit = np.zeros(len(g), bool)
    if len(b) and len(g):
        iou = iou_mat(b, g)
        for i in np.argsort(-s):
            cand = [k for k in np.argsort(-iou[i]) if iou[i, k] >= 0.1 and not hit[k] and bool(c[i] == 1) == bool(gr[k])]
            if cand:
                hit[cand[0]] = True; tp[i] = True
    return b, c, tp, g[~hit], len(g)


def prf(fr, thr):
    """-> (precision, recall, F1, n_det, n_gt, TP, FP, FN)"""
    b, c, tp, miss, ng = match(fr, thr)
    t = int(tp.sum()); nd = len(b)
    p, r = (t / nd if nd else 0.0), (t / ng if ng else 0.0)
    return (p, r, 2 * p * r / (p + r + 1e-9), nd, ng, t, nd - t, len(miss))


def sector(b, rmax_px, n=40):
    """polar box -> closed polygon (col, row) in reciprocal pixel coordinates (origin at the corner q = 0)."""
    r1, r2 = b[0] / PW * rmax_px, b[2] / PW * rmax_px
    p1, p2 = np.deg2rad(b[1] / PH * 90), np.deg2rad(b[3] / PH * 90)
    ph = np.linspace(p1, p2, n)
    arc2 = np.stack([r2 * np.cos(ph), r2 * np.sin(ph)], 1); arc1 = np.stack([r1 * np.cos(ph[::-1]), r1 * np.sin(ph[::-1])], 1)
    return np.concatenate([arc2, arc1, arc2[:1]], 0), arc2, arc1


TPC, FPC, FNC = 'lime', 'red', 'cyan'


def draw_polar(ax, fr, thr, title):
    img = np.asarray(fr['img'], np.float32)
    ax.imshow(img, cmap='gray', origin='upper', aspect='auto', extent=(0, PW, PH, 0))
    b, c, tp, miss, _ = match(fr, thr)
    for m in miss:
        ax.add_patch(Rectangle((m[0], m[1]), m[2] - m[0], m[3] - m[1], fill=False, ec=FNC, lw=1.0, ls='--'))
    for x, t in zip(b, tp):
        ax.add_patch(Rectangle((x[0], x[1]), x[2] - x[0], x[3] - x[1], fill=False, ec=TPC if t else FPC, lw=1.0))
    ax.set_xlim(0, PW); ax.set_ylim(PH, 0); ax.set_title(title, fontsize=7); ax.set_xticks([]); ax.set_yticks([])


def draw_recip(ax, rec, fr, thr):
    img, qx, qz = rec
    n_r, n_c = img.shape
    rmax_px = np.hypot(n_r - 1, n_c - 1)
    v = np.log1p(np.clip(img, 0, None)); hi = np.percentile(v[v > 0], 99.5) if (v > 0).any() else 1.0
    ax.imshow(v, cmap='gray', origin='lower', vmin=0, vmax=hi, extent=(0, qx, 0, qz), aspect='equal')
    sx, sz = qx / (n_c - 1), qz / (n_r - 1)
    def poly(bx, color, lw, ls, ring):
        full, a2, a1 = sector(bx, rmax_px)
        if ring:
            for arc in (a2, a1):
                ax.plot(arc[:, 0] * sx, arc[:, 1] * sz, color=color, lw=lw, ls=ls)
        else:
            ax.plot(full[:, 0] * sx, full[:, 1] * sz, color=color, lw=lw, ls=ls)
    b, c, tp, miss, _ = match(fr, thr)
    for m in miss:
        poly(m, FNC, 1.0, '--', is_ring_box(m, fr['mask']))
    for x, cl, t in zip(b, c, tp):
        poly(x, TPC if t else FPC, 1.0, '-', cl == 1)
    ax.set_xlim(0, qx); ax.set_ylim(0, qz); ax.set_xlabel('q_xy', fontsize=6); ax.set_ylabel('q_z', fontsize=6); ax.tick_params(labelsize=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--baseline', action='append', required=True, help='label=dump.pkl (left columns)')
    p.add_argument('--model', action='append', required=True, help='label=dump.pkl (one set of figures per model, last column)')
    p.add_argument('--sets', nargs='*', default=['organic', '41']); p.add_argument('--thr', type=float, default=0.3)
    p.add_argument('--top', type=int, default=5); p.add_argument('--frames', nargs='*', default=[], help='set:index')
    p.add_argument('--scale', type=float, default=1.0)
    a = p.parse_args()
    ld = lambda m: (m.split('=', 1)[0], pickle.load(open(m.split('=', 1)[1], 'rb'))['sets'])
    bases = [ld(m) for m in a.baseline]
    runs = [ld(m) for m in a.model]
    recips = {ds: load_recip(ds) for ds in a.sets}
    for rlab, rsets in runs:
        models = bases + [(rlab, rsets)]
        odir = os.path.join(a.out, ''.join(ch if ch.isalnum() or ch in '.-_' else '_' for ch in rlab)); os.makedirs(odir, exist_ok=True)
        rows, chosen = [], []
        for ds in a.sets:
            n = min(len(sets[ds]) for _, sets in models)
            stats = []
            for i in range(n):
                st = [prf(sets[ds][i], a.thr) for _, sets in models]
                spread = max(x[2] for x in st) - min(x[2] for x in st)
                stats.append((i, spread, st))
                rows.append([ds, i, round(spread, 3)] + [v for x in st for v in (round(x[0], 3), round(x[1], 3), round(x[2], 3), x[5], x[6], x[7])])
            pick = [i for i, _, _ in sorted(stats, key=lambda t: -t[1])[:a.top]]
            pick += [int(f.split(':')[1]) for f in a.frames if f.split(':')[0] == ds and int(f.split(':')[1]) not in pick]
            chosen += [(ds, i, next(sp for j, sp, _ in stats if j == i)) for i in pick]
        with open(os.path.join(odir, 'frame_stats.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['set', 'frame', 'F1_spread'] + [f'{lab}_{k}' for lab, _ in models for k in ('P', 'R', 'F1', 'TP', 'FP', 'FN')])
            w.writerows(rows)
        for ds, i, spread in chosen:
            k = len(models)
            fig, axs = plt.subplots(2, k, figsize=(4.4 * k * a.scale, 5.8 * a.scale), gridspec_kw=dict(height_ratios=[1, 1.15]))
            axs = np.asarray(axs).reshape(2, k)
            for c, (lab, sets) in enumerate(models):
                fr = sets[ds][i]; pr, rc, f1, nd, ng, tpn, fpn, fnn = prf(fr, a.thr)
                draw_polar(axs[0, c], fr, a.thr, f'{lab}\nTP {tpn} (green) | FP {fpn} (red) | FN {fnn} (cyan dashed) | GT {ng}\nP {pr:.2f} R {rc:.2f} F1 {f1:.2f}')
                draw_recip(axs[1, c], recips[ds][i], fr, a.thr)
            fig.suptitle(f'{ds} frame {i} | score > {a.thr} | green = correct detection, red = wrong detection (FP), cyan dashed = missed GT\neach panel against its own scaled GT | F1 spread {spread:.2f}', fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(os.path.join(odir, f'{ds}_frame{i:02d}.png'), dpi=110); plt.close(fig)
            print('wrote', rlab, ds, i, f'spread {spread:.3f}', flush=True)


if __name__ == '__main__':
    main()
