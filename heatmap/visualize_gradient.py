"""Side-by-side images: the fixed score cut (0.3) vs a q-dependent ("gradient") score cut, on the real labeled frames.
Does 0.3 cut off good boxes at high q while keeping the same kind of boxes at low q? (+nms decode only; nothing is overwritten)

  python heatmap/visualize_gradient.py --ckpt <run_dir>/final_checkpoint.pth --out <run_dir>/images [--t0 0.3 --t1 0.1] [--sets organic 41]

Writes <out>/nms_gradient_t0<t0>_t1<t1>/<set>/frame_XX_fixed_vs_gradient.png + summary.txt + README.txt. The existing <out>/nms/ images are untouched.
Per frame: LEFT fixed cut (score > 0.3), RIGHT gradient cut (score > thr(x), thr(x) = t0 + (t1 - t0) * x / 1024, x = box-centre column = q),
BOTTOM score-vs-q scatter of every detection with score > 0.05 (blue = matched a GT, red = false positive) with both cuts drawn:
points that sit BETWEEN the two lines are exactly the boxes the gradient adds. In the right panel those added boxes are violet (matched a GT)
or pink (false positive); boxes both cuts keep keep the usual colours (blue dashed = matched, red = false positive; GT green = found, orange = missed).
Env as visualize.py (HM_DATA_DIR, HM_BB_PATH)."""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from models.heatmap_head import decode
from heatmap import evaluation as E
from heatmap.visualize import load, classify, add_boxes, GT_C, TP_C, FP_C, MISS_C, SURFACE, INK

NEW_TP_C, NEW_FP_C = '#7b3fe4', '#ff5fa2'
W = 1024.0
THIRDS = [(0, 341.3), (341.3, 682.7), (682.7, 1024.1)]


def thr_x(t0, t1, b):
    cx = (b[:, 0] + b[:, 2]) / 2
    return t0 + (t1 - t0) * cx / W


def panel(ax, img, gt, pb, tp, ri, title, new=None):
    miss = np.ones(len(gt), bool); miss[ri] = False
    ax.imshow(img, cmap='gray', aspect='equal')
    add_boxes(ax, gt[~miss], GT_C, 1.6); add_boxes(ax, gt[miss], MISS_C, 1.6)
    old = np.ones(len(pb), bool) if new is None else ~new
    add_boxes(ax, pb[old & tp], TP_C, 1.1, '--'); add_boxes(ax, pb[old & ~tp], FP_C, 1.1)
    if new is not None:
        add_boxes(ax, pb[new & tp], NEW_TP_C, 1.8, '--'); add_boxes(ax, pb[new & ~tp], NEW_FP_C, 1.6)
    ax.set_title(title, loc='left', color=INK, fontsize=10)
    ax.set_xlabel('q pixel'); ax.set_ylabel('chi pixel')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--sets', nargs='+', default=['organic', '41'])
    p.add_argument('--fixed', type=float, default=0.3)
    p.add_argument('--t0', type=float, default=0.3); p.add_argument('--t1', type=float, default=0.1)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, ep = load(a.ckpt, a.device)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    root = os.path.join(a.out, f'nms_gradient_t0{a.t0:g}_t1{a.t1:g}'); os.makedirs(root, exist_ok=True)
    open(os.path.join(root, 'README.txt'), 'w').write(__doc__ + f'\nRun: {run} (epoch {ep}); fixed cut {a.fixed}; gradient {a.t0} at q=0 -> {a.t1} at q=1024.\n')
    for ds in a.sets:
        os.makedirs(os.path.join(root, ds), exist_ok=True)
        allgts, d_fix, d_grad, cfg0 = [], [], [], None
        third = np.zeros((3, 4), int)        # per q-third: extra TP, extra FP, GT found by fixed, GT found by gradient
        for fi, (cfg, ic) in enumerate(E.iter_frames(E.DATASETS[ds])):
            if a.max_frames and fi >= a.max_frames:
                break
            with torch.no_grad():
                x = (E.frame_inputs(ic, a.device) if model.chan_mode == 'he'
                     else E.frame_inputs(ic, a.device, chan_mode=model.chan_mode))
                o = model(x, E.frame_mask(ic, a.device))
            cfg.POSTPROCESSING_SCORE = 0.0; cfg0 = cfg
            pb, sc = E.heatmap_dets(cfg, decode(o, model.out_stride, 225)[0], use_nms=True)     # every detection after NMS
            g = E.gt_of(ic); gt = g['gt']; allgts.append(g)
            img = np.asarray(ic.converted_polar_image[0, 0])
            th = thr_x(a.t0, a.t1, pb) if len(pb) else np.zeros(0)
            kf = sc > a.fixed; kg = sc > th
            d_fix.append((pb[kf], sc[kf])); d_grad.append((pb[kg], sc[kg]))
            cf, rf = classify(gt, pb[kf]); cg, rg = classify(gt, pb[kg])
            tpf = np.zeros(kf.sum(), bool); tpf[cf] = True
            tpg = np.zeros(kg.sum(), bool); tpg[cg] = True
            new = ~kf[kg]                                       # kept by the gradient, not by the fixed cut
            # scatter: matching on every detection above 0.05
            ks = sc > 0.05
            cs, _ = classify(gt, pb[ks]); tps = np.zeros(ks.sum(), bool); tps[cs] = True
            fig = plt.figure(figsize=(24, 13.5), facecolor=SURFACE)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.28, wspace=0.08)
            ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
            panel(ax1, img, gt, pb[kf], tpf, rf, f'FIXED cut: score > {a.fixed}  |  GT found {len(rf)}/{len(gt)}, boxes {int(kf.sum())} '
                                                 f'(matched {int(tpf.sum())}, false positive {int((~tpf).sum())})')
            panel(ax2, img, gt, pb[kg], tpg, rg, f'GRADIENT cut: score > {a.t0} at low q -> {a.t1} at high q  |  GT found {len(rg)}/{len(gt)}, boxes {int(kg.sum())} '
                                                 f'(matched {int(tpg.sum())}, false positive {int((~tpg).sum())}); added by gradient: '
                                                 f'{int((new & tpg).sum())} matched, {int((new & ~tpg).sum())} false positive', new=new)
            sx = (pb[ks][:, 0] + pb[ks][:, 2]) / 2
            ax3.scatter(sx[tps], sc[ks][tps], s=14, c=TP_C, label='matched a GT (matching on all dets > 0.05)', zorder=3)
            ax3.scatter(sx[~tps], sc[ks][~tps], s=14, c=FP_C, label='false positive', zorder=2, alpha=0.8)
            xs = np.linspace(0, W, 50)
            ax3.axhline(a.fixed, color=INK, ls='--', lw=1.3, label=f'fixed cut {a.fixed}')
            ax3.plot(xs, a.t0 + (a.t1 - a.t0) * xs / W, color=NEW_TP_C, lw=2, label='gradient cut')
            ax3.set_xlim(0, W); ax3.set_ylim(0.04, 1.0); ax3.set_yscale('log')
            ax3.set_xlabel('q pixel (box centre)'); ax3.set_ylabel('score (log)')
            ax3.set_title('every detection after NMS with score > 0.05: points BETWEEN the two cut lines are what the gradient adds', loc='left', color=INK, fontsize=10)
            ax3.legend(loc='upper right', ncol=4, fontsize=9, frameon=False)
            handles = [mp.Patch(fc='none', ec=GT_C, label='GT found'), mp.Patch(fc='none', ec=MISS_C, label='GT missed'),
                       mp.Patch(fc='none', ec=TP_C, label='prediction, matched (both cuts)'), mp.Patch(fc='none', ec=FP_C, label='false positive (both cuts)'),
                       mp.Patch(fc='none', ec=NEW_TP_C, label='added by gradient, matched'), mp.Patch(fc='none', ec=NEW_FP_C, label='added by gradient, false positive')]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=6, fontsize=10, frameon=False)
            fig.suptitle(f'{run} (epoch {ep}) | {ds} frame {fi} | +nms decode', x=0.01, y=0.965, ha='left', fontsize=11, color=INK)
            fig.subplots_adjust(top=0.93, bottom=0.05, left=0.04, right=0.99)
            fig.savefig(os.path.join(root, ds, f'frame_{fi:02d}_fixed_vs_gradient.png'), dpi=85, facecolor=SURFACE); plt.close(fig)
            # per-third bookkeeping for the summary
            bx = (pb[kg][:, 0] + pb[kg][:, 2]) / 2
            for t, (lo, hi) in enumerate(THIRDS):
                m = (bx >= lo) & (bx < hi)
                third[t, 0] += int((new & tpg & m).sum()); third[t, 1] += int((new & ~tpg & m).sum())
            print(f'{ds} frame {fi}: fixed {kf.sum()} boxes ({tpf.sum()} matched) | gradient {kg.sum()} ({tpg.sum()} matched), added {int(new.sum())}', flush=True)
        rf_ = E.evaluate_dets(d_fix, allgts, cfg0, thr_list=(0.0,)); rg_ = E.evaluate_dets(d_grad, allgts, cfg0, thr_list=(0.0,))
        L = [f'{run} (epoch {ep}) | {ds} | +nms | fixed cut {a.fixed} vs gradient {a.t0} (q=0) -> {a.t1} (q=1024)', '']
        for nm, r in (('fixed', rf_), ('gradient', rg_)):
            t = r['thr'][0.0]
            L.append(f'{nm:9s} AP {r["ap"]:.4f}  recall {t["recall"]:.3f} (seg {t["seg_recall"]:.3f} ring {t["ring_recall"]:.3f})  precision {t["precision"]:.3f}  '
                     f'boxes {t["n_pred"]}  FP {t["fp"]}  NN<5px recall {t["euclid"]["<5"][1]:.3f} (n={t["euclid"]["<5"][0]})')
        L += ['', 'boxes added by the gradient, by q-third of the box centre (matched = would count as a found GT, false positive = extra FP):']
        for (lo, hi), row in zip(THIRDS, third):
            L.append(f'  q {int(lo):4d}-{min(int(hi),1024):4d}: +{row[0]} matched, +{row[1]} false positive')
        open(os.path.join(root, ds, 'summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
