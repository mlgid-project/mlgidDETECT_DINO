"""Images of the heatmap detector on the REAL labeled frames (organic, 41). Runs without compiled DINO ops.

  python heatmap/visualize.py --ckpt <run_dir>/checkpoint.pth --out <dir> [--sets organic 41] [--thr 0.3]
         [--max_frames N] [--device cuda|cpu]       (--ckpt none = untrained net, for a plumbing test)

Per frame:  <out>/<set>/frame_XX.png  (top: polar image + boxes; bottom: predicted heatmap)
   boxes: GT green; matched prediction blue; false positive red; missed GT orange (native decode: peak picking, no NMS)
Per set:    <out>/<set>/closepairs.png  zoomed crops around GT peaks whose nearest GT neighbour is < 5 px away
   (top row: image + boxes, bottom row: heatmap with the picked peaks marked)
Env: HM_DATA_DIR (dir with organic_labeled.h5 / 41.h5), HM_BB_PATH (SimMIM weights if the run used them).
"""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):     # skip DINO package inits (compiled ops)
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from models.heatmap_head import HeatmapNet, decode
from heatmap import evaluation as E
from util.misc import clean_state_dict

GT_C, TP_C, FP_C, MISS_C = '#1baf7a', '#2a78d6', '#e34948', '#eda100'
SURFACE, INK = '#fcfcfb', '#0b0b0b'


def load(ckpt, device):
    if ckpt == 'none':
        return HeatmapNet(None, freeze_backbone=True, out_stride=2).to(device).eval()
    ck = torch.load(ckpt, map_location='cpu')
    a = ck['hm_args']
    model = HeatmapNet(None, freeze_backbone=a['freeze_backbone'], out_stride=a['out_stride'], amp_backbone=a.get('amp_backbone', False))
    model.load_state_dict(ck['model'], strict=False)
    if a.get('bb') != 'random' and a['freeze_backbone']:        # frozen backbone is not stored in the checkpoint
        bb = torch.load(os.environ.get('HM_BB_PATH', a['bb_path']), map_location='cpu')
        bb = clean_state_dict(bb.get('model', bb)); pre = a['bb_prefix']
        bb = {k[len(pre):]: v for k, v in bb.items() if k.startswith(pre) and 'head' not in k[len(pre):]}
        print('backbone:', model.backbone.load_state_dict(bb, strict=False), flush=True)
    return model.to(device).eval()


def add_boxes(ax, boxes, color, lw=0.9, ls='-'):
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))


def classify(gt, pb):
    """-> (matched pred idx, matched GT idx) with the repo's q-matcher"""
    if len(gt) == 0 or len(pb) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    _, ri, ci = E.matcher(torch.tensor(gt).float(), torch.tensor(pb).float())
    return np.asarray(ci, int), np.asarray(ri, int)


def frame_figure(img, heat, gt, pb, sc, ci, ri, thr, title, path):
    tp = np.zeros(len(pb), bool); tp[ci] = True
    miss = np.ones(len(gt), bool); miss[ri] = False
    fig, ax = plt.subplots(2, 1, figsize=(14, 14), facecolor=SURFACE)
    ax[0].imshow(img, cmap='gray', aspect='equal')
    add_boxes(ax[0], gt[~miss], GT_C, 1.6); add_boxes(ax[0], gt[miss], MISS_C, 1.6)
    add_boxes(ax[0], pb[tp], TP_C, 1.1, '--'); add_boxes(ax[0], pb[~tp], FP_C, 1.1)   # dashed on top of the GT so both stay visible
    ax[0].legend(handles=[mp.Patch(fc='none', ec=GT_C, label=f'GT found ({int((~miss).sum())})'),
                          mp.Patch(fc='none', ec=MISS_C, label=f'GT missed ({int(miss.sum())})'),
                          mp.Patch(fc='none', ec=TP_C, label=f'prediction, matched ({int(tp.sum())})'),
                          mp.Patch(fc='none', ec=FP_C, label=f'prediction, false positive ({int((~tp).sum())})')],
                 loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=4, fontsize=9, frameon=False)   # outside the image
    ax[0].set_title(f'{title}   (score > {thr}, native decode)', loc='left', color=INK)
    ax[1].imshow(heat, cmap='Blues', vmin=0, vmax=1, extent=(0, img.shape[1], img.shape[0], 0), aspect='equal')
    ax[1].set_title('predicted heatmap (max over classes)', loc='left', color=INK)
    for a_ in ax:
        a_.set_xlabel('q pixel'); a_.set_ylabel('chi pixel')
    fig.tight_layout(); fig.savefig(path, dpi=110, facecolor=SURFACE); plt.close(fig)


def closepair_figure(items, thr, title, path):
    n = len(items)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7.4), facecolor=SURFACE, squeeze=False)
    for k, it in enumerate(items):
        x0, x1, y0, y1 = it['win']
        ax[0][k].imshow(it['img'], cmap='gray', aspect='equal')
        ax[0][k].set_xlim(x0, x1); ax[0][k].set_ylim(y1, y0)
        add_boxes(ax[0][k], it['gt'], GT_C, 1.6); add_boxes(ax[0][k], it['pb'], TP_C, 1.8, '--')
        ax[1][k].imshow(it['heat'], cmap='Blues', vmin=0, vmax=1, extent=(0, it['img'].shape[1], it['img'].shape[0], 0))
        ax[1][k].set_xlim(x0, x1); ax[1][k].set_ylim(y1, y0)
        for (cx, cy) in it['peaks']:
            ax[1][k].plot(cx, cy, 'x', color=FP_C, ms=6)
        ax[0][k].set_title(it['label'], fontsize=8, color=INK)
        for r in (0, 1):
            ax[r][k].set_xticks([]); ax[r][k].set_yticks([])
    fig.suptitle(f'{title}: close pairs (GT green; predictions blue dashed, score > {thr}; heatmap peaks red x)', x=0.01,
                 ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=110, facecolor=SURFACE); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--sets', nargs='+', default=['organic', '41'])
    p.add_argument('--thr', type=float, default=0.3)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--n_crops', type=int, default=6)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model = load(a.ckpt, a.device)
    for ds in a.sets:
        os.makedirs(os.path.join(a.out, ds), exist_ok=True)
        crops = []
        for fi, (cfg, ic) in enumerate(E.iter_frames(E.DATASETS[ds])):
            if a.max_frames and fi >= a.max_frames:
                break
            with torch.no_grad():
                o = model(E.frame_inputs(ic, a.device))
            img = np.asarray(ic.converted_polar_image[0, 0])
            g = E.gt_of(ic); gt = g['gt']
            pb, sc, cls = [t.cpu().numpy() for t in decode(o, model.out_stride, 225)[0]]
            k = sc > cfg.POSTPROCESSING_SCORE
            pb_all, sc_all = pb[k], sc[k]
            sel = sc_all > a.thr
            pb_t = pb_all[sel]
            ci, ri = classify(gt, pb_t)
            heat = torch.nn.functional.interpolate(o['heat'].sigmoid().max(1, keepdim=True)[0], size=img.shape,
                                                   mode='nearest')[0, 0].cpu().numpy()
            frame_figure(img, heat, gt, pb_t, sc_all[sel], ci, ri, a.thr, f'{ds} frame {fi}',
                         os.path.join(a.out, ds, f'frame_{fi:02d}.png'))
            # close-pair crops: GT segments with a neighbour < 5 px (Euclid), ring-free
            eu, _, ring = E.nn_distances(gt, g['mask'])
            found = np.zeros(len(gt), bool); found[ri] = True
            used = []                                                  # one crop per cluster
            for i in np.where(eu < 5)[0]:
                cx, cy = (gt[i, 0] + gt[i, 2]) / 2, (gt[i, 1] + gt[i, 3]) / 2
                if any(abs(cx - ux) < 30 and abs(cy - uy) < 30 for ux, uy in used):
                    continue
                used.append((cx, cy))
                win = (cx - 25, cx + 25, cy - 25, cy + 25)
                near = [j for j in range(len(gt)) if win[0] < (gt[j, 0] + gt[j, 2]) / 2 < win[1]
                        and win[2] < (gt[j, 1] + gt[j, 3]) / 2 < win[3] and not ring[j]]
                nf = int(found[near].sum())
                pk = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in pb_all
                      if win[0] < (b[0] + b[2]) / 2 < win[1] and win[2] < (b[1] + b[3]) / 2 < win[3]]
                crops.append(dict(win=win, img=img, heat=heat, gt=gt[near], pb=pb_t, peaks=pk,
                                  label=f'f{fi}: {nf}/{len(near)} GT found, {len(pk)} peaks', nf=nf, n=len(near)))
            print(f'{ds} frame {fi}: GT {len(gt)}, pred>{a.thr} {len(pb_t)}, matched {len(ci)}', flush=True)
        if crops:
            rng = np.random.RandomState(0)
            pick = rng.choice(len(crops), min(a.n_crops, len(crops)), replace=False)
            closepair_figure([crops[j] for j in sorted(pick)], a.thr, ds, os.path.join(a.out, ds, 'closepairs.png'))
            print(f'{ds}: {len(crops)} close-pair GT candidates, drew {len(pick)} (fixed seed)', flush=True)


if __name__ == '__main__':
    main()
