"""Are the realbkg models' extra 41 false positives FRAGMENTS of real rings?

gap41_split.py found the realbkg 41 gap is confidence (TP-FP score separation 0.11-0.19 vs the
baseline's 0.65), and that the FP excess sits in ring-heavy frames (precision loss vs ring fraction
Spearman -0.58 to -0.67) while only 9-16 FP/frame are themselves ring-shaped. The guess: rings get
broken into many short segment boxes. This tests it, read-only, at the deployed point:

  * every FP classified: ON A RING (centre inside a GT ring's radial span, +-2 px), NEXT TO A
    SEGMENT (IoU >= 0.1 with a GT segment), or BACKGROUND
  * on-ring FPs: chi height as a fraction of the valid chi span at their radius -- fragments are
    short, duplicates of the whole ring are ~1.0
  * boxes per GT ring: predictions whose centre lies inside the ring's radial span
  * TP score on rings vs on segments -- is the low confidence ring-specific?
  * figures of frames 33 and 39 (worst ring frames): GT | reference | each other model

  python diagnostics/ring_fp_probe.py name=/ckpt.pth [...]      (first target = reference)
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap41_split import (load_model, _config, is_ring_geom, iou_matrix, matcher, _P, DEV,
                         DATASETS, standard_preprocessing, H5GIWAXSDataset)
from util.postprocessing import onnx_to_xyxy, filter_boxes

FRAMES_TO_DRAW = [33, 39, 28]
OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/11_ring_fragments'


def cache41(model, args):
    cfg = _config(DATASETS['41'])
    ds = H5GIWAXSDataset(cfg, path=DATASETS['41'], preprocess_func=standard_preprocessing, buffer_size=5)
    fr = []
    for ic in ds.iter_images():
        x = np.asarray(ic.converted_polar_image[:, 0], np.float32)
        with torch.no_grad():
            o = model(torch.tensor(x).unsqueeze(0).to(DEV).repeat(1, args.num_channels, 1, 1))
        c = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), [o['pred_logits'].cpu().numpy(),
                                                       o['pred_boxes'].cpu().numpy()], num_select=225))
        fr.append(dict(img=x[0], mask=np.asarray(ic.converted_mask).reshape(512, 1024).astype(bool),
                       gt=np.asarray(ic.polar_labels.boxes, np.float32),
                       pb=np.asarray(c.boxes, np.float32).reshape(-1, 4),
                       sc=np.asarray(c.scores, np.float32)))
    return fr


def chi_frac(boxes, mask):
    out = []
    for x0, y0, x1, y1 in boxes:
        xc = int(np.clip((x0 + x1) / 2, 0, mask.shape[1] - 1))
        out.append((y1 - y0) / max(int(mask[:, xc].sum()), 1))
    return np.array(out)


def analyse(fr):
    A = dict(fp=0, on_ring=0, near_seg=0, bg=0, onring_chi=[], per_ring=[], tp_ring=[], tp_seg=[],
             fp_ring_sc=[], n=len(fr))
    for f in fr:
        gt, pb, sc, m = f['gt'], f['pb'], f['sc'], f['mask']
        ring = is_ring_geom(gt, m) if len(gt) else np.zeros(0, bool)
        if len(pb) and len(gt):
            _, ri, ci = matcher(torch.tensor(gt).float(), torch.tensor(pb).float())
            ri, ci = np.asarray(ri, int), np.asarray(ci, int)
        else:
            ri = ci = np.zeros(0, int)
        A['tp_ring'] += list(sc[ci][ring[ri]]); A['tp_seg'] += list(sc[ci][~ring[ri]])
        fp = np.setdiff1d(np.arange(len(pb)), ci)
        pc = (pb[:, 0] + pb[:, 2]) / 2 if len(pb) else np.zeros(0)
        rg = gt[ring]
        for g in rg:
            A['per_ring'].append(int(((pc >= g[0] - 2) & (pc <= g[2] + 2)).sum()))
        if not len(fp):
            continue
        A['fp'] += len(fp)
        onr = np.zeros(len(fp), bool)
        for g in rg:
            onr |= (pc[fp] >= g[0] - 2) & (pc[fp] <= g[2] + 2)
        seg = gt[~ring]
        nrs = (iou_matrix(pb[fp], seg).max(1) >= 0.1) if len(seg) else np.zeros(len(fp), bool)
        nrs &= ~onr
        A['on_ring'] += int(onr.sum()); A['near_seg'] += int(nrs.sum()); A['bg'] += int((~onr & ~nrs).sum())
        A['onring_chi'] += list(chi_frac(pb[fp][onr], m)); A['fp_ring_sc'] += list(sc[fp][onr])
    return A


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def draw(models, k):
    names = list(models)
    fig, ax = plt.subplots(len(names) + 1, 1, figsize=(14, 4.2 * (len(names) + 1)))
    f0 = models[names[0]][k]
    ring = is_ring_geom(f0['gt'], f0['mask']) if len(f0['gt']) else np.zeros(0, bool)
    panels = [('ground truth', f0['gt'], None, ring)] + \
             [(n, models[n][k]['pb'], models[n][k]['sc'], None) for n in names]
    for a, (title, bx, sc, rg) in zip(ax, panels):
        a.imshow(f0['img'], cmap='gray', aspect='auto', vmin=0, vmax=1)
        for i, b in enumerate(bx):
            col = (('#7CFC00' if rg[i] else '#00E5FF') if rg is not None
                   else plt.cm.plasma(float(np.clip(sc[i], 0, 1))))
            a.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec=col, lw=0.8))
        extra = (f'{len(bx)} boxes (green ring, cyan segment)' if rg is not None else
                 f'{len(bx)} predictions, colour = score (dark 0.1 -> yellow 1.0)')
        a.set_title(f'41 frame {k} - {title}: {extra}', fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(f'{OUT}/ringfp_frame{k:02d}.png', dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    os.makedirs(OUT, exist_ok=True)
    models = {}
    for name, ckpt in targets:
        model, args, ep = load_model(ckpt)
        models[name] = cache41(model, args)
        del model
        A = analyse(models[name]); n = A['n']
        print(f'\n===== {name} (epoch {ep}) on 41', flush=True)
        print(f'  FP/frame {A["fp"]/n:.1f}: ON A RING {A["on_ring"]/n:.1f} | next to a GT segment '
              f'{A["near_seg"]/n:.1f} | background {A["bg"]/n:.1f}')
        print(f'  on-ring FP chi height / valid span: p10 {q(A["onring_chi"],10):.2f} '
              f'p50 {q(A["onring_chi"],50):.2f} p90 {q(A["onring_chi"],90):.2f}   '
              f'(short = fragment, ~1 = duplicate ring)')
        print(f'  on-ring FP score: p50 {q(A["fp_ring_sc"],50):.3f} p90 {q(A["fp_ring_sc"],90):.3f}')
        pr = np.array(A['per_ring'])
        print(f'  predictions per GT ring: mean {pr.mean():.2f}  p50 {q(pr,50):.0f}  p90 {q(pr,90):.0f}  '
              f'max {pr.max()}  rings with >=3: {(pr >= 3).mean():.0%}')
        print(f'  TP score: rings p50 {q(A["tp_ring"],50):.3f} | segments p50 {q(A["tp_seg"],50):.3f}',
              flush=True)
    for k in FRAMES_TO_DRAW:
        draw(models, k)
    print(f'\nwrote ringfp_frame*.png to {OUT}')
