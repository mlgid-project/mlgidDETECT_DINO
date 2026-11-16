"""Are run 7's low SEGMENT scores on 41 caused by segments that sit on rings?

ring_fp_probe.py (2026-10-08): run 7's ring fragmentation fell to legacy level at epoch 18 (1.83 boxes
per ring vs 1.75) but its TP score on 41 SEGMENTS is 0.38 (ep 18) / 0.51 (ep 73) against 0.88 legacy,
0.77 run 6, 0.73 conv3. r7 drops 88% of the sim's segments whose box overlaps a ring radially
(realbkg_seg_on_ring_keep 0.12); real 41 has such segments. Hypothesis: they are foreign to run 7.

Per model on real 41, every GT segment is classed by whether its box overlaps a GT ring box radially
(x-ranges intersect) or is isolated, and recall + TP score are reported per class. On the r7 sim the
same is done for run 7 with seg_on_ring_keep 0.12 (as trained) vs 1.0 (segments on rings allowed):
if run 7 scores segments-on-rings low only where they were rare in training, the hypothesis holds.
The share of segments on rings is printed for real 41, legacy, run 6 sim and r7 sim.

  python diagnostics/seg_context_probe.py name=/ckpt.pth [...]
"""
import os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
import peak_shape_compare as PS
from ring_fp_probe import cache41
from util.postprocessing import onnx_to_xyxy, filter_boxes

N = int(os.environ.get('SEGCTX_FRAMES', 60))
R7 = 'config/DINO/DINO_4scale_swin_realbkg_r7.py'
BASE = dict(realbkg_ring_min_gap=8.0, realbkg_rings_cap=(2, 60), realbkg_mosaic_seed=1234)


def seg_classes(gt, mask):
    """(is_ring, seg_on_ring) per GT box: a segment is on a ring if its x-range intersects a ring's."""
    ring = G.is_ring_geom(gt, mask) if len(gt) else np.zeros(0, bool)
    on = np.zeros(len(gt), bool)
    for i in np.where(~ring)[0]:
        for r in gt[ring]:
            if min(gt[i, 2], r[2]) > max(gt[i, 0], r[0]):
                on[i] = True
                break
    return ring, on


def predict(model, args, img, cfg):
    with torch.no_grad():
        o = model(torch.tensor(img)[None, None].to(G.DEV).repeat(1, args.num_channels, 1, 1))
    c = filter_boxes(cfg, onnx_to_xyxy(cfg, G._P(), [o['pred_logits'].cpu().numpy(),
                                                   o['pred_boxes'].cpu().numpy()], num_select=225))
    return np.asarray(c.boxes, np.float32).reshape(-1, 4), np.asarray(c.scores, np.float32)


def tally(items):
    """items: list of (gt, mask, pb, sc). Per segment class: n, recall, TP score p50/p25, and the share."""
    acc = {k: dict(n=0, hit=0, sc=[]) for k in ('on ring', 'isolated')}
    for gt, m, pb, sc in items:
        if not len(gt):
            continue
        ring, on = seg_classes(gt, m)
        matched = {}
        if len(pb):
            _, ri, ci = G.matcher(torch.tensor(gt).float(), torch.tensor(pb).float())
            matched = dict(zip(np.asarray(ri, int).tolist(), np.asarray(ci, int).tolist()))
        for i in np.where(~ring)[0]:
            a = acc['on ring' if on[i] else 'isolated']
            a['n'] += 1
            if i in matched:
                a['hit'] += 1; a['sc'].append(float(sc[matched[i]]))
    return acc


def fmt(acc):
    n = acc['on ring']['n'] + acc['isolated']['n']
    out = []
    for k in ('on ring', 'isolated'):
        a = acc[k]
        s = a['sc']
        out.append(f"{k:<8} n {a['n']:4d} ({100*a['n']/max(n,1):4.1f}%) recall {a['hit']/max(a['n'],1):.2f} "
                   f"TP score p50 {np.median(s) if s else float('nan'):.2f} p25 "
                   f"{np.percentile(s, 25) if s else float('nan'):.2f}")
    return '\n    '.join(out)


def share(tag, frames):
    c = np.zeros(2)
    for img, m, gt in frames:
        if len(gt):
            ring, on = seg_classes(gt, m)
            c += [on.sum(), (~ring).sum()]
    print(f'  {tag:<22} segments {int(c[1]):5d} | on a ring {100*c[0]/max(c[1],1):4.1f}%', flush=True)


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    cfg = G._config(G.DATASETS['41'])

    print('=== share of GT segments whose box overlaps a ring box (x-range)', flush=True)
    sets = {}
    # SEGCTX_SETS="name|key=val,key=val;name2|..." overrides the default r7 variants (values are floats)
    spec = os.environ.get('SEGCTX_SETS')
    if spec:
        variants = {}
        for item in spec.split(';'):
            nm, kv = item.split('|')
            variants[nm] = {k: float(v) for k, v in (e.split('=') for e in kv.split(',') if e)}
    else:
        variants = {'r7 keep 0.12': {}, 'r7 keep 1.0': dict(realbkg_seg_on_ring_keep=1.0)}
    for nm, ov in variants.items():
        random.seed(11); np.random.seed(11); torch.manual_seed(11)
        sets[nm] = list(PS.realbkg6(N, config=R7, overrides=dict(BASE, **ov)))
    if not spec:
        random.seed(11); np.random.seed(11); torch.manual_seed(11)
        sets['legacy sim'] = list(PS.legacy(N))
    fr41 = None
    for name, ckpt in targets[:1]:
        model, args, ep = G.load_model(ckpt)
        fr41 = cache41(model, args)
        del model
    share('real 41', [(f['img'], f['mask'], f['gt']) for f in fr41])
    for k, v in sets.items():
        share(k, v)

    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep})', flush=True)
        f41 = cache41(model, args)
        print('  real 41 segments:\n    ' + fmt(tally([(f['gt'], f['mask'], f['pb'], f['sc']) for f in f41])),
              flush=True)
        for k, frames in sets.items():
            items = []
            for img, m, gt in frames:
                pb, sc = predict(model, args, img, cfg)
                items.append((gt, m, pb, sc))
            print(f'  {k} sim segments:\n    ' + fmt(tally(items)), flush=True)
        del model
