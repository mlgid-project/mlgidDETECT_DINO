"""Ring / segment type confusion on cached detections: how many detections carry the wrong type, and what would AP be if they did not?

  python heatmap/type_confusion.py <dets.pkl> [--coef 3.0] [--sets organic 41] [--floor 0.1] [--out <file>.txt]

dets.pkl = heatmap/dump_dets.py output (+nms decode, floor 0.1, top-225); GT is rescaled to the run's training box convention (coef, as HM_GT_BOX_COEF=auto).
GT type = is_ring_geom (box height >= 70% of the valid column). Every detection is assigned to the GT 'regions' that contain its centre: segment GT -> centre inside the (scaled) box,
ring GT -> centre x inside the ring's x extent (rings span chi). Detection types: 'consistent' = a region of its OWN type contains it, 'mistyped' = only regions of the OTHER type contain it,
'background' = no GT region. The headline AP matches all detections against all GT (the type is not part of the matching, only the box shape is).
ORACLE: every mistyped detection is re-boxed as the other type (keeping its x extent; ring -> the y extent of the containing ring GT, segment -> the y extent of the containing segment GT) and AP is recomputed
with the evaluator of the headline number: the AP that perfect typing (and the y position that comes with it) would give on these detections. CPU only."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E
from util.evaluation import Evaluator, get_full_conf_results


def ap_of(frames, boxes_of):
    ev = Evaluator()
    for i, fr in enumerate(frames):
        b, s = boxes_of(i)
        ev.get_exp_metrics(torch.tensor(np.asarray(b, np.float32).reshape(-1, 4)), torch.tensor(np.asarray(s, np.float32)), torch.tensor(fr['gt']), fr['gtconf'])
    df1, df2 = get_full_conf_results(ev.metrics)
    return float(df2['ap_total'].values[0])


def regions(gt, ring, b):
    """[M, N_gt] bool: does the centre of detection m lie in the region of GT n (segment: box; ring: x extent)."""
    if len(b) == 0 or len(gt) == 0:
        return np.zeros((len(b), len(gt)), bool)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    inx = (cx[:, None] >= gt[None, :, 0]) & (cx[:, None] <= gt[None, :, 2])
    iny = (cy[:, None] >= gt[None, :, 1]) & (cy[:, None] <= gt[None, :, 3])
    return np.where(ring[None, :], inx, inx & iny)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dets'); p.add_argument('--coef', type=float, default=3.0); p.add_argument('--sets', nargs='*', default=['organic', '41'])
    p.add_argument('--floor', type=float, default=0.1); p.add_argument('--out', default=None)
    a = p.parse_args()
    E.set_gt_box_coef(a.coef, a.coef)
    d = pickle.load(open(a.dets, 'rb'))
    L = [f'TYPE CONFUSION  run={d.get("run")} epoch={d.get("epoch")}  dets={a.dets}  GT scaled to coef {a.coef}, detections score > {a.floor}', '']
    for ds in a.sets:
        frames = d['sets'][ds]
        for fr in frames:
            fr['gt'] = E._scale_gt(np.asarray(fr['gt'], np.float32).reshape(-1, 4), fr['mask'])
        base = lambda i: (np.asarray(frames[i]['boxes']).reshape(-1, 4)[np.asarray(frames[i]['scores']) > a.floor], np.asarray(frames[i]['scores'])[np.asarray(frames[i]['scores']) > a.floor])
        ap0 = ap_of(frames, base)
        cnt = {thr: dict(cons=0, mis=0, bg=0, mis_seg=0, mis_ring=0) for thr in (0.1, 0.3)}
        gtc = {thr: dict(seg=0, ring=0, only_wrong_seg=0, only_wrong_ring=0, none_seg=0, none_ring=0) for thr in (0.1, 0.3)}
        new = []
        for fr in frames:
            gt = fr['gt']; ring = E.is_ring_geom(gt, fr['mask']) if len(gt) else np.zeros(0, bool)
            sc = np.asarray(fr['scores']); k = sc > a.floor
            b = np.asarray(fr['boxes']).reshape(-1, 4)[k].astype(np.float32); s = sc[k]; c = np.asarray(fr['cls'])[k]
            R = regions(gt, ring, b)                                   # [M, N]
            own = np.where(c[:, None] == 1, R & ring[None, :], R & ~ring[None, :]) if len(b) and len(gt) else R
            oth = np.where(c[:, None] == 1, R & ~ring[None, :], R & ring[None, :]) if len(b) and len(gt) else R
            consistent = own.any(1) if len(gt) else np.zeros(len(b), bool)
            mistyped = (~consistent) & (oth.any(1) if len(gt) else np.zeros(len(b), bool))
            for thr in cnt:
                m = s > thr
                cnt[thr]['cons'] += int((consistent & m).sum()); cnt[thr]['mis'] += int((mistyped & m).sum()); cnt[thr]['bg'] += int((~consistent & ~mistyped & m).sum())
                cnt[thr]['mis_seg'] += int((mistyped & m & (c == 1)).sum())      # a ring-typed detection sitting on a segment GT
                cnt[thr]['mis_ring'] += int((mistyped & m & (c == 0)).sum())     # a segment-typed detection sitting on a ring GT
                for n in range(len(gt)):
                    key = 'ring' if ring[n] else 'seg'; gtc[thr][key] += 1
                    inside = R[:, n] & m
                    if not inside.any():
                        gtc[thr]['none_' + key] += 1
                    elif not (inside & (c == (1 if ring[n] else 0))).any():
                        gtc[thr]['only_wrong_' + key] += 1
            b2 = b.copy()
            for mi in np.nonzero(mistyped)[0]:
                n = int(np.nonzero(oth[mi])[0][0])
                b2[mi, 1], b2[mi, 3] = gt[n, 1], gt[n, 3]
            new.append((b2, s))
        ap1 = ap_of(frames, lambda i: new[i])
        L.append(f'== {ds}: {len(frames)} frames')
        L.append(f'  AP as scored (type-agnostic matching, IoU on box shape): {ap0:.4f}')
        L.append(f'  ORACLE AP with every mistyped detection re-boxed as the right type: {ap1:.4f}   (gain {ap1 - ap0:+.4f})')
        for thr in (0.1, 0.3):
            c_, g_ = cnt[thr], gtc[thr]; tot = c_['cons'] + c_['mis'] + c_['bg']
            L.append(f'  detections score>{thr}: {tot}: consistent {c_["cons"]} ({c_["cons"] / max(tot, 1):.1%}), MISTYPED {c_["mis"]} ({c_["mis"] / max(tot, 1):.1%}; ring-typed on a segment GT {c_["mis_seg"]}, segment-typed on a ring GT {c_["mis_ring"]}), background {c_["bg"]} ({c_["bg"] / max(tot, 1):.1%})')
            L.append(f'     GT regions: {g_["seg"]} segment (no detection inside {g_["none_seg"]} = {g_["none_seg"] / max(g_["seg"], 1):.1%}, only wrong-type detections inside {g_["only_wrong_seg"]} = {g_["only_wrong_seg"] / max(g_["seg"], 1):.1%}), '
                     f'{g_["ring"]} ring (none {g_["none_ring"]} = {g_["none_ring"] / max(g_["ring"], 1):.1%}, only wrong-type {g_["only_wrong_ring"]} = {g_["only_wrong_ring"] / max(g_["ring"], 1):.1%})')
        L.append('')
        print('\n'.join(L[-8:]), flush=True)
    if a.out:
        open(a.out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
