"""The crowding numbers of bench_sim.py, measured on the REAL labels (organic, 41), so a simulator can be compared with them. CPU only, small.

  python heatmap/gt_crowding.py [--sets organic 41]
Per frame, for segment boxes (rings, i.e. boxes taller than 70 % of the valid height, are counted separately): number of segment / ring boxes,
share of the frame covered by the union of the segment boxes, overlapping segment pairs, box width and height. Same definitions as heatmap/bench_sim.py."""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
from heatmap import evaluation as E


def main():
    p = argparse.ArgumentParser(); p.add_argument('--sets', nargs='*', default=['organic', '41'])
    a = p.parse_args()
    for ds in a.sets:
        nseg, nring, cover, ovl, bw, bh = [], [], [], [], [], []
        for cfg, ic in E.iter_frames(E.DATASETS[ds]):
            g = E.gt_of(ic)
            gt = g['gt']
            ring = E.is_ring_geom(gt, g['mask']) if len(gt) else np.zeros(0, bool)
            sg = gt[~ring]
            nseg.append(len(sg)); nring.append(int(ring.sum()))
            if len(sg) == 0:
                continue
            occ = np.zeros((512, 1024), bool)
            for q_ in sg:
                occ[max(int(q_[1]), 0):int(q_[3]) + 1, max(int(q_[0]), 0):int(q_[2]) + 1] = True
            cover.append(occ.mean())
            ix = np.clip(np.minimum(sg[:, None, 2], sg[None, :, 2]) - np.maximum(sg[:, None, 0], sg[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(sg[:, None, 3], sg[None, :, 3]) - np.maximum(sg[:, None, 1], sg[None, :, 1]), 0, None)
            ovl.append(int(((ix * iy > 0).sum() - len(sg)) // 2))
            bw += list(sg[:, 2] - sg[:, 0]); bh += list(sg[:, 3] - sg[:, 1])
        P = lambda v, ps: '/'.join(f'{np.percentile(v, q):.0f}' for q in ps)
        print(f'{ds}: {len(nseg)} frames | segment boxes/frame mean {np.mean(nseg):.1f} (p10/50/90 {P(nseg, (10, 50, 90))}, max {max(nseg)}); ring boxes/frame mean {np.mean(nring):.2f}, '
              f'frames without ring {np.mean(np.array(nring) == 0):.0%} | segment-box union covers mean {np.mean(cover):.1%} (p90 {np.percentile(cover, 90):.1%}) | '
              f'overlapping pairs/frame mean {np.mean(ovl):.1f} (p90 {np.percentile(ovl, 90):.0f}) | box width px p10/50/90 {P(bw, (10, 50, 90))}, height {P(bh, (10, 50, 90))}', flush=True)


if __name__ == '__main__':
    main()
