"""Cache a DINO checkpoint's detections (same pickle layout as dump_dets.py) so heatmap and DINO can be drawn side by side.

  python heatmap/dump_dets_dino.py --ckpt <run_dir>/checkpoint.pth --out <file>.pkl [--sets organic 41] [--no_img]
Env: HM_DATA_DIR as visualize.py; HM_GT_BOX_COEF="a,w" scales the stored GT to the run's training box convention (old DINO: 2.8,1.3; the
new-box DINO trained at +-1 sigma = 2,2 needs none). Detections = top-225 + class-aware NMS + floor 0.1 (E.dino_dets, exactly as --eval);
cls = 1 for boxes spanning >= 70% of the valid chi range at their q column (E.is_ring_geom), else 0 (DINO has one class).
Boxes are polar-image pixels (512 x 1024): x = q axis, y = chi axis. Needs the compiled DINO ops (Galvani)."""
import os, sys, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from heatmap import evaluation as E
from heatmap.evaluate import load_dino


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--sets', nargs='*', default=['organic', '41'])
    p.add_argument('--no_img', action='store_true')
    a = p.parse_args()
    dev = 'cuda'
    model, args = load_dino(a.ckpt)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    sets = {}
    with torch.no_grad():
        for ds in a.sets:
            fr = []
            for cfg, ic in E.iter_frames(E.DATASETS[ds]):
                o = model(E.frame_inputs(ic, dev, args.num_channels))
                b, s = E.dino_dets(cfg, o)
                g = E.gt_of(ic)
                c = E.is_ring_geom(b, g['mask']).astype(int) if len(b) else np.zeros(0, int)
                fr.append(dict(img=None if a.no_img else np.asarray(ic.converted_polar_image[0, 0]).astype(np.float16), mask=g['mask'], gt=g['gt'],
                               gtconf=g['gtconf'], boxes=b, scores=s, cls=c))
                print(f'{ds} frame {len(fr) - 1}: {len(b)} detections', flush=True)
            sets[ds] = fr
    pickle.dump(dict(run=run, epoch=None, ckpt=a.ckpt, sets=sets), open(a.out, 'wb'))
    print(f'wrote {a.out}: ' + ', '.join(f'{k} {len(v)} frames' for k, v in sets.items()), flush=True)


if __name__ == '__main__':
    main()
