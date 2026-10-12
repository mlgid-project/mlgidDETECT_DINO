"""Run a checkpoint on simulator frames saved by heatmap/bench_sim.py (sim_frames.npz) and write a dump_dets-format pickle, so that
heatmap/miss_analysis.py (hit rate of rings / segments, false positives by predicted class) can be applied to the model's OWN training distribution.

  python heatmap/probe_frames.py --ckpt <ckpt.pth> --npz <bench dir>/sim_frames.npz [<more npz> ...] --out <file>.pkl [--device cpu]
Env as dump_dets.py (HM_DATA_DIR, HM_BB_PATH). Set name in the pickle: 'sim_<folder name>' per npz file, plus 'sim_all' with all frames."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from models.heatmap_head import decode
from heatmap import evaluation as E
from heatmap.visualize import load
from heatmap.predict import detect


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--npz', nargs='+', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    model, ep = (None, -1) if a.ckpt == 'none' else load(a.ckpt, a.device)      # --ckpt none: only repackage the frames (no detections), e.g. for appearance_compare.py
    cfg = E.make_config(E.DATASETS['organic']) if model is not None else None
    sets, allf = {}, []
    for f in a.npz:
        z = np.load(f)
        ks = sorted({int(k.split('_')[0]) for k in z.files})
        fr = []
        for k in ks:
            img = torch.from_numpy(z[f'{k}_img']).float(); mask = torch.from_numpy(z[f'{k}_mask']).bool(); boxes = z[f'{k}_boxes']; ring = z[f'{k}_ring']
            if model is not None:
                x = img[None, None].to(a.device); m = mask.reshape(1, *img.shape[-2:]).to(a.device)
                with torch.no_grad():
                    o = model(x, m)
                b, s, c = detect(cfg, decode(o, model.out_stride, 225)[0])
            else:
                b, s, c = np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
            fr.append(dict(img=img.numpy().astype(np.float16), mask=mask.numpy(), gt=boxes.astype(np.float32), gtconf=np.ones(len(boxes), np.float32), boxes=b, scores=s, cls=c, gt_ring=ring))
        name = 'sim_' + os.path.basename(os.path.dirname(os.path.abspath(f))).replace('realsim_', '')
        sets[name] = fr; allf += fr
        print(f'{name}: {len(fr)} frames, {sum(len(f_["gt"]) for f_ in fr)} GT boxes ({sum(int(f_["gt_ring"].sum()) for f_ in fr)} rings)', flush=True)
    sets['sim_all'] = allf
    run = os.path.splitext(os.path.basename(a.ckpt))[0]
    pickle.dump(dict(run=run, epoch=ep, ckpt=a.ckpt, sets=sets), open(a.out, 'wb'))
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
