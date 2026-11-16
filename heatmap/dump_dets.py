"""Cache a heatmap checkpoint's detections (+nms decode, floor 0.1, top-225) together with the images and ground truth, so that
post-processing ideas (heatmap/blob_diag.py, heatmap/split_blobs.py) can be developed on the CPU without re-running the network.

  python heatmap/dump_dets.py --ckpt <run_dir>/final_checkpoint.pth --out <file>.pkl [--sets organic 41] [--sim 300] [--sim_seed 0]
         [--box_coef 2.80,1.30] [--device cuda]
Env as visualize.py (HM_DATA_DIR, HM_BB_PATH).

<file>.pkl = {'run': ..., 'epoch': ..., 'sets': {name: [frame, ...]}}, frame = dict(img [H,W] float16 (the HE image the swin sees),
mask [H,W] bool (True = valid), gt [N,4] xyxy px, gtconf [N], boxes [M,4], scores [M], cls [M] (0 segment, 1 ring)).
'sim' frames come from the training simulator (fixed seed, same box convention as training) and have gtconf = 1; they are the set to tune on.
Boxes are polar-image pixels (512 x 1024): x = q axis, y = chi axis."""
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


def forward(model, x, mask, device):
    with torch.no_grad():
        return model(x, mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--sets', nargs='*', default=['organic', '41'])
    p.add_argument('--sim', type=int, default=0, help='number of simulator frames to add as set "sim"')
    p.add_argument('--sim_seed', type=int, default=0); p.add_argument('--box_coef', default='2.80,1.30')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, ep = load(a.ckpt, a.device)
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    sets = {}
    cfg_any = None
    for ds in a.sets:
        fr = []
        for cfg, ic in E.iter_frames(E.DATASETS[ds]):
            cfg_any = cfg
            x = (E.frame_inputs(ic, a.device) if model.chan_mode == 'he' else E.frame_inputs(ic, a.device, chan_mode=model.chan_mode))
            o = forward(model, x, E.frame_mask(ic, a.device), a.device)
            b, s, c = detect(cfg, decode(o, model.out_stride, 225)[0])
            g = E.gt_of(ic)
            fr.append(dict(img=np.asarray(ic.converted_polar_image[0, 0]).astype(np.float16), mask=g['mask'], gt=g['gt'], gtconf=g['gtconf'],
                           boxes=b, scores=s, cls=c))
            print(f'{ds} frame {len(fr) - 1}: {len(b)} detections', flush=True)
        sets[ds] = fr
    if a.sim:
        from simulation import FastSimulation, SimulationConfig
        cfg_sim = SimulationConfig(); cfg_sim.a_coef, cfg_sim.w_coef = (float(v) for v in a.box_coef.split(','))
        torch.manual_seed(a.sim_seed); np.random.seed(a.sim_seed)
        sim = FastSimulation(sim_config=cfg_sim, device='cuda' if torch.cuda.is_available() else 'cpu')
        if cfg_any is None:
            cfg_any = E.make_config(E.DATASETS['organic'])
        fr = []
        while len(fr) < a.sim:
            try:
                img, boxes, mask, is_ring = sim.simulate_img()
            except Exception:
                continue
            img = img if img.dim() == 3 else img[None]
            x = img[None].to(a.device).float(); m = mask.reshape(1, *img.shape[-2:]).to(a.device).bool()
            o = forward(model, x, m, a.device)
            b, s, c = detect(cfg_any, decode(o, model.out_stride, 225)[0])
            fr.append(dict(img=img[0].detach().cpu().numpy().astype(np.float16), mask=m[0].cpu().numpy(), gt=boxes.float().cpu().numpy(),
                           gtconf=np.ones(len(boxes), np.float32), boxes=b, scores=s, cls=c))
            if len(fr) % 50 == 0:
                print(f'sim frame {len(fr)}', flush=True)
        sets['sim'] = fr
    pickle.dump(dict(run=run, epoch=ep, ckpt=a.ckpt, sets=sets), open(a.out, 'wb'))
    print(f'wrote {a.out}: ' + ', '.join(f'{k} {len(v)} frames' for k, v in sets.items()), flush=True)


if __name__ == '__main__':
    main()
