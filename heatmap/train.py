"""Train the heatmap-first detector on the same simulator stream as ssl1.
  python heatmap/train.py --out <dir> [--bb simmim1|ssl1] [--unfreeze] ...
Resumable (reads <out>/checkpoint.pth)."""
import os, sys, time, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import types
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):   # skip DINO package inits (compiled ops)
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch

from models.heatmap_head import HeatmapNet, decode
from heatmap.targets_loss import build_targets, heatmap_loss

SIMMIM = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth'
SSL1 = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--bb', default='simmim1', choices=['simmim1', 'ssl1', 'random'])
    p.add_argument('--bb_path', default=None, help='override the backbone weights file (SimMIM export, or ssl1 checkpoint)')
    p.add_argument('--box_coef', default='2.80,1.30', help="label convention a_coef,w_coef; 'legacy' = ssl1's 3.5/1.0")
    p.add_argument('--unfreeze', action='store_true')
    p.add_argument('--out_stride', type=int, default=2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr_drop', type=int, default=45)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr_backbone', type=float, default=1e-5)
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--steps_per_epoch', type=int, default=250)   # x bs images
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_steps', type=int, default=0, help='smoke test: stop after N steps')
    return p.parse_args()


@torch.no_grad()
def quick_eval(model, epoch, out):
    try:
        from heatmap import evaluation as E
    except Exception as e:                       # e.g. eval deps (h5py/cv2/pandas) missing
        print(f'[epoch {epoch}] eval skipped: {type(e).__name__}: {e}', flush=True)
        return
    model.eval()
    for ds, path in E.DATASETS.items():
        gts, dets = [], []
        if not os.path.exists(path):
            print(f'[epoch {epoch}] eval skipped, missing {path}', flush=True)
            continue
        for cfg, ic in E.iter_frames(path):
            o = model(E.frame_inputs(ic, 'cuda'))
            gts.append(E.gt_of(ic))
            dets.append(E.heatmap_dets(cfg, decode(o, model.out_stride, 225)[0], use_nms=False))
        r = E.evaluate_dets(dets, gts, cfg)
        t = r['thr'][0.3]
        line = (f'{epoch}\t{r["ap"]:.4f}\trecall0.3 {t["recall"]:.3f}\tprec0.3 {t["precision"]:.3f}\t'
                f'chigap<5 {t["chigap"]["<5"][1]:.3f}')
        print(f'[epoch {epoch}] {ds}: {line}', flush=True)
        with open(os.path.join(out, f'exp_ap_{ds}.txt'), 'a') as f:
            f.write(line + '\n')
    model.train()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    import random; random.seed(a.seed)
    bb_path, bb_prefix = (SIMMIM, '') if a.bb == 'simmim1' else (SSL1, 'backbone.0.')
    bb_path = a.bb_path or bb_path
    if a.bb == 'random':                      # control arm: frozen RANDOM-init swin, no weights loaded
        bb_path, bb_prefix = None, ''
    model = HeatmapNet(backbone_ckpt=bb_path, backbone_prefix=bb_prefix, freeze_backbone=not a.unfreeze,
                       out_stride=a.out_stride).cuda()
    hm_args = dict(freeze_backbone=not a.unfreeze, out_stride=a.out_stride, bb_path=bb_path,
                   bb_prefix=bb_prefix, bb=a.bb)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    groups = [dict(params=head_params, lr=a.lr)]
    if a.unfreeze:
        groups.append(dict(params=[p for n, p in model.named_parameters() if n.startswith('backbone.')], lr=a.lr_backbone))
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, a.lr_drop)
    start = 0
    ck_path = os.path.join(a.out, 'checkpoint.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu')
        model.load_state_dict(ck['model'], strict=False)
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['lr_scheduler'])
        start = ck['epoch'] + 1
        print(f'[train] resumed at epoch {start}', flush=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=2)
    print('[train] trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    from simulation import FastSimulation
    cfg_sim = None
    if a.box_coef != 'legacy':               # current main convention; 'legacy' = ssl1's default 3.5/1.0
        from simulation import SimulationConfig
        cfg_sim = SimulationConfig()
        cfg_sim.a_coef, cfg_sim.w_coef = (float(v) for v in a.box_coef.split(','))
    print(f'[train] sim box convention: a_coef,w_coef = {a.box_coef}', flush=True)
    sim = FastSimulation(sim_config=cfg_sim, device='cuda')

    def sim_sample():
        while True:                           # same retry-on-failure as main.SimulationDataset
            try:
                img, boxes, _mask, is_ring = sim.simulate_img()
                return img[None], boxes.float(), is_ring.long()
            except Exception:
                pass
    stride = model.out_stride
    model.train()
    step = 0
    for epoch in range(start, a.epochs):
        t0 = time.time(); agg = dict(loss=0, loss_heat=0, loss_reg=0)
        for it in range(a.steps_per_epoch):
            imgs, hs, rs, ws = [], [], [], []
            for _ in range(a.bs):
                img, xyxy, lab = sim_sample()
                H, W = img.shape[-2:]
                h, r, w = build_targets(xyxy, lab, H, W, stride)
                imgs.append(img); hs.append(h); rs.append(r); ws.append(w)
            out = model(torch.stack(imgs))
            loss, parts = heatmap_loss(out, torch.stack(hs), torch.stack(rs), torch.stack(ws))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            agg['loss'] += loss.item(); agg['loss_heat'] += parts['loss_heat']; agg['loss_reg'] += parts['loss_reg']
            step += 1
            if a.max_steps and step >= a.max_steps:
                print(f'[smoke] step {step} loss {loss.item():.3f} {parts} {(time.time()-t0)/step:.2f}s/step', flush=True)
                return
        sched.step()
        n = a.steps_per_epoch
        print(f'[epoch {epoch}] loss {agg["loss"]/n:.4f} heat {agg["loss_heat"]/n:.4f} reg {agg["loss_reg"]/n:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        # (a random-init backbone can't be re-read from a file, so that arm saves it)
        torch.save(dict(model={k: v for k, v in model.state_dict().items()
                               if not (k.startswith('backbone.') and not a.unfreeze and a.bb != 'random')},
                        optimizer=opt.state_dict(), lr_scheduler=sched.state_dict(), epoch=epoch, hm_args=hm_args),
                   ck_path)
        if epoch % a.eval_interval == 0 or epoch == a.epochs - 1:
            quick_eval(model, epoch, a.out)


if __name__ == '__main__':
    main()
