"""Train the heatmap detector on simulated GIWAXS frames (polar images, 512 x 1024).

  python heatmap/train.py --out runs/my_run --bb_path <DINOv3 ConvNeXt-L weights .pth>

The defaults are the best recipe found so far (scaled-GT AP, 3 seeds, organic / 41: 0.692 / 0.697 with ConvNeXt-L):
DINOv3 ConvNeXt-L backbone trained along (backbone lr 1e-5, gradient checkpointing), head lr 3e-4, ridge target for rings,
boxes = centre +- 1.5 sigma (box_coef 3.0,3.0), 120 epochs of 250 steps x batch 4, lr x0.1 after epochs 90 and 112, TF32.
About 7 h on one 12 GB consumer GPU (peak 11 GB). Resumable: run the same command again and it continues from <out>/checkpoint.pth.

Other backbones: --bb dinov3_vith16plus (same AP, 2.4x the compute, needs a 40 GB GPU) or --bb simmim1 (our SimMIM Swin-L).
Every in-training evaluation scores the labelled sets in HM_DATA_DIR (organic_labeled.h5, 41.h5) when they exist.
Experimental options (other simulators, head sizes, input channels, ...) live on the dev branch."""
import os, sys, time, json, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import types
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):   # skip the DINO package inits (compiled ops are not needed here)
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch

from models.heatmap_head import HeatmapNet, decode
from heatmap.targets_loss import build_targets, heatmap_loss

BACKBONES = ('dinov3_convnext_large', 'dinov3_vith16plus', 'simmim1')
SIMMIM = os.environ.get('HM_SIMMIM_WEIGHTS', os.path.join(ROOT, 'weights', 'swin_large_patch4_window12_384_22k.pth'))


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='run folder (created; holds checkpoint.pth, train log values, exp_ap_*.txt)')
    p.add_argument('--bb', default='dinov3_convnext_large', choices=BACKBONES, help='backbone (default: DINOv3 ConvNeXt-L)')
    p.add_argument('--bb_path', default=None, help='backbone weights file: the DINOv3 .pth from Meta for dinov3_*, default weights/swin_large_patch4_window12_384_22k.pth for simmim1')
    p.add_argument('--frozen', action='store_true', help='keep the backbone frozen (faster, about 0.03-0.10 AP lower)')
    p.add_argument('--box_coef', default='3.0,3.0', help='box size in the simulator, a_coef,w_coef = full box extent in sigma (3.0 = centre +- 1.5 sigma)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr_drops', type=int, nargs='+', default=[90, 112], help='epochs after which the lr is multiplied by 0.1')
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--steps_per_epoch', type=int, default=250)
    p.add_argument('--lr', type=float, default=3e-4, help='head lr')
    p.add_argument('--lr_backbone', type=float, default=1e-5, help='backbone lr (unless --frozen)')
    p.add_argument('--eval_interval', type=int, default=2, help='epochs between evaluations on the labelled sets')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_steps', type=int, default=0, help='smoke test: stop after N steps')
    a = p.parse_args()
    if a.bb_path is None:
        if a.bb == 'simmim1':
            a.bb_path = SIMMIM
        else:
            p.error(f'--bb {a.bb} needs --bb_path <weights .pth> (DINOv3 weights are downloaded from Meta, see README)')
    if not os.path.exists(a.bb_path):
        p.error(f'backbone weights not found: {a.bb_path}')
    return a


@torch.no_grad()
def quick_eval(model, epoch, out):
    try:
        from heatmap import evaluation as E
    except Exception as e:                       # e.g. eval deps (h5py/cv2/pandas) missing
        print(f'[epoch {epoch}] eval skipped: {type(e).__name__}: {e}', flush=True)
        return
    model.eval()
    for ds, path in E.DATASETS.items():
        if not os.path.exists(path):
            print(f'[epoch {epoch}] eval skipped, missing {path} (set HM_DATA_DIR)', flush=True)
            continue
        gts, dets = [], []
        for cfg, ic in E.iter_frames(path):
            o = model(E.frame_inputs(ic, 'cuda'), E.frame_mask(ic, 'cuda'))
            gts.append(E.gt_of(ic))
            dets.append(E.heatmap_dets(cfg, decode(o, model.out_stride, 225)[0], use_nms=True))
        try:
            r = E.evaluate_dets(dets, gts, cfg, thr_list=(0.3,)); t = r['thr'][0.3]
        except Exception as e:                   # e.g. a model that detects nothing yet: never let an evaluation stop the training
            print(f'[epoch {epoch}] {ds}: evaluation failed ({type(e).__name__}: {e}), training continues', flush=True)
            continue
        print(f'[epoch {epoch}] {ds}: AP {r["ap"]:.4f} | score>0.3 recall {t["recall"]:.3f} prec {t["precision"]:.3f} '
              f'ring recall {t["ring_recall"]:.3f} FP {t["fp"]}', flush=True)
        with open(os.path.join(out, f'exp_ap_{ds}.txt'), 'a') as f:
            f.write(f'{epoch}\t{r["ap"]:.4f}\trecall0.3 {t["recall"]:.3f}\tprec0.3 {t["precision"]:.3f}\n')
    model.train()


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    unfreeze = not a.frozen
    bb_kind = 'swin' if a.bb == 'simmim1' else a.bb
    model = HeatmapNet(backbone_ckpt=a.bb_path, backbone_prefix='', freeze_backbone=not unfreeze, grad_ckpt=unfreeze, bb_kind=bb_kind).cuda()
    # hm_args: what evaluate.py / predict.py / visualize.py need to rebuild the model (same keys as on the dev branch)
    hm_args = dict(freeze_backbone=not unfreeze, out_stride=model.out_stride, bb_path=a.bb_path, bb_prefix='', bb=a.bb, bb_kind=bb_kind,
                   amp_backbone=False, tf32=True, ring_target='ridge', chan_mode='he', zero_invalid=False,
                   dim=128, tower_ch=64, tower_depth=2, stem_ch=32, ring_head_stride=0, box_coef=a.box_coef)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    groups = [dict(params=head_params, lr=a.lr)]
    if unfreeze:
        groups.append(dict(params=[p for n, p in model.named_parameters() if n.startswith('backbone.')], lr=a.lr_backbone))
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, a.lr_drops)
    start = 0
    ck_path = os.path.join(a.out, 'checkpoint.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location='cpu')
        model.load_state_dict(ck['model'], strict=False)
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['lr_scheduler'])
        start = ck['epoch'] + 1
        print(f'[train] resumed at epoch {start}', flush=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=2)
    try:                                         # evaluations score against GT boxes scaled to this run's box convention
        from heatmap import evaluation as _E
        print('[train] eval GT box scale (x,y):', _E.set_gt_box_coef(*a.box_coef.split(',')), flush=True)
    except Exception as e:
        print(f'[train] GT box scale not set: {type(e).__name__}: {e}', flush=True)
    print('[train] trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    from simulation import FastSimulation, SimulationConfig
    cfg_sim = SimulationConfig()
    cfg_sim.a_coef, cfg_sim.w_coef = (float(v) for v in a.box_coef.split(','))
    sim = FastSimulation(sim_config=cfg_sim, device='cuda')

    def sim_sample():
        while True:                               # the simulator fails now and then on a random draw: draw again
            try:
                img, boxes, mask, is_ring = sim.simulate_img()
                return (img if img.dim() == 3 else img[None]), boxes.float(), is_ring.long(), mask
            except Exception:
                pass

    stride = model.out_stride
    model.train()
    step = 0
    for epoch in range(start, a.epochs):
        t0 = time.time(); agg = dict(loss=0.0, loss_heat=0.0, loss_reg=0.0)
        for it in range(a.steps_per_epoch):
            imgs, hs, rs, ws, ms = [], [], [], [], []
            for _ in range(a.bs):
                img, xyxy, lab, msk = sim_sample()
                h, r, w = build_targets(xyxy, lab, img.shape[-2], img.shape[-1], stride, 'ridge')
                imgs.append(img); hs.append(h); rs.append(r); ws.append(w); ms.append(msk)
            out = model(torch.stack(imgs), torch.stack(ms))
            loss, parts = heatmap_loss(out, torch.stack(hs), torch.stack(rs), torch.stack(ws))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            agg['loss'] += loss.item(); agg['loss_heat'] += parts['loss_heat']; agg['loss_reg'] += parts['loss_reg']
            step += 1
            if a.max_steps and step >= a.max_steps:
                print(f'[smoke] step {step} loss {loss.item():.3f} {(time.time() - t0) / step:.2f} s/step, '
                      f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
                return
        sched.step()
        n = a.steps_per_epoch
        print(f'[epoch {epoch}] loss {agg["loss"] / n:.4f} heat {agg["loss_heat"] / n:.4f} reg {agg["loss_reg"] / n:.4f} ({time.time() - t0:.0f}s)', flush=True)
        # a frozen backbone is not stored (it is re-read from --bb_path); tmp + rename so a failed write never leaves a broken checkpoint
        torch.save(dict(model={k: v for k, v in model.state_dict().items() if unfreeze or not k.startswith('backbone.')},
                        optimizer=opt.state_dict(), lr_scheduler=sched.state_dict(), epoch=epoch, hm_args=hm_args), ck_path + '.tmp')
        os.replace(ck_path + '.tmp', ck_path)
        if epoch % a.eval_interval == 0 or epoch == a.epochs - 1:
            quick_eval(model, epoch, a.out)


if __name__ == '__main__':
    main()
