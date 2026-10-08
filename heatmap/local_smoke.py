"""Local smoke test (single consumer GPU, e.g. RTX 5070 12 GB). No compiled DINO ops, no main.py.

  python heatmap/local_smoke.py --bb /path/swin_large_patch4_window12_384_22k.pth --steps 30 [--bs 1] [--amp]

Does: build the frozen-backbone heatmap net, train N steps on the live simulator, report loss,
s/step and peak GPU memory, then decode one fresh sim frame and save an overlay PNG
(green = GT, red = prediction with score > 0.3). Optionally scores a labeled .h5 with --eval_h5.
Importing models.heatmap_head would normally run models/__init__ -> dino -> compiled CUDA ops;
the stub below skips those package __init__s (nothing in the repo is modified)."""
import os, sys, time, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for name, sub in (('models', 'models'), ('models.dino', 'models/dino')):
    m = types.ModuleType(name); m.__path__ = [os.path.join(ROOT, sub)]; sys.modules[name] = m

import numpy as np
import torch
from models.heatmap_head import HeatmapNet, decode
from heatmap.targets_loss import build_targets, heatmap_loss


def sim_sample(sim):
    while True:
        try:
            img, boxes, mask, is_ring = sim.simulate_img()
            return img, boxes, is_ring.long()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bb', required=True, help='SimMIM export swin_large_patch4_window12_384_22k.pth')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--bs', type=int, default=1)
    p.add_argument('--amp', action='store_true', help='bf16 autocast (use if 12 GB is tight)')
    p.add_argument('--out', default='local_smoke_out')
    p.add_argument('--eval_h5', default=None, help='optional labeled .h5 to score (needs the pygid loader deps)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
    from simulation import FastSimulation
    sim = FastSimulation(device='cuda')
    model = HeatmapNet(backbone_ckpt=a.bb, freeze_backbone=True, out_stride=2).cuda().train()
    params = [q for n, q in model.named_parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(a.steps):
        imgs, hs, rs, ws = [], [], [], []
        for _ in range(a.bs):
            img, boxes, lab = sim_sample(sim)
            H, W = img.shape[-2:]
            h, r, w = build_targets(boxes.float(), lab, H, W, model.out_stride)
            imgs.append(img[:1]); hs.append(h); rs.append(r); ws.append(w)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
            out = model(torch.stack(imgs))
        out = {k: v.float() for k, v in out.items()}
        loss, parts = heatmap_loss(out, torch.stack(hs), torch.stack(rs), torch.stack(ws))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 5 == 0 or s == a.steps - 1:
            print(f'step {s:4d} loss {loss.item():.3f} {parts} {(time.time()-t0)/(s+1):.2f}s/step '
                  f'peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
    # decode one fresh frame -> overlay
    model.eval()
    with torch.no_grad():
        img, boxes, lab = sim_sample(sim)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
            o = model(img[:1][None])
        b, sc, cl = decode({k: v.float() for k, v in o.items()}, model.out_stride, 225)[0]
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt, matplotlib.patches as mp
    fig, ax = plt.subplots(figsize=(14, 7)); ax.imshow(img[0].cpu(), cmap='gray')
    for x0, y0, x1, y1 in boxes.cpu().numpy():
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='lime', lw=0.8))
    k = sc > 0.3
    for (x0, y0, x1, y1) in b[k].cpu().numpy():
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='red', lw=0.8))
    ax.set_title(f'GT {len(boxes)} (green) | pred score>0.3: {int(k.sum())} (red)')
    fig.savefig(os.path.join(a.out, 'overlay.png'), dpi=100, bbox_inches='tight')
    print('saved', os.path.join(a.out, 'overlay.png'), flush=True)
    if a.eval_h5:
        from heatmap import evaluation as E
        gts, dets = [], []
        for cfg, ic in E.iter_frames(a.eval_h5):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                oo = model(E.frame_inputs(ic, 'cuda'))
            gts.append(E.gt_of(ic))
            dets.append(E.heatmap_dets(cfg, decode({k: v.float() for k, v in oo.items()}, 2, 225)[0], use_nms=False))
        print(E.format_result('local', os.path.basename(a.eval_h5), E.evaluate_dets(dets, gts, cfg)))


if __name__ == '__main__':
    main()
