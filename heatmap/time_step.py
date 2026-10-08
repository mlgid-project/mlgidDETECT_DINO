"""Where does a training step spend its time? Run when the GPU is otherwise idle.

  python heatmap/time_step.py --bb_path <SimMIM .pth> [--bs 4 8] [--n 15]

Times, on the real code paths of train.py: (1) the simulator alone, per image; (2) target building, per image;
(3) the network step alone (forward+backward+optimizer) on a FIXED batch, i.e. no simulator in the loop;
(4) the full step (simulate bs images + targets + network). Also peak GPU memory per batch size.
Note the simulator itself runs on the GPU, so it competes with the network for it."""
import os, sys, time, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import torch
from models.heatmap_head import HeatmapNet
from heatmap.targets_loss import build_targets, heatmap_loss
from simulation import FastSimulation, SimulationConfig


def sync():
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bb_path', required=True); p.add_argument('--bs', type=int, nargs='+', default=[4, 8])
    p.add_argument('--n', type=int, default=15)
    p.add_argument('--box_coef', default='2.80,1.30')
    a = p.parse_args()
    cfg = SimulationConfig(); cfg.a_coef, cfg.w_coef = (float(v) for v in a.box_coef.split(','))
    sim = FastSimulation(sim_config=cfg, device='cuda')

    def sample():
        while True:
            try:
                img, boxes, _m, ring = sim.simulate_img()
                return img[None], boxes.float(), ring.long()
            except Exception:
                pass

    model = HeatmapNet(backbone_ckpt=a.bb_path, freeze_backbone=True, out_stride=2).cuda().train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    for _ in range(3):
        sample()                                                     # warm up
    sync(); t = time.time(); pool = [sample() for _ in range(a.n * 2)]; sync()
    t_sim = (time.time() - t) / len(pool)
    t = time.time()
    tg = [build_targets(b, l, 512, 1024, 2) for _, b, l in pool]; sync()
    t_tg = (time.time() - t) / len(pool)
    print(f'simulator alone: {t_sim:.3f} s/img | target building: {t_tg:.4f} s/img', flush=True)

    def step(imgs, tgs):
        out = model(torch.stack(imgs))
        loss, _ = heatmap_loss(out, torch.stack([x[0] for x in tgs]), torch.stack([x[1] for x in tgs]),
                               torch.stack([x[2] for x in tgs]))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    for bs in a.bs:
        torch.cuda.reset_peak_memory_stats()
        imgs = [pool[i][0] for i in range(bs)]; tgs = tg[:bs]
        for _ in range(3):
            step(imgs, tgs)
        sync(); t = time.time()
        for _ in range(a.n):
            step(imgs, tgs)
        sync(); t_net = (time.time() - t) / a.n
        sync(); t = time.time()
        for _ in range(a.n):
            batch = [sample() for _ in range(bs)]
            step([b[0] for b in batch], [build_targets(b[1], b[2], 512, 1024, 2) for b in batch])
        sync(); t_full = (time.time() - t) / a.n
        print(f'bs {bs}: network step alone {t_net:.3f} s ({t_net/bs:.3f} s/img) | full step {t_full:.3f} s '
              f'({t_full/bs:.3f} s/img) | simulator share of full step ~{bs*t_sim/t_full*100:.0f}% | '
              f'peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
