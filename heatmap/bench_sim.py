"""Speed and look of a simulator, for deciding how to feed the training loop (GPU; run on colorbox1, never on the login node).

  python heatmap/bench_sim.py --sim realbkg [--n 100] [--save 12] [--out <dir>] [--realsim_opt p_ring=0.4 ...]
  python heatmap/bench_sim.py --sim legacy  --n 100                      (same numbers for the simulator all runs so far used)

Prints: seconds per frame (first 5 frames excluded), share of calls that returned nothing, boxes per frame, ring share, resident memory.
With --save K it writes K frames as <out>/sim_XX.png (HE image, segment boxes green, ring boxes red) and <out>/sim_frames.npz."""
import os, sys, types, time, argparse, ast, resource
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', default='realbkg', choices=['legacy', 'realbkg'])
    p.add_argument('--n', type=int, default=100); p.add_argument('--save', type=int, default=0)
    p.add_argument('--out', default='sim_bench'); p.add_argument('--box_coef', default='2.80,1.30')
    p.add_argument('--realsim_opt', nargs='*', default=[]); p.add_argument('--realsim_preset', default='r6', choices=['r6', 'r7', 'r8']); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    import random
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    coef = tuple(float(v) for v in a.box_coef.split(','))
    t0 = time.time()
    if a.sim == 'realbkg':
        from realsim.factory import make_realbkg_sim
        sim = make_realbkg_sim(device='cuda', box_coef=coef, overrides={k: ast.literal_eval(v) for k, v in (o.split('=', 1) for o in a.realsim_opt)}, preset=a.realsim_preset)
    else:
        from simulation import FastSimulation, SimulationConfig
        cfg = SimulationConfig(); cfg.a_coef, cfg.w_coef = coef
        sim = FastSimulation(sim_config=cfg, device='cuda')
    print(f'construction: {time.time() - t0:.1f} s, peak RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB', flush=True)
    ts, nb, nr, none, frames = [], [], [], 0, []
    cover, ovl, bw, bh = [], [], [], []
    for i in range(a.n):
        t = time.time()
        try:
            out = sim.simulate_img()
        except Exception as e:
            out = None; print(f'frame {i}: exception {type(e).__name__}: {e}', flush=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        if out is None:
            none += 1; continue
        img, boxes, mask, is_ring = out
        if i >= 5:
            ts.append(dt)
        nb.append(len(boxes)); nr.append(int(is_ring.sum()))
        sg = boxes[~is_ring.bool()].float().cpu().numpy()
        if len(sg):
            occ = np.zeros((512, 1024), bool)
            for q_ in sg:
                occ[max(int(q_[1]), 0):int(q_[3]) + 1, max(int(q_[0]), 0):int(q_[2]) + 1] = True
            cover.append(occ.mean())
            ix = np.clip(np.minimum(sg[:, None, 2], sg[None, :, 2]) - np.maximum(sg[:, None, 0], sg[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(sg[:, None, 3], sg[None, :, 3]) - np.maximum(sg[:, None, 1], sg[None, :, 1]), 0, None)
            ovl.append(int(((ix * iy > 0).sum() - len(sg)) // 2))
            bw += list(sg[:, 2] - sg[:, 0]); bh += list(sg[:, 3] - sg[:, 1])
        if len(frames) < a.save:
            frames.append(dict(img=img.float().cpu().numpy(), boxes=boxes.float().cpu().numpy(), mask=mask.cpu().numpy(), ring=is_ring.cpu().numpy()))
    print(f'{a.sim}: {len(ts)} timed frames, {np.mean(ts) * 1000:.0f} ms/frame (median {np.median(ts) * 1000:.0f}, p90 {np.percentile(ts, 90) * 1000:.0f}); '
          f'returned nothing {none}/{a.n}; boxes/frame mean {np.mean(nb):.1f} (p10 {np.percentile(nb, 10):.0f}, p90 {np.percentile(nb, 90):.0f}); '
          f'ring boxes/frame {np.mean(nr):.2f}, frames without any ring {np.mean(np.array(nr) == 0):.0%}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB', flush=True)
    if cover:
        print(f'segment boxes: share of the frame covered by their union mean {np.mean(cover):.1%} (p90 {np.percentile(cover, 90):.1%}), '
              f'overlapping pairs/frame mean {np.mean(ovl):.1f} (p90 {np.percentile(ovl, 90):.0f}); box width px p10/50/90 {np.percentile(bw, 10):.0f}/{np.percentile(bw, 50):.0f}/{np.percentile(bw, 90):.0f}, '
              f'height {np.percentile(bh, 10):.0f}/{np.percentile(bh, 50):.0f}/{np.percentile(bh, 90):.0f}', flush=True)
    if frames:
        import cv2
        os.makedirs(a.out, exist_ok=True)
        for k, f in enumerate(frames):
            im = cv2.cvtColor((np.clip(f['img'], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for b, r in zip(f['boxes'], f['ring']):
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255) if r else (0, 200, 0), 1)
            cv2.imwrite(os.path.join(a.out, f'sim_{k:02d}.png'), im)
        np.savez_compressed(os.path.join(a.out, 'sim_frames.npz'), **{f'{k}_{n}': v for k, f in enumerate(frames) for n, v in f.items()})
        print(f'wrote {len(frames)} frames to {a.out}', flush=True)


if __name__ == '__main__':
    main()
