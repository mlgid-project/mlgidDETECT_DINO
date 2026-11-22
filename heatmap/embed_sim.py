"""Latent map step 1b: the same feature vector as embed_corpus.py, for frames of the SIMULATOR the heatmap models train on (GPU).

  python heatmap/embed_sim.py --out <file>.npz [--n 2000] [--seed 0] [--box_coef 3.0,3.0] [--sim legacy] [--bb dinov3_vith16plus --weights <pth>]

Frames come from simulation.FastSimulation (--sim legacy) or realsim_v2.dataset.RealSimV2 (--sim realsim_v2 --v2_data <folder>) exactly as train.py draws them (simulate_img, same retry on failure, no flip); the image is what the model sees (the simulator output, invalid pixels set to 0 as in the
real embeddings), the vector = masked mean of the stride-32 map averaged over chi and its flip (embed_corpus.embed). Output npz: feat [N, C] float16, set ('sim_<sim>'), frame (draw index), n_boxes, bb, weights.
Look at the sim frames in the map with heatmap/latent_sim.py."""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from heatmap.embed_corpus import embed, DATA, DinoV3Backbone, torch, np
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--n', type=int, default=2000); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--box_coef', default='3.0,3.0'); p.add_argument('--sim', default='legacy', choices=['legacy', 'realsim_v2']); p.add_argument('--v2_data', default=None, help='with --sim realsim_v2: folder of config.json + shards/ (realsim_v2_data/v2_60k)'); p.add_argument('--v2_flip', default='random', choices=['random', 'none'])
    p.add_argument('--cfg', default=None, help='legacy sim: JSON dict of SimulationConfig overrides, e.g. {"poisson_range": [10, 100], "mode_probs": [0.7, 0.1, 0.2], "chi_gradient": [0.1, 0.4]} (lists become tuples)')
    p.add_argument('--real_masks_region', type=int, default=None, help='intersect the sim mask of every frame with the no-data mask of a random real corpus frame of this latent-map region (k=6, latent_map/vith_sk2/regions_k6.csv)')
    p.add_argument('--bb', default='dinov3_vith16plus')
    p.add_argument('--weights', default=f'{DATA}/pretrain_runs/vith16plus_3k_sk2/backbone_teacher.pth')
    p.add_argument('--equalize', action='store_true', help='re-equalise the sim image over its VALID pixels like the real pipeline (simulation.contrast_like_real: clip 5/99.5, HE on the mask only); the legacy sim equalises over the whole image')
    p.add_argument('--stats', action='store_true', help='also store heatmap/frame_stats.py statistics of every frame (npz field stats, names)')
    p.add_argument('--dump_png', type=int, default=0, help='also save the first N frames as an npy stack next to --out (for a look)')
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.sim == 'realsim_v2':
        from realsim_v2.dataset import RealSimV2                       # frames as train.py --sim realsim_v2 draws them (deployed contrast, masked pixels 0, random flips)
        sim = RealSimV2(a.v2_data, box_coef=tuple(float(v) for v in a.box_coef.split(',')), flip=a.v2_flip, seed=a.seed, device='cuda')
    else:
        from simulation import FastSimulation, SimulationConfig
        cfg = SimulationConfig(); cfg.a_coef, cfg.w_coef = (float(v) for v in a.box_coef.split(','))
        if a.cfg:
            import json
            for k, v in json.loads(a.cfg).items():
                assert hasattr(cfg, k), k; setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
        sim = FastSimulation(sim_config=cfg, device='cuda')
    if a.real_masks_region is not None:
        import csv, h5py
        from heatmap.embed_corpus import decode
        z = np.load(f'{DATA}/latent_map/embed_vith_sk2.npz'); reg = np.array([int(r['region']) for r in csv.DictReader(open(f'{DATA}/latent_map/vith_sk2/regions_k6.csv'))])
        pool = np.nonzero((reg == a.real_masks_region) & (z['set'] == 'corpus'))[0]; rpath, rframe = z['path'], z['frame']
        rng = np.random.default_rng(a.seed)

        def real_mask():
            while True:
                j = int(rng.choice(pool))
                with h5py.File(str(rpath[j]), 'r') as h:
                    v = h['images'][int(rframe[j])]
                if v.shape == (512, 1024) and (v > 0).any():
                    return v > 0
    bb = DinoV3Backbone(a.bb)
    print('weights:', bb.net.load_state_dict(torch.load(a.weights, map_location='cpu', mmap=True), strict=True), flush=True)
    bb = bb.to('cuda').eval()
    feats, nb, keep, st = [], [], [], []
    if a.stats:
        from heatmap.frame_stats import frame_stats, NAMES
    t0 = time.time()
    for i in range(a.n):
        while True:
            try:
                img, boxes, mask, is_ring = sim.simulate_img(); break
            except Exception:
                pass
        img = (img if img.dim() == 3 else img[None])[0].float().cpu().numpy(); m = mask.cpu().numpy().astype(bool).reshape(img.shape)
        if a.real_masks_region is not None:
            rm = real_mask(); assert rm.shape == img.shape, (rm.shape, img.shape)
            m = m & rm                                      # keep the sim's own invalid pixels (dark wedge, gaps carry arbitrary values) AND the real no-data mask
        if a.equalize:
            from simulation import contrast_like_real
            img = contrast_like_real(torch.tensor(img, device='cuda'), torch.tensor(m, device='cuda'), clip=(5.0, 99.5), log=False, he=True).cpu().numpy()
        img = np.where(m, img, 0).astype(np.float32)
        feats.append(embed(bb, img, m, 'cuda').astype(np.float16)); nb.append(len(boxes))
        if a.stats:
            st.append([frame_stats(img, m)[n] for n in NAMES])
        if i < a.dump_png:
            keep.append(img)
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{a.n} frames, {(i + 1) / (time.time() - t0):.1f}/s', flush=True)
    np.savez(a.out, feat=np.stack(feats), set=np.array([f'sim_{a.sim}'] * a.n), frame=np.arange(a.n), n_boxes=np.array(nb), bb=a.bb, weights=a.weights, **(dict(stats=np.array(st, np.float32), names=np.array(NAMES)) if a.stats else {}))
    if keep:
        np.save(a.out.replace('.npz', '_frames.npy'), np.stack(keep))
    print('saved', a.out, a.n, 'frames', flush=True)


if __name__ == '__main__':
    main()
