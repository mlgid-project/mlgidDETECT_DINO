"""Sim-to-real translation, step 1: the training data (GPU only for the simulator).

  HM_DATA_DIR=<dir> PYTHONPATH=. python heatmap/translate_prep.py --out <dir> [--n_real 4000] [--n_sim 4000] [--regions 4 5]

real_<R>.npy / real_<R>_mask.npy : uint8 [N, 512, 1024] frames of the latent-map regions (corpus rows of embed_vith_sk2.npz, k=6 regions, NOT the held-out beamtimes), contrast = DEPLOYED (clip 5/99.5, log, HE), quantised to 0..255, plus the no-data mask (uint8 0/1).
sim.npy / sim_mask.npy / sim_boxes.pkl : legacy FastSimulation frames (box_coef 3,3) on a random REAL mask of the regions, re-equalised over the valid pixels (simulation.contrast_like_real, as embed_sim.py --real_masks_region --equalize), boxes [x1,y1,x2,y2] px
(simulate_img already applied flip_image, so they match the image).
The translation model (translate_train.py) sees (image, mask) and has to keep the boxes valid."""
import argparse, csv, os, pickle, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, h5py, torch
from heatmap.embed_corpus import decode, DEPLOYED, DATA
from util.exp_preprocess import apply_contrast


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--n_real', type=int, default=4000); p.add_argument('--n_sim', type=int, default=4000); p.add_argument('--regions', type=int, nargs='+', default=[4, 5]); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip_real', action='store_true'); p.add_argument('--skip_sim', action='store_true')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    z = np.load(f'{DATA}/latent_map/embed_vith_sk2.npz'); reg = np.array([int(r['region']) for r in csv.DictReader(open(f'{DATA}/latent_map/vith_sk2/regions_k6.csv'))])
    ok_rows = (z['set'] == 'corpus') & ~z['held_out']
    pools = {r: np.nonzero(ok_rows & (reg == r))[0] for r in a.regions}
    print({r: len(v) for r, v in pools.items()}, 'usable corpus rows per region (held-out beamtimes excluded)', flush=True)

    def load(j):
        with h5py.File(str(z['path'][j]), 'r') as h:
            raw, ok = decode(h['images'][int(z['frame'][j])], float(h.attrs['log_max']))
        return raw, ok

    if not a.skip_real:
        for r in a.regions:
            rows = rng.choice(pools[r], min(a.n_real, len(pools[r])), replace=False)
            I = np.lib.format.open_memmap(f'{a.out}/real_{r}.npy', 'w+', np.uint8, (len(rows), 512, 1024)); M = np.lib.format.open_memmap(f'{a.out}/real_{r}_mask.npy', 'w+', np.uint8, (len(rows), 512, 1024))
            k = 0
            for j in rows:
                raw, ok = load(j)
                if raw.shape != (512, 1024) or not ok.any():
                    continue
                I[k] = np.round(np.where(ok, apply_contrast(raw, ok, DEPLOYED), 0) * 255).astype(np.uint8); M[k] = ok; k += 1
            I.flush(); M.flush(); np.save(f'{a.out}/real_{r}_n.npy', k); print(f'real region {r}: {k} frames', flush=True)

    if not a.skip_sim:
        from simulation import FastSimulation, SimulationConfig, contrast_like_real
        cfg = SimulationConfig(); cfg.a_coef, cfg.w_coef = 3.0, 3.0
        sim = FastSimulation(sim_config=cfg, device='cuda')
        allpool = np.concatenate([pools[r] for r in a.regions])
        I = np.lib.format.open_memmap(f'{a.out}/sim.npy', 'w+', np.uint8, (a.n_sim, 512, 1024)); M = np.lib.format.open_memmap(f'{a.out}/sim_mask.npy', 'w+', np.uint8, (a.n_sim, 512, 1024))
        boxes, t0 = [], time.time()
        for i in range(a.n_sim):
            while True:                                                           # draw until the frame is usable (redraw instead of skipping, so boxes stay aligned with frames)
                try:
                    img, b, mask, is_ring = sim.simulate_img()
                except Exception:
                    continue
                img = (img if img.dim() == 3 else img[None])[0].float()
                _, rm = load(int(rng.choice(allpool)))
                if rm.shape != (512, 1024):
                    continue
                m = rm & mask.cpu().numpy().astype(bool).reshape(img.shape)      # the sim's own invalid pixels (wedge, gaps) stay invalid AND the real no-data mask
                if m.sum() >= 0.2 * m.size:
                    break
            mt = torch.tensor(m, device='cuda')
            img = contrast_like_real(img, mt, clip=(5.0, 99.5), log=False, he=True).cpu().numpy()
            I[i] = np.round(np.where(m, img, 0) * 255).astype(np.uint8); M[i] = m
            boxes.append(dict(boxes=b.cpu().numpy().astype(np.float32), is_ring=is_ring.cpu().numpy().astype(bool)))
            if (i + 1) % 500 == 0:
                print(f'sim {i + 1}/{a.n_sim}  {(i + 1) / (time.time() - t0):.1f}/s', flush=True)
        I.flush(); M.flush(); pickle.dump(boxes, open(f'{a.out}/sim_boxes.pkl', 'wb')); print('saved sim', flush=True)


if __name__ == '__main__':
    main()
