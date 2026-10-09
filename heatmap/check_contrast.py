"""Sanity checks for --chan contrast (run on colorbox1, GPU for the sim part):
 1. real frames: contrast ch0 (log+HE from raw_polar_image) == converted_polar_image (the deployed contrast)
 2. per-channel mean/std on valid pixels, real (organic, 41) vs simulated, and finite/range checks
 3. the simulator in contrast mode returns (3,H,W), invalid==0, and the default mode is unchanged in shape."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heatmap import evaluation as E


def stats(x, m):
    return ' | '.join(f'ch{i} {x[i][m].mean():.3f}+-{x[i][m].std():.3f} [{x[i].min():.2f},{x[i].max():.2f}]' for i in range(len(x)))


def main():
    for ds, path in E.DATASETS.items():
        worst = 0.0; agg = []
        for cfg, ic in E.iter_frames(path):
            x = E.frame_inputs(ic, 'cpu', chan_mode='contrast')[0].numpy()
            conv = np.asarray(ic.converted_polar_image)[0, 0]
            m = np.asarray(ic.converted_mask).reshape(*E.POLAR).astype(bool)
            d = np.abs(x[0] - conv).max(); worst = max(worst, d)
            assert np.isfinite(x).all()
            assert (x[:, ~m] == 0).all(), 'invalid pixels must be 0 in every channel'
            agg.append(np.array([[x[i][m].mean(), x[i][m].std()] for i in range(3)]))
        assert worst < 1e-3, f'contrast ch0 differs from converted_polar_image by {worst}'   # exit code != 0 stops an unattended queue
        a = np.mean(agg, 0)
        print(f'[{ds}] frames {len(agg)}: max|ch0 - converted_polar_image| = {worst:.2e} (expect ~0); '
              + ' | '.join(f'ch{i} {a[i,0]:.3f}+-{a[i,1]:.3f}' for i in range(3)), flush=True)
    from simulation import FastSimulation, SimulationConfig
    for flag in (False, True):
        c = SimulationConfig(); c.a_coef, c.w_coef = 2.80, 1.30; c.contrast_channels = flag
        sim = FastSimulation(sim_config=c, device='cuda')
        agg = []
        for _ in range(20):
            img, boxes, mask, is_ring = sim.simulate_img()
            img = img.float().cpu().numpy(); mask = mask.bool().cpu().numpy()
            if flag:
                assert img.shape == (3,) + mask.shape, img.shape
                assert np.isfinite(img).all() and (img[:, ~mask] == 0).all()
                agg.append(np.array([[img[i][mask].mean(), img[i][mask].std()] for i in range(3)]))
            else:
                assert img.shape == mask.shape, img.shape
        print(f'[sim contrast_channels={flag}] ok', (' | '.join(f'ch{i} {v[0]:.3f}+-{v[1]:.3f}' for i, v in enumerate(np.mean(agg, 0))) if flag else ''), flush=True)


if __name__ == '__main__':
    main()
