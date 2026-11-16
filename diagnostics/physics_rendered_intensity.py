"""Does the physics intensity distribution SURVIVE rendering?

The label-level check (peak_intensity_gate.py) measures intensities as handed to img_from_labels.
This measures them where it actually matters: in the FINAL rendered image, after background,
Poisson noise, dark areas, detector gaps and the contrast chain -- using the same estimator
peak_intensity_gate.real_organic() applies to real frames (patch max minus a local background
annulus), so sim and real numbers are directly comparable.

The number that decides the design is `frac invisible`: GT boxes whose measured amplitude does not
clear the local noise. Those are labels on peaks the detector cannot see, and they train it to
hallucinate. The old gamma compression could not produce them; a linear many-decade mapping can.
"""
import argparse, sys, numpy as np, torch

sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
from simulation import SimulationConfig, FastSimulation
from physics_simulation import PhysicsSimulation


def measure(img, boxes, half=6, bg_lo=10, bg_hi=16):
    """amplitude = max over a patch at the box centre - median over a surrounding annulus;
    noise = MAD of the annulus. Returns (amps, noises)."""
    H, W = img.shape
    amps, noises = [], []
    for x1, y1, x2, y2 in boxes:
        x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        y0, y1_ = max(0, y - bg_hi), min(H, y + bg_hi + 1)
        x0, x1_ = max(0, x - bg_hi), min(W, x + bg_hi + 1)
        win = img[y0:y1_, x0:x1_]
        if win.size < 25:
            continue
        yy, xx = np.ogrid[y0 - y:y1_ - y, x0 - x:x1_ - x]
        r = np.hypot(yy, xx)
        core, ring = win[r <= half], win[(r >= bg_lo) & (r <= bg_hi)]
        if not core.size or not ring.size:
            continue
        bg = np.median(ring)
        amps.append(float(core.max() - bg))
        noises.append(float(np.median(np.abs(ring - bg)) * 1.4826))   # MAD -> sigma
    return np.asarray(amps), np.asarray(noises)


def frames(a):
    """Yield (img, boxes) already in the SAME representation for every source: the 512x1024 polar
    image AFTER its own preprocessing, with its own labels. Comparing a post-contrast sim image
    against pre-contrast real amplitudes is meaningless -- log+HE is a monotonic remap that
    flattens the histogram by design, so anything measured after it looks compressed."""
    if a.source == 'real':
        from util.configuration import Config
        from util.exp_preprocess import standard_preprocessing
        from util.pygidloader import PyGIDDataset
        cfg = Config(); cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]; cfg.INPUT_DATASET = a.real
        ds = PyGIDDataset(cfg, path=a.real, preprocess_func=standard_preprocessing,
                          buffer_size=5, load_labels=True)
        for ic in ds.iter_images():
            if ic.polar_labels is None:
                continue
            b = np.asarray(ic.polar_labels.boxes)
            if b.ndim != 2 or len(b) == 0:
                continue
            yield ic.converted_polar_image[0, 0].copy(), b
        return
    cfg = SimulationConfig(); cfg.a_coef, cfg.w_coef = 2.80, 1.30
    sim = (PhysicsSimulation(a.bank, sim_config=cfg, device='cuda', unify_contrast=a.unify)
           if a.source == 'physics' else FastSimulation(sim_config=cfg, device='cuda'))
    for _ in range(a.n):
        img, boxes, mask, is_ring = sim.simulate_img()
        yield img.detach().cpu().numpy(), boxes.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['physics', 'standard', 'real'], default='physics')
    ap.add_argument('--bank')
    ap.add_argument('--real', default='/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--unify', action='store_true')
    a = ap.parse_args()

    pooled, invisible, nbox = [], [], []
    for img, boxes in frames(a):
        amps, noises = measure(img, boxes)
        if len(amps) < 5 or amps.max() <= 0:
            continue
        pooled.append(np.clip(amps, 1e-9, None) / amps.max())
        invisible.append((amps < 3 * noises).mean())
        nbox.append(len(boxes))

    tag_src = a.source + (f' unify={a.unify}' if a.source == 'physics' else '')
    if not pooled:
        print(f'{tag_src:<26} NO USABLE FRAMES (no labeled boxes survived the estimator)')
        return
    v = np.concatenate(pooled)
    tag = a.source + (f' unify={a.unify}' if a.source == 'physics' else '')
    print(f'{tag:<26} images={len(pooled):<4} boxes/img={np.mean(nbox):5.1f}  '
          f'med I/Imax {np.median(v):.4f}  frac<0.1 {(v < 0.1).mean():.3f}  '
          f'frac_invisible {np.mean(invisible):.3f}')


if __name__ == '__main__':
    main()
