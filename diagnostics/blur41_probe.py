"""Causal test of "the realbkg sim's peaks are too blurry": blur REAL 41 and re-score.

peak_shape_compare.py (job 2945413) measured, on the preprocessed image: real 41 segments have a
visible radial FWHM of 8 px and edge steepness 0.53/px; the legacy sim (41 AP 0.761) matches at
8 px / 0.37, while realbkg run 6 draws 15 px / 0.22 and physics4 16 px / 0.32 -- organic-like
(14 px). If the realbkg models learned "a peak is a wide soft bump", blurring 41's sharp peaks
towards that should RAISE their confidence and AP, and LOWER the legacy-trained model's.

The blur is applied to the preprocessed image the network receives, along x (radial q) only,
mask-normalised so the dark/masked area does not bleed in. Nothing in the repo changes.

  python diagnostics/blur41_probe.py name=/ckpt.pth [...]
"""
import os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G

SIGMAS = [0.0, 1.5, 3.0, 5.0]


def blur_x(img, mask, s):
    if s <= 0:
        return img
    m = mask.astype(np.float32)
    num = gaussian_filter1d(img * m, s, axis=-1, mode='constant')
    den = gaussian_filter1d(m, s, axis=-1, mode='constant')
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), img)
    return np.where(mask, out, img).astype(np.float32)


def cache_blur(model, args, s):
    cfg = G._config(G.DATASETS['41'])
    ds = G.H5GIWAXSDataset(cfg, path=G.DATASETS['41'], preprocess_func=G.standard_preprocessing,
                           buffer_size=5)
    fr = []
    for ic in ds.iter_images():
        x = np.asarray(ic.converted_polar_image[:, 0], np.float32)[0]
        m = np.asarray(ic.converted_mask).reshape(512, 1024).astype(bool)
        xb = blur_x(x, m, s)
        with torch.no_grad():
            o = model(torch.tensor(xb)[None, None].to(G.DEV).repeat(1, args.num_channels, 1, 1))
        fr.append(dict(logits=o['pred_logits'].cpu().numpy(), boxes=o['pred_boxes'].cpu().numpy(),
                       mask=m, gt=np.asarray(ic.polar_labels.boxes, np.float32),
                       gtconf=np.asarray(ic.polar_labels.confidences, np.float32)))
    return fr, cfg


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep}) on 41, radial blur of the input', flush=True)
        for s in SIGMAS:
            fr, cfg = cache_blur(model, args, s)
            S, _ = G.analyse(fr, cfg)
            print(f'  sigma {s:3.1f} px | ap_total {S["ap"]:.4f} | recall {S["tp"]/S["gt"]:.3f} '
                  f'(rings {S["tp_r"]/max(S["gt_r"],1):.3f}, segs {S["tp_s"]/max(S["gt_s"],1):.3f}) | '
                  f'precision {S["tp"]/max(S["pred"],1):.3f} | FP/frame {(S["pred"]-S["tp"])/len(fr):.1f} | '
                  f'TP {S["tp_med"]:.3f} FP {S["fp_med"]:.3f} sep {S["tp_med"]-S["fp_med"]:.3f}',
                  flush=True)
        del model
