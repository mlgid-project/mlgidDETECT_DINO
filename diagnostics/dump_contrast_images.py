"""Render the same frame under several contrasts, side by side, with the ground-truth boxes
drawn on top -- the visual counterpart to the numbers in diagnostics/sweep_contrast.py.

Point of the tool: ap_total tells you which contrast the detector scores best on, not WHY.
Seeing the frames shows which peaks a contrast reveals or destroys, which is what you need
before choosing the contrasts to train on.

CPU only, no model is loaded -- it runs on a login node in seconds.

USAGE
  # the three contrasts going into the multi-contrast run, on 3 organic frames
  python diagnostics/dump_contrast_images.py --dataset organic --frames 3

  # the best 5 settings for organic, straight out of the sweep CSV
  python diagnostics/dump_contrast_images.py --dataset organic --top 5 --by organic

  # named settings, by their canonical sweep label
  python diagnostics/dump_contrast_images.py --settings clip=5/99.5_log_he clip=1/99_log_he

  # zoom on a q/chi window (polar pixels: x is q of 1024, y is chi of 512)
  python diagnostics/dump_contrast_images.py --crop 200 500 0 512
"""
import argparse, csv, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.contrast_grids import build_grid, label, by_label
from util.exp_preprocess import standard_preprocessing, apply_contrast
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.channels import CONTRAST_CHANNELS

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DATASETS = {'41': f'{CUR}/41.h5', 'organic': f'{CUR}/organic_labeled.h5'}
SWEEPS = f'{CUR}/DINO_BACKBONE_curation/sweeps'
POLAR_SHAPE = [512, 1024]


def load_frames(path, n):
    """Raw (pre-contrast) polar image + mask + GT boxes for the first n frames."""
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = POLAR_SHAPE
    cfg.INPUT_DATASET = path
    if detect_dataset_type(path) == 'pygid':
        ds = PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                          buffer_size=3, load_labels=True)
    else:
        ds = H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                             buffer_size=3)
    frames = []
    for ic in ds.iter_images():
        raw = np.asarray(ic.raw_polar_image, dtype=np.float32)
        mask = np.asarray(ic.converted_mask).reshape(raw.shape).astype(bool)
        frames.append((raw, mask, np.array(ic.polar_labels.boxes),
                       getattr(ic.polar_labels, 'img_name', None) or f'frame{len(frames)}'))
        if len(frames) >= n:
            break
    return frames


def sweep_scores(csv_paths):
    """{(dataset, setting_label): ap_total} from whatever sweep CSVs exist."""
    out = {}
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    out[(row['dataset'], row['setting'])] = float(row['ap_total'])
                except (KeyError, ValueError):
                    pass
    return out


def pick_settings(a, scores):
    if a.settings:
        return [by_label(s) for s in a.settings]
    if a.top:
        ranked = sorted((v, k[1]) for k, v in scores.items() if k[0] == a.by)
        if not ranked:
            raise SystemExit(f'no sweep rows for dataset {a.by!r}; run the sweep first')
        return [by_label(n) for _, n in ranked[::-1][:a.top]]
    #default: exactly the contrasts the multi-contrast run trains on
    return [{k: v for k, v in c.items() if k != 'name'} for c in CONTRAST_CHANNELS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='organic', help="'41', 'organic', or a path to a .h5")
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--settings', nargs='*', default=None, help='canonical sweep labels')
    p.add_argument('--top', type=int, default=None, help='take the best N settings from the sweep CSV')
    p.add_argument('--by', default='organic', help='dataset whose ap_total ranks --top')
    p.add_argument('--crop', nargs=4, type=int, default=None, metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help='polar-pixel window: x is q (0-1024), y is chi (0-512)')
    p.add_argument('--out', default=f'{SWEEPS}/images')
    p.add_argument('--dpi', type=int, default=100)
    a = p.parse_args()

    path = DATASETS.get(a.dataset, a.dataset)
    scores = sweep_scores([f'{SWEEPS}/contrast_ssl1.csv', f'{SWEEPS}/contrast_ssl1_extraclips.csv'])
    settings = pick_settings(a, scores)
    os.makedirs(a.out, exist_ok=True)

    print(f'{path}\n{len(settings)} contrasts x {a.frames} frames -> {a.out}')
    frames = load_frames(path, a.frames)

    for i, (raw, mask, boxes, name) in enumerate(frames):
        n = len(settings)
        fig, axes = plt.subplots(n, 1, figsize=(16, 8 * n))
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, settings):
            img = apply_contrast(raw, mask, s)
            if a.crop:
                x0, x1, y0, y1 = a.crop
                ax.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1,
                          extent=(x0, x1, y1, y0), aspect='auto')
            else:
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='auto')
            for b in boxes:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, edgecolor='lime', lw=0.8))
            lab = label(s)
            ap = scores.get((a.dataset, lab))
            ax.set_title(lab + (f'   (sweep ap_total on {a.dataset}: {ap:.4f})' if ap else ''),
                         fontsize=13)
            ax.set_xlabel('q (polar px)'); ax.set_ylabel('chi (polar px)')
            if a.crop:
                ax.set_xlim(a.crop[0], a.crop[1]); ax.set_ylim(a.crop[3], a.crop[2])
        fig.suptitle(f'{a.dataset}  frame {i}: {name}   ({len(boxes)} labeled peaks, green)',
                     fontsize=15, y=0.999)
        fig.tight_layout()
        f = os.path.join(a.out, f'{a.dataset}_frame{i}{"_crop" if a.crop else ""}.png')
        fig.savefig(f, dpi=a.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {f}')


if __name__ == '__main__':
    main()
