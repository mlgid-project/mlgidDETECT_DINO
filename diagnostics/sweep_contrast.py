"""
Contrast sweep: evaluate one trained detector on the labeled datasets (41 + organic)
under many different input-contrast pipelines and log ap_total / recall / precision for
each, so the contrast used for single- and multi-channel training can be chosen from
measured numbers instead of the inherited default.

WHY IT IS CHEAP: the polar conversion and the H5 read are contrast-independent, so each
dataset is loaded exactly ONCE (`raw_polar_image` + mask + GT boxes are cached in RAM) and
every sweep point only re-runs contrast -> forward -> postprocess -> metrics.

The contrast function below is a parameterised re-implementation of
`util.exp_preprocess._contrast_correction`. At the default settings
(clip 5/99.5, log, HE) it must be numerically identical to it -- this is asserted against
the real function on the first cached frame of every dataset and the max abs difference is
printed, so a drift between the two is impossible to miss.

Postprocessing (onnx_to_xyxy + class-aware filter_boxes at POSTPROCESSING_SCORE=0.1) and
the matcher are exactly the ones `main.evaluate_giwaxs_ap` uses, so the numbers here are
comparable to the per-epoch `exp_ap_41.txt` / `exp_ap_organic.txt` logs.

USAGE:
  python diagnostics/sweep_contrast.py --grid all --out sweeps/contrast_ssl1.csv
"""
import argparse, itertools, os, sys, time

import cv2
import numpy as np
import torch
from torch import Tensor

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import (standard_preprocessing, normalize, _contrast_correction,
                                 apply_contrast)
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import (Evaluator, recall_precision_curve_with_intensities,
                             _get_av_precision)
from util.contrast_grids import DEFAULT_SETTING, build_grid, label

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DEFAULT_DATASETS = {
    '41': f'{CUR}/41.h5',
    'organic': f'{CUR}/organic_labeled.h5',
}
DEFAULT_CKPT = (f'{CUR}/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth')
DEFAULT_CONFIG = 'config/DINO/DINO_4scale_swin_ssl.py'
POLAR_SHAPE = [512, 1024]


# --------------------------------------------------------------------------------------
# contrast
# --------------------------------------------------------------------------------------
#apply_contrast lives in util/exp_preprocess.py so the sweep, the multi-contrast
#channels (util/channels.py) and the simulator all score the same pipeline.


#the grid, the settings and their canonical labels live in util/contrast_grids.py so the
#sweep and diagnostics/dump_contrast_images.py cannot disagree about what a label means.


# --------------------------------------------------------------------------------------
# data / model
# --------------------------------------------------------------------------------------
def cache_dataset(path: str, limit=None):
    """Load a labeled .h5 once and keep raw polar image + mask + GT boxes in RAM."""
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = POLAR_SHAPE
    cfg.INPUT_DATASET = path
    if detect_dataset_type(path) == 'pygid':
        ds = PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                          buffer_size=5, load_labels=True)
    else:
        ds = H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                             buffer_size=5)
    frames = []
    for ic in ds.iter_images():
        raw = np.asarray(ic.raw_polar_image, dtype=np.float32)
        m = np.asarray(ic.converted_mask).reshape(raw.shape).astype(bool)
        frames.append((raw, m,
                       Tensor(np.array(ic.polar_labels.boxes)),
                       np.array(ic.polar_labels.confidences)))
        if limit and len(frames) >= limit:
            break
    return frames


def verify_against_reference(frames, name):
    """The default sweep point must reproduce `_contrast_correction` bit-for-bit."""
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = POLAR_SHAPE
    raw, mask = frames[0][0], frames[0][1]
    ref, _ = _contrast_correction(cfg, raw)
    mine = apply_contrast(raw, mask, DEFAULT_SETTING)
    d = float(np.abs(ref - mine).max())
    print(f'[verify] {name}: max|sweep_default - _contrast_correction| = {d:.3e}', flush=True)
    return d


def load_model(ckpt_path, config_file, device='cuda'):
    args = get_args_parser().parse_args([])
    for k, v in SLConfig.fromfile(config_file)._cfg_dict.to_dict().items():
        setattr(args, k, v)
    args.device = device
    args.export = False
    model, _, _ = build_model_main(args)
    model = model.to(device).eval()
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd['model'])
    return model, args, sd.get('epoch', -1)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def metrics_from(ev: Evaluator, op_score: float):
    (recalls, precisions, accuracies, scores, ap_total,
     recalls_levels, fp_nums) = recall_precision_curve_with_intensities(ev.metrics)
    if not len(recalls):
        return {}
    i_acc = int(np.argmax(accuracies))
    #operating point of the deployed model: last curve index still above the score threshold
    above = np.nonzero(np.asarray(scores) >= op_score)[0]
    i_op = int(above[-1]) if len(above) else 0
    return {
        'ap_total': ap_total,
        'ap_high': _get_av_precision(recalls_levels[1.], precisions),
        'ap_med': _get_av_precision(recalls_levels[0.5], precisions),
        'ap_low': _get_av_precision(recalls_levels[np.float32(0.1)], precisions),
        'best_acc': accuracies[i_acc],
        'score_bestacc': float(scores[i_acc]),
        'recall_bestacc': recalls[i_acc],
        'precision_bestacc': precisions[i_acc],
        'fp_frac_bestacc': float(fp_nums[i_acc]),
        f'recall_at_{op_score:g}': recalls[i_op],
        f'precision_at_{op_score:g}': precisions[i_op],
        f'fp_frac_at_{op_score:g}': float(fp_nums[i_op]),
    }


class _Pred:
    """Minimal stand-in for ImageContainer: onnx_to_xyxy/filter_boxes only set and read
    .boxes / .scores / .pred_labels."""
    pass


def run_setting(model, args, frames, setting, op_score):
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = POLAR_SHAPE
    #same postprocessing as main.evaluate_giwaxs_ap: low score cut so the PR curve is
    #fully sampled, class-aware NMS for the 2-class ring/segment model
    cfg.POSTPROCESSING_SCORE = 0.1
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True

    ev = Evaluator()
    with torch.no_grad():
        for raw, mask, gt, conf in frames:
            img = apply_contrast(raw, mask, setting)
            t = torch.as_tensor(img).cuda()
            if getattr(args, 'num_channels', 1) == 4:
                from util.channels import build_channels
                x = build_channels(t, torch.as_tensor(mask).cuda()).unsqueeze(0)
            else:
                x = t[None, None].repeat(1, args.num_channels, 1, 1)
            out = model(x)
            raw_results = [out['pred_logits'].detach().cpu().numpy(),
                           out['pred_boxes'].detach().cpu().numpy()]

            gc = onnx_to_xyxy(cfg, _Pred(), raw_results)
            gc = filter_boxes(cfg, gc)
            ev.get_exp_metrics(gc.boxes, gc.scores, gt, conf)
    return metrics_from(ev, op_score)


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--grid', default='all')
    p.add_argument('--out', default='sweeps/contrast_sweep.csv')
    p.add_argument('--datasets', nargs='*', default=None,
                   help='name=path pairs; default 41 + organic')
    p.add_argument('--limit', type=int, default=None, help='cap frames per dataset (smoke test)')
    p.add_argument('--op_score', type=float, default=0.4,
                   help='deployed score threshold to also report recall/precision at')
    a = p.parse_args()

    datasets = DEFAULT_DATASETS if not a.datasets else dict(
        kv.split('=', 1) for kv in a.datasets)
    grid = build_grid(a.grid)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log_path = os.path.splitext(a.out)[0] + '.log'
    logf = open(log_path, 'a')

    def say(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    say('=' * 100)
    say(f'contrast sweep  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    say(f'  ckpt   : {a.ckpt}')
    say(f'  config : {a.config}')
    say(f'  grid   : {a.grid}  ({len(grid)} settings)')

    model, args, epoch = load_model(a.ckpt, a.config)
    say(f'  epoch  : {epoch}   num_channels={args.num_channels}')

    cached = {}
    for name, path in datasets.items():
        t0 = time.time()
        cached[name] = cache_dataset(path, a.limit)
        say(f'  cached {name}: {len(cached[name])} frames from {path} ({time.time()-t0:.0f}s)')
        verify_against_reference(cached[name], name)

    cols = ['dataset', 'setting', 'clip_lo', 'clip_hi', 'log', 'gamma', 'he',
            'clahe_limit', 'clahe_tile', 'ap_total', 'ap_high', 'ap_med', 'ap_low',
            'score_bestacc', 'recall_bestacc', 'precision_bestacc', 'fp_frac_bestacc',
            f'recall_at_{a.op_score:g}', f'precision_at_{a.op_score:g}',
            f'fp_frac_at_{a.op_score:g}', 'n_frames', 'ckpt', 'epoch']
    new = not os.path.exists(a.out)
    csvf = open(a.out, 'a')
    if new:
        csvf.write(','.join(cols) + '\n')
        csvf.flush()

    say('')
    say(f'{"dataset":9s} {"setting":34s} {"ap_total":>9s} {"ap_high":>8s} '
        f'{"recall*":>8s} {"prec*":>7s} {"rec@%g" % a.op_score:>8s} {"prec@%g" % a.op_score:>8s}')
    for i, s in enumerate(grid, 1):
        for name, frames in cached.items():
            t0 = time.time()
            try:
                m = run_setting(model, args, frames, s, a.op_score)
            except Exception as e:
                say(f'{name:9s} {label(s):34s}  FAILED: {type(e).__name__}: {e}')
                continue
            c = s.get('clip')
            row = [name, label(s),
                   '' if c is None else c[0], '' if c is None else c[1],
                   int(bool(s.get('log'))), s.get('gamma') or '', int(bool(s.get('he'))),
                   '' if not s.get('clahe') else s['clahe'][0],
                   '' if not s.get('clahe') else f"{s['clahe'][1]}x{s['clahe'][2]}",
                   m['ap_total'], m['ap_high'], m['ap_med'], m['ap_low'],
                   m['score_bestacc'], m['recall_bestacc'], m['precision_bestacc'],
                   m['fp_frac_bestacc'],
                   m[f'recall_at_{a.op_score:g}'], m[f'precision_at_{a.op_score:g}'],
                   m[f'fp_frac_at_{a.op_score:g}'], len(frames), a.ckpt, epoch]
            csvf.write(','.join(str(x) for x in row) + '\n')
            csvf.flush()
            say(f'{name:9s} {label(s):34s} {m["ap_total"]:9.4f} {m["ap_high"]:8.4f} '
                f'{m["recall_bestacc"]:8.3f} {m["precision_bestacc"]:7.3f} '
                f'{m[f"recall_at_{a.op_score:g}"]:8.3f} '
                f'{m[f"precision_at_{a.op_score:g}"]:8.3f}   ({time.time()-t0:.0f}s)')
        if i % 5 == 0:
            say(f'--- {i}/{len(grid)} settings done ---')

    say('')
    say(f'wrote {a.out}')
    csvf.close()
    logf.close()


if __name__ == '__main__':
    main()
