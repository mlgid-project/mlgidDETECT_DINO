"""Does the top-k cap in onnx_to_xyxy bind on the physics-sim models?

`main.evaluate_giwaxs_ap` calls `onnx_to_xyxy(config, container, raw_results)` with NO
num_select, so every per-epoch `exp_ap_*.txt` curve we rank runs by is at the function
default 225 (`util/postprocessing.py:32`). The config's `num_select = 150`
(`DINO_4scale_swin.py:102`) feeds `PostProcess` in `dino.py` and is never used on this path.

`diagnostics/postproc_diag.py` measured 225 -> 450 as organic +0.017 on ssl1 and +0.0035 on
lr4e5 (41 -0.006), i.e. strongly model-dependent, and never took the gain. The physics-sim
models are trained on far denser frames (branch B is ~98 segments), so the cap should bind
harder on them. This re-measures on the physics checkpoints.

Runs each checkpoint ONCE over both labeled gates, caches the raw (pred_logits, pred_boxes),
then replays the deployed postprocessing at every num_select off that cache for free -- so
the sweep costs one forward pass, not one per setting.

  python diagnostics/numselect_sweep.py name=/path/to/run/checkpoint.pth [name2=... ...]

Needs a GPU. Result 2026-09-15: 225 is a joint optimum on all four checkpoints measured; see
the table in the commit message. Do not re-propose raising it as a free win.
"""
import os, sys, json, argparse, copy
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from main import build_model_main
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results
import util.misc as utils

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DATASETS = {'41': f'{CUR}/41.h5', 'organic': f'{CUR}/organic_labeled.h5'}
#900 queries x 2 classes = 1800 entries in the flattened topk grid, so 1800 == no cap at all.
SELECTS = [150, 225, 300, 450, 900, 1800]


def _eval_config(dset_path):
    """Byte-identical to main.evaluate_giwaxs_ap's Config block."""
    config = Config()
    config.EVAL_EPOCH = '0'
    config.EVAL_OUTPUT_FOLDER = os.environ.get('SWEEP_OUT', '/tmp')
    config.INPUT_DATASET = dset_path
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    return config


class _P:  # onnx_to_xyxy / filter_boxes only touch .boxes/.scores/.pred_labels
    pass


def load_model(ckpt_path):
    run_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(run_dir, 'config_args_all.json')) as f:
        args = argparse.Namespace(**json.load(f))
    model, _, _ = build_model_main(args)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck['model'] if 'model' in ck else ck
    out = model.load_state_dict(utils.clean_state_dict(sd), strict=False)
    if out.missing_keys:
        print(f'  !! {len(out.missing_keys)} missing keys, first: {out.missing_keys[:3]}')
    model.cuda().eval()
    return model, args, ck.get('epoch', '?')


def cache_raw(model, args, dset_path):
    """One forward pass per frame; keep the raw head outputs and the GT."""
    config = _eval_config(dset_path)
    if detect_dataset_type(dset_path) == 'pygid':
        data = PyGIDDataset(config, path=dset_path, preprocess_func=standard_preprocessing,
                            buffer_size=5, load_labels=True)
    else:
        data = H5GIWAXSDataset(config, path=dset_path, preprocess_func=standard_preprocessing,
                               buffer_size=5)
    cache = []
    for cont in data.iter_images():
        img = cont.converted_polar_image
        img = torch.tensor(img[:, 0, :, :]).unsqueeze(0).cuda().repeat(1, args.num_channels, 1, 1)
        with torch.no_grad():
            o = model(img)
        labels = cont.polar_labels
        cache.append((o['pred_logits'].detach().cpu().numpy(),
                      o['pred_boxes'].detach().cpu().numpy(),
                      np.asarray(labels.boxes), np.asarray(labels.confidences)))
    return cache, config


def replay(cache, config, num_select):
    ev = Evaluator()
    n_kept = []
    for logits, boxes, gt_boxes, gt_conf in cache:
        c = _P()
        c = onnx_to_xyxy(config, c, [logits, boxes], num_select=num_select)
        c = filter_boxes(config, c)
        n_kept.append(len(c.boxes))
        ev.get_exp_metrics(c.boxes, c.scores, torch.tensor(gt_boxes), gt_conf)
    _, df2 = get_full_conf_results(ev.metrics)
    return float(df2['ap_total'].values[0]), float(np.mean(n_kept))


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    for name, ckpt in targets:
        model, args, epoch = load_model(ckpt)
        print(f'\n===== {name}  (epoch {epoch})  {ckpt}', flush=True)
        for dname, dpath in DATASETS.items():
            cache, config = cache_raw(model, args, dpath)
            print(f'  -- {dname}: {len(cache)} frames, '
                  f'{np.mean([len(c[2]) for c in cache]):.1f} GT boxes/frame', flush=True)
            rows = [(ns,) + replay(cache, config, ns) for ns in SELECTS]
            base = dict((ns, ap) for ns, ap, _ in rows)[225]
            for ns, ap, kept in rows:
                tag = '  <- deployed' if ns == 225 else ''
                print(f'     num_select {ns:5d}   ap_total {ap:.4f}   '
                      f'delta_vs_225 {ap - base:+.4f}   kept/frame {kept:6.1f}{tag}', flush=True)
        del model
        torch.cuda.empty_cache()
