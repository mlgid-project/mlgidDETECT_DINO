"""Where does the realbkg 41 gap come from?  Recall or precision, rings or segments, which frames.

Every realbkg run sits at 41 ap_total 0.38-0.45 against lr4e5_1's 0.761, while matching or beating
it on organic. physics-sim-recall-not-precision found the physics/41 gap was PRECISION at equal
recall (physics4 recall 0.856 vs 0.888, precision 0.256 vs 0.445). The realbkg sim is a different
simulator (real donor backgrounds, ~3.4 rings/frame against 41's 16.9) and nobody has split its
gap. This does, read-only, off one forward pass per checkpoint per gate:

  * ap_total (re-computed, must equal the run's exp_ap log at that epoch)
  * recall / precision at the DEPLOYED point (top-225, class-aware NMS, score > 0.1)
  * recall split by GEOMETRIC ring vs segment -- 41.h5 never populates is_ring, so the
    validated criterion (box spans >= 70% of the valid chi rows at its radius) is used
  * recall uncapped (top-1800) -- does the model see peaks it does not rank?
  * median TP and FP score, and their separation
  * FPs split: near a GT (IoU >= 0.1 with some GT box) vs in background; ring-shaped vs not
  * per-frame recall/precision, so the frames carrying the gap can be named

  python diagnostics/gap41_split.py name=/path/checkpoint.pth [...]   (CPU or GPU)
"""
import os, sys, json, argparse
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
from util.matchers import get_matcher
import util.misc as utils

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DATASETS = {'41': f'{CUR}/41.h5', 'organic': f'{CUR}/organic_labeled.h5'}
POLAR = (512, 1024)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cpu':
    # The compiled MSDeformAttn kernel is CUDA-only ("Not implemented on the CPU"). Swap in the
    # repo's own pure-PyTorch reference implementation, in THIS process only -- no repo file
    # changes. Correctness check: run6_best41 must reproduce its logged epoch-18 41 AP (0.4022).
    import models.dino.ops.modules.ms_deform_attn as _msda
    from models.dino.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

    class _CPUDeformAttn:
        @staticmethod
        def apply(value, shapes, level_start_index, sampling_locations, attention_weights, step):
            return ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
    _msda.MSDeformAttnFunction = _CPUDeformAttn
matcher = get_matcher('q', min_iou=0.1)   # what Evaluator(match_criterion='q', 0.1) builds


class _P:
    pass


def _config(path):
    c = Config()
    c.EVAL_EPOCH = '0'; c.EVAL_OUTPUT_FOLDER = os.environ.get('SWEEP_OUT', '/tmp')
    c.INPUT_DATASET = path; c.PREPROCESSING_POLAR_SHAPE = list(POLAR)
    c.POSTPROCESSING_SCORE = 0.1; c.POSTPROCESSING_CLASSAWARE_NMS = True
    return c


def load_model(ckpt):
    with open(os.path.join(os.path.dirname(ckpt), 'config_args_all.json')) as f:
        args = argparse.Namespace(**json.load(f))
    model, _, _ = build_model_main(args)
    ck = torch.load(ckpt, map_location='cpu')
    out = model.load_state_dict(utils.clean_state_dict(ck.get('model', ck)), strict=False)
    if out.missing_keys:
        print(f'  !! {len(out.missing_keys)} missing keys, first: {out.missing_keys[:3]}')
    return model.to(DEV).eval(), args, ck.get('epoch', '?')


def cache(model, args, path):
    cfg = _config(path)
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True)
          if detect_dataset_type(path) == 'pygid' else
          H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    fr = []
    try:
        for ic in ds.iter_images():
            img = torch.tensor(ic.converted_polar_image[:, 0]).unsqueeze(0).to(DEV)
            img = img.repeat(1, args.num_channels, 1, 1)
            with torch.no_grad():
                o = model(img)
            fr.append(dict(logits=o['pred_logits'].cpu().numpy(), boxes=o['pred_boxes'].cpu().numpy(),
                           mask=np.asarray(ic.converted_mask).reshape(*POLAR).astype(bool),
                           gt=np.asarray(ic.polar_labels.boxes, np.float32),
                           gtconf=np.asarray(ic.polar_labels.confidences, np.float32)))
    finally:
        if callable(getattr(ds, 'close', None)):
            ds.close()
    return fr, cfg


def is_ring_geom(boxes, mask, frac=0.70):
    """postproc_diag.is_ring_geom: box spans >= frac of the valid chi rows at its radius."""
    H, W = mask.shape
    out = []
    for x0, y0, x1, y1 in np.asarray(boxes):
        xc = int(np.clip((x0 + x1) / 2, 0, W - 1))
        out.append((y1 - y0) >= frac * max(int(mask[:, xc].sum()), 1))
    return np.array(out, bool)


def iou_matrix(a, b):
    a = np.asarray(a, np.float64)[:, None]; b = np.asarray(b, np.float64)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    ua = (a[..., 2]-a[..., 0])*(a[..., 3]-a[..., 1]) + (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter / np.maximum(ua - inter, 1e-9)


def analyse(fr, cfg, num_select=225):
    ev = Evaluator()
    S = dict(gt=0, tp=0, pred=0, gt_r=0, tp_r=0, gt_s=0, tp_s=0, fp_near=0, fp_bg=0, fp_ringshape=0)
    tps, fps, per = [], [], []
    for f in fr:
        c = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), [f['logits'], f['boxes']], num_select=num_select))
        pb = np.asarray(c.boxes, np.float32).reshape(-1, 4); sc = np.asarray(c.scores, np.float32)
        gt = f['gt']; ring = is_ring_geom(gt, f['mask']) if len(gt) else np.zeros(0, bool)
        ev.get_exp_metrics(c.boxes, c.scores, torch.tensor(gt), f['gtconf'])
        if len(pb) and len(gt):
            _, ri, ci = matcher(torch.tensor(gt).float(), torch.tensor(pb).float())
            ri, ci = np.asarray(ri, int), np.asarray(ci, int)
        else:
            ri = ci = np.zeros(0, int)
        fp = np.setdiff1d(np.arange(len(pb)), ci)
        S['gt'] += len(gt); S['tp'] += len(ri); S['pred'] += len(pb)
        S['gt_r'] += int(ring.sum()); S['tp_r'] += int(ring[ri].sum())
        S['gt_s'] += int((~ring).sum()); S['tp_s'] += int((~ring[ri]).sum())
        if len(fp):
            near = (iou_matrix(pb[fp], gt).max(1) >= 0.1) if len(gt) else np.zeros(len(fp), bool)
            S['fp_near'] += int(near.sum()); S['fp_bg'] += int((~near).sum())
            S['fp_ringshape'] += int(is_ring_geom(pb[fp], f['mask']).sum())
        tps += list(sc[ci]); fps += list(sc[fp])
        per.append(dict(n_gt=len(gt), ring_frac=float(ring.mean()) if len(gt) else 0.0,
                        rec=len(ri)/max(len(gt), 1), prec=len(ri)/max(len(pb), 1),
                        n_fp=len(fp), valid=float(f['mask'].mean())))
    _, df2 = get_full_conf_results(ev.metrics)
    S['ap'] = float(df2['ap_total'].values[0])
    S['tp_med'] = float(np.median(tps)) if tps else float('nan')
    S['fp_med'] = float(np.median(fps)) if fps else float('nan')
    return S, per


def uncapped_recall(fr, cfg):
    n = t = 0
    for f in fr:
        c = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), [f['logits'], f['boxes']], num_select=1800))
        n += len(f['gt'])
        if len(c.boxes) and len(f['gt']):
            _, ri, _ = matcher(torch.tensor(f['gt']).float(), torch.as_tensor(c.boxes).float())
            t += len(ri)
    return t / max(n, 1)


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    print(f'device {DEV}', flush=True)
    perframe = {}
    for name, ckpt in targets:
        model, args, ep = load_model(ckpt)
        print(f'\n===== {name}  (epoch {ep})  {ckpt}', flush=True)
        for dn, dp in DATASETS.items():
            fr, cfg = cache(model, args, dp)
            S, per = analyse(fr, cfg)
            ur = uncapped_recall(fr, cfg)
            nf = len(fr); fpn = S['pred'] - S['tp']
            print(f'  -- {dn}: {nf} frames, {S["gt"]/nf:.1f} GT/frame '
                  f'({S["gt_r"]/nf:.1f} geometric rings, {S["gt_s"]/nf:.1f} segments)')
            print(f'     ap_total {S["ap"]:.4f} | recall {S["tp"]/S["gt"]:.3f} '
                  f'(rings {S["tp_r"]/max(S["gt_r"],1):.3f}, segments {S["tp_s"]/max(S["gt_s"],1):.3f}) '
                  f'| uncapped recall {ur:.3f}')
            print(f'     precision {S["tp"]/max(S["pred"],1):.3f} | preds/frame {S["pred"]/nf:.1f} '
                  f'| FP/frame {fpn/nf:.1f}: near-GT {S["fp_near"]/nf:.1f}, background {S["fp_bg"]/nf:.1f}, '
                  f'ring-shaped {S["fp_ringshape"]/nf:.1f}')
            print(f'     median score TP {S["tp_med"]:.3f} FP {S["fp_med"]:.3f} '
                  f'separation {S["tp_med"]-S["fp_med"]:.3f}', flush=True)
            perframe[(name, dn)] = per
        del model
        if DEV == 'cuda':
            torch.cuda.empty_cache()

    # per-frame on 41: first target is the reference, every other is compared to it
    names = [n for n, _ in targets]
    ref = names[0]
    for other in names[1:]:
        a, b = perframe[(ref, '41')], perframe[(other, '41')]
        d_rec = np.array([y['rec'] - x['rec'] for x, y in zip(a, b)])
        d_prec = np.array([y['prec'] - x['prec'] for x, y in zip(a, b)])
        rf = np.array([x['ring_frac'] for x in a]); ng = np.array([x['n_gt'] for x in a])
        print(f'\n===== 41 per frame: {other} minus {ref}')
        print(f'  recall delta: mean {d_rec.mean():+.3f}, frames worse by >0.1: {(d_rec < -0.1).sum()}/{len(a)}')
        print(f'  precision delta: mean {d_prec.mean():+.3f}, frames worse by >0.1: {(d_prec < -0.1).sum()}/{len(a)}')
        if len(a) > 2:
            from scipy.stats import spearmanr
            print(f'  spearman(recall delta, GT ring fraction) {spearmanr(d_rec, rf)[0]:+.2f}   '
                  f'(precision delta, ring fraction) {spearmanr(d_prec, rf)[0]:+.2f}   '
                  f'(recall delta, n_gt) {spearmanr(d_rec, ng)[0]:+.2f}')
        print('  frame  n_gt ring_frac | rec ref -> other | prec ref -> other | FP ref -> other')
        for i in np.argsort(d_rec + d_prec)[:10]:
            x, y = a[i], b[i]
            print(f'  {i:5d} {x["n_gt"]:5d} {x["ring_frac"]:9.2f} | {x["rec"]:.2f} -> {y["rec"]:.2f}      '
                  f'| {x["prec"]:.2f} -> {y["prec"]:.2f}       | {x["n_fp"]:3d} -> {y["n_fp"]:3d}')
