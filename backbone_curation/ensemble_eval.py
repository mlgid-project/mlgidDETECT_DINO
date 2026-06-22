"""
Ensemble two DINO detector checkpoints and compare vs each alone, on a labeled eval set.

Motivation: the round-1 SSL detector (ssl1) and the from-scratch baseline are COMPLEMENTARY —
baseline wins 41 (0.768 vs 0.762), ssl1 wins organic (0.586 vs 0.554). An ensemble might beat
either on both. This script verifies that, cheaply, with no retraining.

Method (detection-level fusion — DETR query slots aren't aligned across independently-trained
models, so we can't average raw outputs): for each image, run BOTH models, take each model's
top-225 detections (onnx_to_xyxy), POOL them, then run the SAME production class-aware NMS +
score filter (filter_boxes) over the union. Scores are comparable (both sigmoid logits in [0,1]).
Everything else (preprocessing, postprocessing, Evaluator, q-matcher) is identical to the
standard --eval path (main.evaluate_giwaxs_ap), so the three numbers are directly comparable.

It prints ap_high/med/low/total for model A, model B, and the ENSEMBLE, plus the ensemble's
delta vs the best single model — run it on each eval set to verify.

Usage (needs a GPU — builds two swin-L models):
  PYTHONPATH=<repo> python backbone_curation/ensemble_eval.py --eval_file <41.h5 | organic.h5>
  # defaults point at ssl1 + baseline checkpoint.pth; override with --ckpt_a/--ckpt_b/--name_*

NOTE: checkpoint.pth is each run's LAST epoch (not its best-AP epoch), so single-model numbers
may read a touch below the reported bests — but A, B, and ENSEMBLE all use the same checkpoints,
so the comparison is internally valid.
"""
import argparse, os, sys
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
DEF_A = f"{CUR}/detector_runs/dino_ssl1/checkpoint.pth"
DEF_B = "/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth"


def build_model_from_ckpt(config_file, ckpt_path, device):
    """Build the DINO model from a config (architecture only) and load a trained checkpoint."""
    parser = get_args_parser()
    a = parser.parse_args(['-c', config_file, '--output_dir', '/tmp/ens_eval', '--eval'])
    cfg = SLConfig.fromfile(a.config_file)
    for k, v in cfg._cfg_dict.to_dict().items():
        if not hasattr(a, k):
            setattr(a, k, v)
    # post-parse defaults main() sets outside the argparser (main.py:500 etc.)
    for k, dv in [('export', False), ('use_ema', False), ('debug', False), ('num_channels', 1)]:
        if not hasattr(a, k):
            setattr(a, k, dv)
    model, _, _ = build_model_main(a)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    out = model.load_state_dict(sd, strict=False)
    print(f"  loaded {os.path.basename(os.path.dirname(ckpt_path))}/{os.path.basename(ckpt_path)}: "
          f"missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
    model.to(device).eval()
    return model, a


def _iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def wbf_fuse(box_lists, score_lists, label_lists, iou_thr=0.55, n_models=2):
    """Weighted Boxes Fusion (Solovyev et al.): cluster overlapping same-class boxes across
    models, fuse each cluster into a confidence-weighted-average box; cluster confidence =
    mean score * (models_in_cluster / n_models) (down-weights single-model detections)."""
    dets = []
    for mi, (b, s, l) in enumerate(zip(box_lists, score_lists, label_lists)):
        bl, sl, ll = b.tolist(), s.tolist(), l.tolist()
        for i in range(len(bl)):
            dets.append((sl[i], bl[i], int(ll[i]), mi))
    dets.sort(key=lambda d: -d[0])
    clusters = []
    for sc, box, lb, mdl in dets:
        best, best_i = None, iou_thr
        for c in clusters:
            if c['label'] == lb and _iou_xyxy(box, c['fused']) > best_i:
                best_i, best = _iou_xyxy(box, c['fused']), c
        if best is None:
            clusters.append({'boxes': [box], 'scores': [sc], 'models': {mdl}, 'label': lb, 'fused': list(box)})
        else:
            best['boxes'].append(box); best['scores'].append(sc); best['models'].add(mdl)
            tot = sum(best['scores'])
            best['fused'] = [sum(bx[k]*ss for bx, ss in zip(best['boxes'], best['scores']))/tot for k in range(4)]
    fb, fs, fl = [], [], []
    for c in clusters:
        tot = sum(c['scores'])
        fb.append([sum(bx[k]*ss for bx, ss in zip(c['boxes'], c['scores']))/tot for k in range(4)])
        fs.append((sum(c['scores'])/len(c['scores'])) * (min(len(c['models']), n_models)/n_models))
        fl.append(c['label'])
    if not fb:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(fb), torch.tensor(fs), torch.tensor(fl, dtype=torch.long)


def topk_dets(config, gc, model, img):
    """raw model outputs -> top-225 xyxy boxes/scores/labels (production onnx_to_xyxy)."""
    out = model(img)
    raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
    onnx_to_xyxy(config, gc, raw)
    return gc.boxes.clone(), gc.scores.clone(), gc.pred_labels.clone()


def nms_filter(config, gc, boxes, scores, labels):
    """production class-aware NMS + score threshold over the given detection pool."""
    gc.boxes, gc.scores, gc.pred_labels = boxes, scores, labels
    filter_boxes(config, gc)
    return gc.boxes, gc.scores


def ap_row(ev, tag):
    _, df2 = get_full_conf_results(ev.metrics)
    r = df2.iloc[0]
    print(f"  {tag:9s}  ap_total={r['ap_total']:.4f}   high={r['ap_high']:.4f}  med={r['ap_med']:.4f}  low={r['ap_low']:.4f}")
    return float(r['ap_total'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_file', required=True, help='labeled .h5 (41 = roi_data, organic = pygid)')
    p.add_argument('--ckpt_a', default=DEF_A)
    p.add_argument('--ckpt_b', default=DEF_B)
    p.add_argument('--name_a', default='ssl1')
    p.add_argument('--name_b', default='baseline')
    p.add_argument('--config', default=os.path.join(_REPO, 'config/DINO/DINO_4scale_swin.py'))
    p.add_argument('--fusion', choices=['nms', 'wbf'], default='nms', help='ensemble fusion method')
    p.add_argument('--wbf_iou', type=float, default=0.55)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: no CUDA — this will be very slow.")

    print(f"building model A ({args.name_a})"); modelA, a = build_model_from_ckpt(args.config, args.ckpt_a, device)
    print(f"building model B ({args.name_b})"); modelB, _ = build_model_from_ckpt(args.config, args.ckpt_b, device)

    # postprocessing config — mirror main.evaluate_giwaxs_ap exactly
    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = args.eval_file

    if detect_dataset_type(args.eval_file) == 'pygid':
        data = PyGIDDataset(config, path=args.eval_file, preprocess_func=standard_preprocessing,
                            buffer_size=5, load_labels=True)
    else:
        data = H5GIWAXSDataset(config, path=args.eval_file, preprocess_func=standard_preprocessing,
                               buffer_size=5)

    evA, evB, evE = Evaluator(), Evaluator(), Evaluator()
    n = 0
    with torch.no_grad():
        for gc in data.iter_images():
            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device).repeat(1, a.num_channels, 1, 1)
            gt_boxes = torch.tensor(gc.polar_labels.boxes)
            gt_conf = gc.polar_labels.confidences
            bA, sA, lA = topk_dets(config, gc, modelA, img)
            bB, sB, lB = topk_dets(config, gc, modelB, img)
            pb, ps = nms_filter(config, gc, bA.clone(), sA.clone(), lA.clone()); evA.get_exp_metrics(pb, ps, gt_boxes, gt_conf)
            pb, ps = nms_filter(config, gc, bB.clone(), sB.clone(), lB.clone()); evB.get_exp_metrics(pb, ps, gt_boxes, gt_conf)
            if args.fusion == 'wbf':
                fb, fs, fl = wbf_fuse([bA, bB], [sA, sB], [lA, lB], iou_thr=args.wbf_iou, n_models=2)
                keep = fs > config.POSTPROCESSING_SCORE
                evE.get_exp_metrics(fb[keep], fs[keep], gt_boxes, gt_conf)
            else:
                pb, ps = nms_filter(config, gc, torch.cat([bA, bB]), torch.cat([sA, sB]), torch.cat([lA, lB]))
                evE.get_exp_metrics(pb, ps, gt_boxes, gt_conf)
            n += 1

    print(f"\n===== {os.path.basename(args.eval_file)}  ({n} frames)  fusion={args.fusion} =====")
    apA = ap_row(evA, args.name_a)
    apB = ap_row(evB, args.name_b)
    apE = ap_row(evE, f'ENS-{args.fusion}')
    best = max(apA, apB); who = args.name_a if apA >= apB else args.name_b
    verdict = 'WIN' if apE > best + 1e-4 else ('tie' if apE > best - 1e-4 else 'no gain')
    print(f"\n  ensemble vs best single ({who} {best:.4f}):  {apE - best:+.4f}   -> {verdict}")


if __name__ == '__main__':
    main()
