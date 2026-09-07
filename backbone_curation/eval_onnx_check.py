"""
Decisive parity check: run the DINO-repo eval code on the ONNX models (instead of the live
PyTorch checkpoint) and print AP. If this reproduces mlgidDETECT's number (~0.55 organic),
it proves the two eval pipelines are identical and the 0.605 gap is purely PyTorch-checkpoint
vs ONNX-export (MSDeformAttn CUDA kernel vs grid-sample core), not a pipeline bug.

Uses the SAME DINO-repo eval components ensemble_eval.py uses (PyGIDDataset/H5GIWAXSDataset,
onnx_to_xyxy, filter_boxes, Evaluator, polar-pixel matching) — only the model engine is ONNX.

Run in an env with onnxruntime (e.g. mlgiddetect-gpu) + PYTHONPATH=<DINO repo>:
  PYTHONPATH=<repo> python backbone_curation/eval_onnx_check.py --eval_file <41|organic .h5>
"""
import argparse, os, sys
import numpy as np
import torch
import onnxruntime as rt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
SSL1 = f"{CUR}/onnx/dino_ssl1.onnx"
BASE = f"{CUR}/onnx/dino_baseline.onnx"


def topk(config, gc, sess, x):
    raw = sess.run(None, {sess.get_inputs()[0].name: x})   # [pred_logits, pred_boxes]
    onnx_to_xyxy(config, gc, raw)
    return gc.boxes.clone(), gc.scores.clone(), gc.pred_labels.clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_file", required=True)
    p.add_argument("--ssl1", default=SSL1)
    p.add_argument("--base", default=BASE)
    p.add_argument("--histeq", default="true", choices=["true", "false"],
                   help="histogram equalization in preprocessing (dino.yaml sets this False)")
    a = p.parse_args()

    config = Config()
    config.PREPROCESSING_HISTOGRAMEQUALIZATION = (a.histeq == "true")
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = a.eval_file

    if detect_dataset_type(a.eval_file) == "pygid":
        data = PyGIDDataset(config, path=a.eval_file, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=True)
    else:
        data = H5GIWAXSDataset(config, path=a.eval_file, preprocess_func=standard_preprocessing, buffer_size=5)

    sA = rt.InferenceSession(a.ssl1, providers=["CPUExecutionProvider"])
    sB = rt.InferenceSession(a.base, providers=["CPUExecutionProvider"])

    evE, evA, evB = Evaluator(), Evaluator(), Evaluator()
    n = 0
    for gc in data.iter_images():
        x = np.asarray(gc.converted_polar_image, np.float32)
        x = x.reshape(1, -1, x.shape[-2], x.shape[-1])[:, :1]   # -> (1,1,512,1024)
        gt_b = torch.tensor(gc.polar_labels.boxes); gt_c = gc.polar_labels.confidences
        bA, scA, lA = topk(config, gc, sA, x)
        bB, scB, lB = topk(config, gc, sB, x)
        # single A
        gc.boxes, gc.scores, gc.pred_labels = bA.clone(), scA.clone(), lA.clone()
        filter_boxes(config, gc); evA.get_exp_metrics(gc.boxes, gc.scores, gt_b, gt_c)
        # single B
        gc.boxes, gc.scores, gc.pred_labels = bB.clone(), scB.clone(), lB.clone()
        filter_boxes(config, gc); evB.get_exp_metrics(gc.boxes, gc.scores, gt_b, gt_c)
        # ensemble
        gc.boxes = torch.cat([bA, bB]); gc.scores = torch.cat([scA, scB]); gc.pred_labels = torch.cat([lA, lB])
        filter_boxes(config, gc); evE.get_exp_metrics(gc.boxes, gc.scores, gt_b, gt_c)
        n += 1

    print(f"\n===== {os.path.basename(a.eval_file)} ({n} frames) — DINO eval code on ONNX models =====")
    for tag, ev in [("ssl1", evA), ("baseline", evB), ("ENSEMBLE", evE)]:
        _, df2 = get_full_conf_results(ev.metrics)
        r = df2.iloc[0]
        print(f"  {tag:9s} ap_total={r['ap_total']:.4f}  high={r['ap_high']:.4f} med={r['ap_med']:.4f} low={r['ap_low']:.4f}")


if __name__ == "__main__":
    main()
