"""Does the MODEL predict boxes that overlap the mask, at the rate the labels do?

This is the prediction-side counterpart to the GT measurement: real labels put 34.8% (41) /
20.1% (organic) of their boxes across an invalid pixel. If the model's predictions fall far
short of that, it is systematically refusing to detect cut-off peaks -- which is the failure
`SimulationConfig.edge_peaks` is meant to address.

CPU only (onnxruntime):  python diagnostics/pred_mask_overlap.py [model.onnx ...]
"""
import os, sys
import numpy as np
import onnxruntime as rt
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.postprocessing import onnx_to_xyxy, filter_boxes

CUR = '/mnt/lustre/work/schreiber/szb389/datasets'
DATASETS = {'41': f'{CUR}/41.h5', 'organic': f'{CUR}/organic_labeled.h5'}
ONNX = sys.argv[1:] or [f'{CUR}/DINO_BACKBONE_curation/onnx/dino_ssl1.onnx',
                        f'{CUR}/DINO_BACKBONE_curation/onnx/dino_lr4e5.onnx']


class _P:  # onnx_to_xyxy / filter_boxes only touch .boxes/.scores/.pred_labels
    pass


def frac_overlapping(boxes, mask):
    """Fraction of boxes covering at least one invalid pixel."""
    n = hit = 0
    H, W = mask.shape
    for x0, y0, x1, y1 in np.asarray(boxes):
        xi0, yi0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
        xi1, yi1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
        if xi1 <= xi0 or yi1 <= yi0:
            continue
        n += 1
        sub = mask[yi0:yi1, xi0:xi1]
        if sub.size and not sub.all():
            hit += 1
    return n, hit


print(f'{"model":22s} {"set":9s} {"GT boxes":>9s} {"GT overlap":>12s} {"pred boxes":>11s} {"pred overlap":>14s}')
for onnx in ONNX:
    sess = rt.InferenceSession(onnx, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    for name, path in DATASETS.items():
        cfg = Config(); cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]; cfg.INPUT_DATASET = path
        cfg.POSTPROCESSING_SCORE = 0.4          # the deployed operating point
        cfg.POSTPROCESSING_CLASSAWARE_NMS = True
        ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=3, load_labels=True)
              if detect_dataset_type(path) == 'pygid' else
              H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=3))
        gtn = gth = pn = ph = 0
        for ic in ds.iter_images():
            m = np.asarray(ic.converted_mask).reshape(512, 1024).astype(bool)
            a, b = frac_overlapping(ic.polar_labels.boxes, m); gtn += a; gth += b
            x = np.asarray(ic.converted_polar_image, np.float32).reshape(1, -1, 512, 1024)[:, :1]
            raw = sess.run(None, {iname: x})
            gc = filter_boxes(cfg, onnx_to_xyxy(cfg, _P(), raw))
            a, b = frac_overlapping(gc.boxes.numpy(), m); pn += a; ph += b
        print(f'{os.path.basename(onnx):22s} {name:9s} {gtn:9d} {100*gth/max(gtn,1):11.1f}% '
              f'{pn:11d} {100*ph/max(pn,1):13.1f}%', flush=True)
