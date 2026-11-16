"""Does evaluating REAL images in the quazipolar representation change AP?

mlgidDETECT has a genuine alternative geometry: `PREPROCESSING_QUAZIPOLAR` routes
`preprocess_geometry` through `calc_quazipolar_image` (exp_preprocess.py:303) instead of
`calc_polar_image`, with the matching GT box transform at labeleddataset.py:131.

NOTE this is NOT the same thing as the simulator's quazipolar DARK AREA (simulation.py:833),
which masks a wedge of a polar image and clamps the boxes to it. The checkpoints were trained on
polar images carrying a quazipolar-shaped dark region -- none of them has ever seen a
quazipolar-RESAMPLED image, so a drop here is a domain shift, not evidence about the dark area.

ONLY 41 IS VALID: 41.h5 is an h5giwaxs file and goes through H5GIWAXSDataset, which applies the
quazipolar GT transform. organic_labeled.h5 is a pygid file and PyGIDDataset has NO quazipolar
handling, so the image would be remapped and the labels left alone. Do not run organic here until
that transform is ported.

RESULT 2026-09-15: real 41 in quazipolar costs lr4e5 -0.015, ssl1 -0.006,
physics4 -0.032, physics5 -0.031, and the RANKING is unchanged. Lever killed. organic could not be
tested -- see the ONLY 41 IS VALID note below. See MODIFICATIONS.md section M.
"""
import os, sys
import numpy as np, torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, 'diagnostics'))
from numselect_sweep import load_model, _P
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results
from util.matchers import get_matcher

DSET = '/mnt/lustre/work/schreiber/szb389/datasets/41.h5'
matcher = get_matcher('q', min_iou=0.1)


def evaluate(model, args, quazipolar):
    config = Config()
    config.EVAL_EPOCH = '0'; config.EVAL_OUTPUT_FOLDER = '/tmp'
    config.INPUT_DATASET = DSET
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.PREPROCESSING_QUAZIPOLAR = bool(quazipolar)
    data = H5GIWAXSDataset(config, path=DSET, preprocess_func=standard_preprocessing, buffer_size=5)
    ev = Evaluator(); ngt = ntp = nfp = 0; nfr = 0
    for cont in data.iter_images():
        img = cont.converted_polar_image
        x = torch.tensor(img[:, 0, :, :]).unsqueeze(0).cuda().repeat(1, args.num_channels, 1, 1)
        with torch.no_grad():
            o = model(x)
        c = filter_boxes(config, onnx_to_xyxy(config, _P(),
              [o['pred_logits'].cpu().numpy(), o['pred_boxes'].cpu().numpy()], num_select=225))
        lab = cont.polar_labels
        t = torch.tensor(np.asarray(lab.boxes)).float()
        ev.get_exp_metrics(c.boxes, c.scores, t, lab.confidences)
        _, r, _cc = matcher(t, c.boxes)
        ngt += len(t); ntp += len(r); nfp += len(c.boxes) - len(r); nfr += 1
    _, df2 = get_full_conf_results(ev.metrics)
    return dict(ap=float(df2['ap_total'].values[0]), n=nfr, gt=ngt/nfr,
                recall=ntp/ngt, prec=ntp/max(ntp+nfp, 1), kept=(ntp+nfp)/nfr)


if __name__ == '__main__':
    print(f'{"model":>10} {"geometry":>11} {"frames":>7} {"GT/fr":>7} {"ap_total":>9} '
          f'{"recall":>8} {"prec":>8} {"kept/fr":>8}')
    for name, ckpt in [a.split('=', 1) for a in sys.argv[1:]]:
        model, args, epoch = load_model(ckpt)
        for q in (False, True):
            s = evaluate(model, args, q)
            print(f'{name:>10} {"quazipolar" if q else "polar":>11} {s["n"]:>7} {s["gt"]:>7.1f} '
                  f'{s["ap"]:>9.4f} {s["recall"]:>8.4f} {s["prec"]:>8.4f} {s["kept"]:>8.1f}',
                  flush=True)
        del model; torch.cuda.empty_cache()
