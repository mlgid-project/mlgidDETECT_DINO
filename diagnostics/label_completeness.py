"""
Step 1 — Is the organic eval label-limited?  (deployed ENSEMBLE, no training)

Prior finding (MODIFICATIONS.md "Physics wins validated & shelved"): the best single model's
confident false positives sit ON diffraction rings, ~93% within 8 px-q of a real labeled peak, at
UNLABELED angles -> they are probably real peaks the ground truth never annotated. If so, the
measured precision/AP is pessimistic and "faint recall" is partly unmeasurable.

This script quantifies that on the *actually deployed* model — the ssl1 + baseline ensemble
(detection-level NMS fusion, exactly as backbone_curation/ensemble_eval.py / ENSEMBLE_DEPLOY.md) —
and brackets how much the eval under-measures:

  - standard precision @score>0.3      (every unmatched detection = FP)
  - label-adjusted precision           (unmatched detections that sit on a labeled ring, i.e.
                                        q-distance to the nearest GT peak < ONRING_PX, are treated
                                        as IGNORE rather than FP -> upper bound on true precision)
  - fraction of FPs that are on-ring   (the "likely real unlabeled" share)
  - the same, restricted to HIGH-confidence FPs (score>0.5) — the ones an expert can adjudicate fast

It also writes a montage (label_completeness.png): GT green, matched blue-dashed, on-ring FP orange
(candidate real peak), off-ring FP red (candidate genuine error) — for expert eyeball confirmation.

Run on a GPU (builds two Swin-L models). See tmp_diag/run_label_completeness.sbatch.
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
CKPT_A = f"{CUR}/detector_runs/dino_ssl1/checkpoint.pth"                                   # ssl1
CKPT_B = "/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth"  # baseline
DSET = "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"
CONFIG = os.path.join(_REPO, "config/DINO/DINO_4scale_swin.py")
ST = 0.3            # operating score (same as the faint/high-q recall probe)
HI = 0.5           # "high confidence" cut for the fast-to-adjudicate subset
ONRING_PX = 8      # q-distance (px) to nearest GT peak below which a detection sits on a labeled ring
HIQ_PCT = 0.5      # I(q) percentile above which the ring actually carries intensity here


def build_model_from_ckpt(config_file, ckpt_path, device):
    parser = get_args_parser()
    a = parser.parse_args(['-c', config_file, '--output_dir', '/tmp/lc_eval', '--eval'])
    cfg = SLConfig.fromfile(a.config_file)
    for k, v in cfg._cfg_dict.to_dict().items():
        if not hasattr(a, k):
            setattr(a, k, v)
    for k, dv in [('export', False), ('use_ema', False), ('debug', False), ('num_channels', 1)]:
        if not hasattr(a, k):
            setattr(a, k, dv)
    model, _, _ = build_model_main(a)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    out = model.load_state_dict(sd, strict=False)
    print(f"  loaded {os.path.basename(os.path.dirname(ckpt_path))}: "
          f"missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
    model.to(device).eval()
    return model, a


def topk_dets(config, gc, model, img):
    out = model(img)
    raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
    onnx_to_xyxy(config, gc, raw)
    return gc.boxes.clone(), gc.scores.clone(), gc.pred_labels.clone()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}")
    print("building model A (ssl1)");     modelA, a = build_model_from_ckpt(CONFIG, CKPT_A, device)
    print("building model B (baseline)"); modelB, _ = build_model_from_ckpt(CONFIG, CKPT_B, device)

    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = DSET
    ds = PyGIDDataset(config, path=DSET, preprocess_func=standard_preprocessing,
                      buffer_size=5, load_labels=True)
    matcher = get_matcher('q', min_iou=0.1)

    n_frames = 0
    tot_gt = 0
    matched_gt = 0
    tp_det = 0
    fp_onring = 0          # unmatched det, on a labeled ring  -> likely real unlabeled peak
    fp_offring = 0         # unmatched det, off any labeled ring -> candidate genuine error
    fp_onring_hi = 0
    fp_offring_hi = 0
    fp_qd_all = []         # q-distance of every FP to nearest GT peak (for the summary histogram)
    montage = []

    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]                # (512,1024) HE image, masked=0
            valid = img_np > 1e-6
            denom = valid.sum(0); denom[denom == 0] = 1
            Iq = (img_np * valid).sum(0) / denom                   # angle-integrated I(q), (1024,)
            Iq_pct = np.argsort(np.argsort(Iq)) / len(Iq)          # percentile rank per q-column

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device).repeat(1, a.num_channels, 1, 1)
            bA, sA, lA = topk_dets(config, gc, modelA, img)
            bB, sB, lB = topk_dets(config, gc, modelB, img)
            # deployed ensemble: pool both models' top-k, one production class-aware NMS + score>0.1
            gc.boxes, gc.scores, gc.pred_labels = torch.cat([bA, bB]), torch.cat([sA, sB]), torch.cat([lA, lB])
            filter_boxes(config, gc)
            pred = gc.boxes; sc = gc.scores
            keep = sc > ST; pred = pred[keep]; sc = sc[keep]

            gt = torch.tensor(np.array(gc.polar_labels.boxes), dtype=torch.float32) if len(gc.polar_labels.boxes) else torch.zeros((0, 4))
            gt_qs = ((gt[:, 0] + gt[:, 2]) / 2).numpy() if len(gt) else np.array([])
            n_frames += 1; tot_gt += len(gt)

            row = np.array([], int); col = np.array([], int)
            if len(gt) and len(pred):
                try: _, row, col = matcher(gt, pred)
                except IndexError: pass
            matched_gt += len(set(row.tolist()))
            cset = set(col.tolist())

            tp_boxes, on_boxes, off_boxes = [], [], []
            for j in range(len(pred)):
                q = float((pred[j, 0] + pred[j, 2]) / 2); qi = int(np.clip(q, 0, 1023))
                if j in cset:
                    tp_det += 1; tp_boxes.append(pred[j].numpy()); continue
                qd = float(np.min(np.abs(gt_qs - q))) if len(gt_qs) else np.inf
                fp_qd_all.append(qd)
                onring = (qd < ONRING_PX) and (Iq_pct[qi] > HIQ_PCT)
                if onring:
                    fp_onring += 1; on_boxes.append(pred[j].numpy())
                    if float(sc[j]) > HI: fp_onring_hi += 1
                else:
                    fp_offring += 1; off_boxes.append(pred[j].numpy())
                    if float(sc[j]) > HI: fp_offring_hi += 1
            if len(montage) < 8:
                montage.append((img_np.copy(), gt.numpy(), np.array(tp_boxes), np.array(on_boxes), np.array(off_boxes)))

    if hasattr(ds, 'close'): ds.close()

    fp_all = fp_onring + fp_offring
    prec_std = tp_det / (tp_det + fp_all) if (tp_det + fp_all) else 0.0
    prec_adj = tp_det / (tp_det + fp_offring) if (tp_det + fp_offring) else 0.0
    recall = matched_gt / tot_gt if tot_gt else 0.0
    fp_hi_all = fp_onring_hi + fp_offring_hi

    print("\n" + "=" * 68)
    print(f"DEPLOYED ENSEMBLE (ssl1 + baseline) on organic  |  {n_frames} frames, {tot_gt} GT peaks")
    print(f"score>{ST}: TP={tp_det}  FP={fp_all}  (on-ring={fp_onring}, off-ring={fp_offring})")
    print("=" * 68)
    print(f"recall (matched GT / GT)          = {recall:.3f}")
    print(f"precision  (standard)             = {prec_std:.3f}")
    print(f"precision  (label-adjusted*)      = {prec_adj:.3f}   *on-ring FPs treated as ignore")
    print(f"on-ring share of FPs              = {fp_onring / fp_all:.2f}" if fp_all else "no FPs")
    fpq = np.array(fp_qd_all)
    if len(fpq):
        print(f"FP q-dist to nearest GT peak      : median={np.median(fpq):.1f}px  "
              f"frac<{ONRING_PX}px={np.mean(fpq < ONRING_PX):.2f}  frac>20px(off-ring)={np.mean(fpq > 20):.2f}")
    print(f"\nHIGH-confidence FPs (score>{HI})   = {fp_hi_all}  (on-ring={fp_onring_hi}, off-ring={fp_offring_hi})")
    if fp_hi_all:
        print(f"  -> {fp_onring_hi / fp_hi_all:.2f} of confident FPs sit on a labeled ring (fast to expert-check)")
    print("\nInterpretation: if on-ring share is high and adjusted >> standard precision, the eval is")
    print("label-LIMITED (precision pessimistic), not model-limited. Confirm on-ring FPs in the montage.")

    # ---- montage for expert review ----
    ncol = 2; nrow = int(np.ceil(len(montage) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4 * nrow))
    for ax, (a_img, gt, tp, on, off) in zip(np.atleast_1d(axes).ravel(), montage):
        ax.imshow(a_img, origin='lower', aspect='auto', cmap='gray')
        for b in gt:  ax.add_patch(mp.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='lime', lw=0.7))
        for b in tp:  ax.add_patch(mp.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='deepskyblue', lw=0.7, ls='--'))
        for b in on:  ax.add_patch(mp.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='orange', lw=1.0))
        for b in off: ax.add_patch(mp.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='red', lw=1.1))
        ax.set_title(f'GT={len(gt)}  TP={len(tp)}  on-ring FP={len(on)} (orange)  off-ring FP={len(off)} (red)', fontsize=9)
        ax.set_xlabel('q'); ax.set_ylabel('angle')
    for ax in np.atleast_1d(axes).ravel()[len(montage):]: ax.axis('off')
    fig.suptitle('Label-completeness: are the orange (on-ring) FPs real unlabeled peaks?  '
                 'GT=green  matched=blue-dash  on-ring FP=orange  off-ring FP=red', fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_completeness.png')
    fig.savefig(out, dpi=95, bbox_inches='tight'); print('\nsaved', out)


if __name__ == '__main__':
    main()
