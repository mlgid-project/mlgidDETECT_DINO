"""Phase AA — the ANGULAR distance of false positives, never measured until now.

Flagged as an open hole since phase Q and left open ever since: the on-ring false positives were
characterised only by their q-distance to the nearest labeled peak (median 1.8 px), never by their
chi-distance. That is the number that decides whether the precision deficit and the close-pair recall
deficit are the SAME weakness.

Context. Organic, single model ssl1, score>0.3: recall 0.605, precision **0.764**. Of the false
positives, ~0.60 sit on a labeled ring (0.74 of the high-confidence ones). Phase Q once read that as
"probably unlabeled real peaks" and concluded the eval was label-limited; **the user corrected that --
the organic labels are COMPLETE -- and phase Q's verdict is retracted in this file.** So these are
genuine errors, and the question is what kind.

  FPs sit CLOSE in chi to a real peak (a few px)  -> the same chi weakness, seen from the other side:
     the model puts a second box beside a peak it cannot place precisely. Fixing chi would then buy
     recall AND precision, which roughly doubles the prize on the close-pair line.
  FPs sit FAR in chi                              -> an independent failure, and the precision side is
     a separate axis that has never had any of the phase X-Z treatment.

Measured per false positive, at the deployed operating point:
  - q-distance to the nearest GT peak            (reproduces the 1.8 px already on record)
  - chi-distance to the nearest GT peak at the SAME q (within 8 px) -- the new number
  - chi-distance to the nearest GT peak anywhere  (a control: if the same-q number is small only
    because peaks are dense everywhere, this one will be small too)

Reference distribution printed alongside: the GT-to-GT chi-gap (nearest same-q neighbour), so "close"
is judged against how far apart real peaks actually are rather than against intuition.

Stratified on-ring / off-ring and by confidence, since the on-ring subset is the one that carries the
precision deficit.

SINGLE MODEL ssl1 -- not the deployed ensemble (standing constraint). GPU, ~6 min.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.label_completeness import (build_model_from_ckpt, topk_dets, CKPT_A, CONFIG,
                                            DSET, ST, HI, ONRING_PX, HIQ_PCT)

QTOL = 8.0          # same-q tolerance, as clusters_gate / verify_clusters / the gap statistics


def nearest_chi(q, c, gt_q, gt_c, qtol=None):
    """chi-distance to the nearest GT peak, optionally restricted to peaks at the same q."""
    if not len(gt_q):
        return np.inf
    m = np.abs(gt_q - q) < qtol if qtol is not None else np.ones(len(gt_q), bool)
    if not m.any():
        return np.inf
    return float(np.min(np.abs(gt_c[m] - c)))


def pct(x, name):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return f"  {name:<34s} (none)"
    return (f"  {name:<34s} n={len(x):5d}  p10={np.percentile(x, 10):6.1f}  "
            f"med={np.median(x):6.1f}  p90={np.percentile(x, 90):6.1f}  "
            f"frac<10px={np.mean(x < 10):.3f}  frac<20px={np.mean(x < 20):.3f}")


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1 (not the ensemble)", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)

    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = 0.1
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = DSET
    ds = PyGIDDataset(config, path=DSET, preprocess_func=standard_preprocessing,
                      buffer_size=5, load_labels=True)
    matcher = get_matcher('q', min_iou=0.1)

    R = dict(fp_q=[], fp_chi_sameq=[], fp_chi_any=[],
             on_chi=[], off_chi=[], hi_chi=[], gt_gap=[])
    n_frames = tp = fp_on = fp_off = 0
    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = gc.converted_polar_image[0, 0]
            valid = img_np > 1e-6
            den = valid.sum(0); den[den == 0] = 1
            Iq = (img_np * valid).sum(0) / den
            Iq_pct = np.argsort(np.argsort(Iq)) / len(Iq)

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, a.num_channels, 1, 1)
            b, s, l = topk_dets(config, gc, model, img)
            gc.boxes, gc.scores, gc.pred_labels = b, s, l
            filter_boxes(config, gc)
            pred, sc = gc.boxes, gc.scores
            keep = sc > ST; pred, sc = pred[keep], sc[keep]

            gt = torch.tensor(np.array(gc.polar_labels.boxes), dtype=torch.float32) \
                if len(gc.polar_labels.boxes) else torch.zeros((0, 4))
            n_frames += 1
            if not len(gt):
                continue
            gq = ((gt[:, 0] + gt[:, 2]) / 2).numpy()
            gcc = ((gt[:, 1] + gt[:, 3]) / 2).numpy()

            # reference: how far apart real peaks actually are, same-q, in this frame
            for i in range(len(gq)):
                m = (np.abs(gq - gq[i]) < QTOL); m[i] = False
                if m.any():
                    R['gt_gap'].append(float(np.min(np.abs(gcc[m] - gcc[i]))))

            col = np.array([], int)
            if len(pred):
                try:
                    _, _row, col = matcher(gt, pred)
                except IndexError:
                    pass
            cset = set(col.tolist())
            for j in range(len(pred)):
                if j in cset:
                    tp += 1
                    continue
                q = float((pred[j, 0] + pred[j, 2]) / 2)
                c = float((pred[j, 1] + pred[j, 3]) / 2)
                qd = float(np.min(np.abs(gq - q)))
                cd_same = nearest_chi(q, c, gq, gcc, QTOL)
                cd_any = nearest_chi(q, c, gq, gcc, None)
                R['fp_q'].append(qd); R['fp_chi_sameq'].append(cd_same); R['fp_chi_any'].append(cd_any)
                onring = (qd < ONRING_PX) and (Iq_pct[int(np.clip(q, 0, 1023))] > HIQ_PCT)
                (R['on_chi'] if onring else R['off_chi']).append(cd_same)
                if onring:
                    fp_on += 1
                else:
                    fp_off += 1
                if float(sc[j]) > HI:
                    R['hi_chi'].append(cd_same)
    if hasattr(ds, 'close'):
        ds.close()

    fp = fp_on + fp_off
    print("\n" + "=" * 92)
    print(f"ssl1 on organic, score>{ST}  |  {n_frames} frames   TP={tp}  FP={fp} "
          f"(on-ring {fp_on}, off-ring {fp_off})   precision={tp / max(tp + fp, 1):.3f}")
    print("=" * 92)
    print("\n  q-distance of FPs to the nearest GT peak (the number already on record):")
    print(pct(R['fp_q'], 'all FPs, q-distance'))
    print("\n  CHI-distance of FPs to the nearest GT peak at the SAME q (<8 px) -- THE NEW NUMBER:")
    print(pct(R['fp_chi_sameq'], 'all FPs'))
    print(pct(R['on_chi'], 'on-ring FPs'))
    print(pct(R['off_chi'], 'off-ring FPs'))
    print(pct(R['hi_chi'], f'high-confidence FPs (>{HI})'))
    print("\n  controls:")
    print(pct(R['fp_chi_any'], 'FP chi-dist, ANY q (control)'))
    print(pct(R['gt_gap'], 'GT-to-GT chi gap, same q (ref)'))
    print("\n  Reading: if the on-ring FP chi-distance is small against the GT-to-GT reference, the")
    print("  precision deficit and the close-pair recall deficit are the same chi weakness.")

    json.dump({k: [float(x) for x in v if np.isfinite(x)] for k, v in R.items()},
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/fp_chi_probe.json', 'w'))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bins = np.arange(0, 105, 5)
    for key, lbl in (('on_chi', 'on-ring FPs'), ('off_chi', 'off-ring FPs'),
                     ('gt_gap', 'GT-to-GT gap (reference)')):
        v = np.asarray([x for x in R[key] if np.isfinite(x)])
        if len(v):
            ax.hist(np.clip(v, 0, 100), bins=bins, density=True, histtype='step', lw=2, label=lbl)
    ax.set_xlabel('χ-distance to nearest same-q labeled peak (px)')
    ax.set_ylabel('density')
    ax.set_title('Phase AA: are false positives angularly adjacent to real peaks?')
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_chi.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}\nPROBE DONE")


if __name__ == '__main__':
    main()
