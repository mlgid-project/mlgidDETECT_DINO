"""
Phase U PRIMARY GATE, done correctly: same-q sibling recall at a MATCHED OPERATING POINT.

The raw sweep (job 2764269) is not a valid read: at the deployed setting the cluster model emits
99.9 det/frame vs ssl1's 65.2 on organic, so its recall is higher everywhere and its precision
21 pts lower. That is the phase-P calibration trap -- recall at a fixed SCORE threshold is
meaningless across models whose score distributions differ. Here we sweep the score threshold per
model, cache predictions once, and compare where the DETECTION COUNT (and separately the PRECISION)
matches.

Stratification is the pre-registered one: recall by chi-gap to the nearest same-q labeled peak
(<5, 5-10, 10-20, 20-33, >=33 px). The <5 and 5-10 buckets carry the verdict; 41 is the
uncontaminated set.
"""
import os, sys, json
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher
from diagnostics.prominence_probe import build_model_from_ckpt, CKPT_A, CONFIG, DSETS, OUT

MODELS = [('ssl1', CKPT_A),
          ('clusters', '/mnt/lustre/work/schreiber/szb389/tmp_diag/clusters_probe_ckpt.pth')]
BASE = 0.05
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]     # 0.05 .. 0.90
QTOL = 8.0
BINS = [(0, 5), (5, 10), (10, 20), (20, 33), (33, 1e9)]


def sibling_gap(gt):
    n = len(gt)
    if n == 0:
        return np.zeros(0)
    q = ((gt[:, 0] + gt[:, 2]) / 2).numpy(); c = ((gt[:, 1] + gt[:, 3]) / 2).numpy()
    out = np.full(n, np.inf)
    for i in range(n):
        m = np.abs(q - q[i]) < QTOL; m[i] = False
        if m.any():
            out[i] = np.min(np.abs(c[m] - c[i]))
    return out


def collect(name, ckpt, path, device):
    model, a = build_model_from_ckpt(CONFIG, ckpt, device)
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    cfg.POSTPROCESSING_SCORE = BASE
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True          # deployed settings, unchanged (T2)
    cfg.INPUT_DATASET = path
    ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                       load_labels=True) if detect_dataset_type(path) == 'pygid'
          else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5))
    frames = []
    with torch.no_grad():
        for gc in ds.iter_images():
            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device)
            img = img.repeat(1, a.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
            gc = filter_boxes(cfg, onnx_to_xyxy(cfg, gc, raw))
            L = gc.polar_labels
            gt = (torch.tensor(np.array(L.boxes), dtype=torch.float32)
                  if len(L.boxes) else torch.zeros((0, 4)))
            frames.append(dict(gt=gt, gap=sibling_gap(gt),
                               pred=gc.boxes.clone(), sc=gc.scores.clone()))
    if hasattr(ds, 'close'):
        ds.close()
    del model; torch.cuda.empty_cache()
    return frames


def evaluate(frames, thr):
    matcher = get_matcher('q', min_iou=0.1)
    hits = {b: [0, 0] for b in BINS}
    ndet = tp = fp = ngt = 0
    for f in frames:
        keep = f['sc'] > thr
        pred = f['pred'][keep]
        ndet += len(pred); ngt += len(f['gt'])
        row = col = np.array([], int)
        if len(f['gt']) and len(pred):
            try:
                _, row, col = matcher(f['gt'], pred)
            except IndexError:
                pass
        mset = set(row.tolist())
        tp += len(mset); fp += len(pred) - len(set(col.tolist()))
        for i in range(len(f['gt'])):
            g = f['gap'][i]
            for lo, hi in BINS:
                if lo <= g < hi:
                    hits[(lo, hi)][1] += 1
                    if i in mset:
                        hits[(lo, hi)][0] += 1
                    break
    r = dict(thr=thr, det=ndet, dpf=ndet / max(len(frames), 1), recall=tp / max(ngt, 1),
             precision=tp / max(tp + fp, 1))
    for b in BINS:
        h, n = hits[b]
        r[f'g{b[0]}'] = (h / n) if n else float('nan')
        r[f'n{b[0]}'] = n
    return r


def fmt(r):
    return (f"thr={r['thr']:.2f} det/fr={r['dpf']:6.1f} R={r['recall']:.3f} P={r['precision']:.3f} | "
            + " ".join(f"{r[f'g{b[0]}']:.3f}" for b in BINS))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary = {}
    for tag, path in DSETS:
        print(f"\n{'='*104}\n{tag.upper()}\n{'='*104}", flush=True)
        sweeps = {}
        for name, ckpt in MODELS:
            print(f"  collecting {name} ...", flush=True)
            fr = collect(name, ckpt, path, device)
            sweeps[name] = [evaluate(fr, t) for t in THRESHOLDS]
            del fr
        hdr = "  " + " " * 42 + "| " + " ".join(f"{('<%d'%b[1]) if b[1]<1e8 else '>=33':>5s}" for b in BINS)
        print(f"\n  gap strata counts: " +
              "  ".join(f"{('<%d'%b[1]) if b[1]<1e8 else '>=33'}: n={sweeps['ssl1'][0][f'n{b[0]}']}"
                        for b in BINS))
        for name in ('ssl1', 'clusters'):
            print(f"\n  --- {name} sweep ---\n{hdr}")
            for r in sweeps[name]:
                print("   ", fmt(r))

        base = next(r for r in sweeps['ssl1'] if abs(r['thr'] - 0.30) < 1e-9)
        mc = min(sweeps['clusters'], key=lambda r: abs(r['dpf'] - base['dpf']))
        mp = min(sweeps['clusters'], key=lambda r: abs(r['precision'] - base['precision']))
        print(f"\n  {'*'*96}")
        print(f"  MATCHED DETECTION COUNT: ssl1@0.30 ({base['dpf']:.1f}/fr) -> "
              f"clusters@{mc['thr']:.2f} ({mc['dpf']:.1f}/fr)")
        print(f"    ssl1     {fmt(base)}\n    clusters {fmt(mc)}")
        print(f"    DELTA by chi-gap: " +
              "  ".join(f"{('<%d'%b[1]) if b[1]<1e8 else '>=33'} {mc[f'g{b[0]}']-base[f'g{b[0]}']:+.3f}"
                        for b in BINS))
        print(f"    overall recall {mc['recall']-base['recall']:+.3f}   "
              f"precision {mc['precision']-base['precision']:+.3f}")
        print(f"\n  MATCHED PRECISION: ssl1@0.30 (P={base['precision']:.3f}) -> "
              f"clusters@{mp['thr']:.2f} (P={mp['precision']:.3f})")
        print(f"    ssl1     {fmt(base)}\n    clusters {fmt(mp)}")
        print(f"    DELTA by chi-gap: " +
              "  ".join(f"{('<%d'%b[1]) if b[1]<1e8 else '>=33'} {mp[f'g{b[0]}']-base[f'g{b[0]}']:+.3f}"
                        for b in BINS))
        print(f"    overall recall {mp['recall']-base['recall']:+.3f}")
        summary[tag] = dict(ssl1_at_030=base, clusters_matched_count=mc, clusters_matched_prec=mp)
    json.dump(summary, open(os.path.join(OUT, 'clusters_gate.json'), 'w'), indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
