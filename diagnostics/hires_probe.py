"""
Decisive gate for the higher-resolution lever (docs/HIRES_INVESTIGATION.md):
faint / high-q RECALL probe, hires (512x2048) vs ssl1 (512x1024), at a MATCHED operating point.

Why matched: the phase-P calibration lesson -- recall at a fixed score threshold is meaningless
across models whose score distributions differ. We therefore sweep the score threshold per model,
cache (gt, preds, scores) once per image, and re-match at every threshold, then compare the two
models at the threshold pair where their DETECTION COUNT (and separately their PRECISION) matches.

q-strata are computed in NORMALIZED q (box q-center / WIDTH) so the thirds are physically the same
band at 1024 and 2048.

argv: none. Prints a full report; the caller records it in MODIFICATIONS.md.
"""
import sys, json
import numpy as np, torch

REPO = '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO'
sys.path.insert(0, REPO)

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CUR = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation'
DSET = '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
CONFIG = f'{REPO}/config/DINO/DINO_4scale_swin.py'   # ssl/hires differ only by backbone_dir/polar_shape
MODELS = [
    ('ssl1_1024', f'{CUR}/detector_runs/dino_ssl1/checkpoint.pth', 1024),
    # verified snapshot of dino_hires1/checkpoint.pth (the live file is still being rewritten)
    ('hires_2048', '/mnt/lustre/work/schreiber/szb389/tmp_diag/hires_probe_ckpt.pth', 2048),
]
BASE_SCORE = 0.05                       # postprocessing floor, so we can sweep upward
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 15)]   # 0.05 .. 0.70


def build(ckpt):
    args = get_args_parser().parse_args([])
    for k, v in SLConfig.fromfile(CONFIG)._cfg_dict.to_dict().items():
        setattr(args, k, v)
    args.device = 'cuda'; args.export = False
    model, _, _ = build_model_main(args)
    model = model.cuda().eval()
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd['model'] if 'model' in sd else sd
    rep = model.load_state_dict(sd, strict=False)
    return model, args, rep


def collect(name, ckpt, width):
    """One forward pass over the organic eval; cache per-image GT strata + preds + scores."""
    model, args, rep = build(ckpt)
    print(f"### {name}  ckpt={ckpt}")
    print(f"    load: missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}")
    cfg = Config()
    cfg.PREPROCESSING_POLAR_SHAPE = [512, width]
    cfg.POSTPROCESSING_SCORE = BASE_SCORE
    cfg.POSTPROCESSING_CLASSAWARE_NMS = True
    cfg.INPUT_DATASET = DSET
    ds = PyGIDDataset(cfg, path=DSET, preprocess_func=standard_preprocessing,
                      buffer_size=5, load_labels=True)
    frames = []
    with torch.no_grad():
        for ic in ds.iter_images():
            img = torch.tensor(ic.converted_polar_image[:, 0, :, :]).unsqueeze(0).cuda()
            img = img.repeat(1, args.num_channels, 1, 1)
            out = model(img)
            raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
            ic = filter_boxes(cfg, onnx_to_xyxy(cfg, ic, raw))
            L = ic.polar_labels
            gt = (torch.tensor(np.array(L.boxes), dtype=torch.float32)
                  if len(L.boxes) else torch.zeros((0, 4)))
            frames.append(dict(
                gt=gt,
                vis=(np.array(L.visibility) if len(L.visibility) else np.zeros(len(gt))),
                isr=(np.array([bool(x) for x in L.is_ring]) if len(L.is_ring)
                     else np.zeros(len(gt), bool)),
                gtq=(((gt[:, 0] + gt[:, 2]) / 2).numpy() / width if len(gt) else np.array([])),
                pred=ic.boxes.clone(), sc=ic.scores.clone(),
            ))
    if hasattr(ds, 'close'):
        ds.close()
    del model
    torch.cuda.empty_cache()
    return frames


def evaluate(frames, thr):
    """Re-match at threshold `thr`; return strata recalls + counts."""
    matcher = get_matcher('q', min_iou=0.1)
    vis, isr, gq, matched = [], [], [], []
    n_det = 0; n_fp = 0
    for f in frames:
        gt = f['gt']
        keep = f['sc'] > thr
        pred = f['pred'][keep]
        n_det += len(pred)
        row = np.array([], int); col = np.array([], int)
        if len(gt) and len(pred):
            try:
                _, row, col = matcher(gt, pred)
            except IndexError:
                pass
        mset = set(row.tolist())
        n_fp += len(pred) - len(set(col.tolist()))
        for i in range(len(gt)):
            vis.append(int(f['vis'][i])); isr.append(bool(f['isr'][i]))
            gq.append(float(f['gtq'][i])); matched.append(i in mset)
    vis = np.array(vis); isr = np.array(isr); gq = np.array(gq); matched = np.array(matched)
    N = len(vis); TP = int(matched.sum())
    r = dict(thr=thr, gt=N, tp=TP, det=n_det, fp=n_fp,
             recall=TP / N if N else 0.0,
             precision=TP / (TP + n_fp) if (TP + n_fp) else 0.0)
    for v in (3, 2, 1):
        m = vis == v
        r[f'recall_vis{v}'] = float(matched[m].mean()) if m.sum() else float('nan')
    r['recall_ring'] = float(matched[isr].mean()) if isr.sum() else float('nan')
    r['recall_seg'] = float(matched[~isr].mean()) if (~isr).sum() else float('nan')
    for lab, lo, hi in (('q_lo', 0.0, 1/3), ('q_mid', 1/3, 2/3), ('q_hi', 2/3, 1.0001)):
        m = (gq >= lo) & (gq < hi)
        r[f'recall_{lab}'] = float(matched[m].mean()) if m.sum() else float('nan')
        r[f'n_{lab}'] = int(m.sum())
    return r


def fmt(r):
    return (f"thr={r['thr']:.2f} det={r['det']:4d} R={r['recall']:.3f} P={r['precision']:.3f} | "
            f"vis 3/2/1 = {r['recall_vis3']:.3f}/{r['recall_vis2']:.3f}/{r['recall_vis1']:.3f} | "
            f"q lo/mid/hi = {r['recall_q_lo']:.3f}/{r['recall_q_mid']:.3f}/{r['recall_q_hi']:.3f} | "
            f"ring/seg = {r['recall_ring']:.3f}/{r['recall_seg']:.3f}")


def main():
    sweeps = {}
    for name, ckpt, width in MODELS:
        frames = collect(name, ckpt, width)
        sweeps[name] = [evaluate(frames, t) for t in THRESHOLDS]
        print(f"--- {name}: sweep ---")
        for r in sweeps[name]:
            print("   ", fmt(r))
        print()
        del frames

    a, b = MODELS[0][0], MODELS[1][0]
    print("=" * 100)
    print(f"MATCHED OPERATING POINTS   ({a} = reference, {b} = hires lever)")
    print("=" * 100)

    # (1) fixed 0.30 -- the deployed threshold, for continuity with prior levers
    ra = next(r for r in sweeps[a] if abs(r['thr'] - 0.30) < 1e-9)
    rb = next(r for r in sweeps[b] if abs(r['thr'] - 0.30) < 1e-9)
    print(f"\n[fixed thr=0.30]\n  {a:11s} {fmt(ra)}\n  {b:11s} {fmt(rb)}")

    # (2) matched DETECTION COUNT: for the reference at 0.30, find hires thr with nearest det count
    rb_m = min(sweeps[b], key=lambda r: abs(r['det'] - ra['det']))
    print(f"\n[matched detection count: {a}@0.30 det={ra['det']}  ->  {b}@{rb_m['thr']:.2f} det={rb_m['det']}]")
    print(f"  {a:11s} {fmt(ra)}\n  {b:11s} {fmt(rb_m)}")

    # (3) matched PRECISION
    rb_p = min(sweeps[b], key=lambda r: abs(r['precision'] - ra['precision']))
    print(f"\n[matched precision: {a}@0.30 P={ra['precision']:.3f}  ->  {b}@{rb_p['thr']:.2f} P={rb_p['precision']:.3f}]")
    print(f"  {a:11s} {fmt(ra)}\n  {b:11s} {fmt(rb_p)}")

    print("\n[DELTAS at matched detection count]  (hires - ssl1; positive = hires better)")
    for k in ('recall', 'recall_vis1', 'recall_vis2', 'recall_vis3',
              'recall_q_lo', 'recall_q_mid', 'recall_q_hi', 'recall_ring', 'recall_seg', 'precision'):
        print(f"    {k:14s} {rb_m[k] - ra[k]:+.3f}   ({ra[k]:.3f} -> {rb_m[k]:.3f})")

    json.dump(sweeps, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/hires_probe.json', 'w'), indent=2)
    print("\nwrote hires_probe.json")


if __name__ == '__main__':
    main()
