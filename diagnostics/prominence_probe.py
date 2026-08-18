"""
Prominence probe — is the residual recall gap a MODEL limit or the DATA's information floor?

Motivation. Eight consecutive levers have been declined (MODIFICATIONS.md phases I-R) and the
one that mattered most, higher input resolution, failed specifically on *recall* of faint/high-q
peaks. Before spending another run on a ninth lever, measure whether the peaks the deployed model
misses are physically distinguishable in the input at all.

The measure is TOPOGRAPHIC PROMINENCE, i.e. 0-dimensional superlevel-set persistence: sweep a
threshold downward, every local maximum births a component at its own height, and when two
components merge the ELDER (higher birth) survives while the younger dies at the merge level.
prominence = birth - death. For a peak this is exactly "how far must I descend from this summit
before reaching ground that leads somewhere higher" -- a measure of how far a peak stands out
from its *local* surroundings, independent of the absolute background level. It needs no
smoothing and no threshold: a noise spike simply births a component with tiny prominence.
(Core routine verified against hand-computed cases in tmp_diag/prom_core_selftest.py.)

Read-out. If the missed peaks sit at low prominence and the high-prominence peaks are all found,
the remaining gap is the information floor of the data and further modelling levers cannot reach
it. If misses are spread across prominence, there is genuine headroom and the probe says where.

Model = the DEPLOYED ensemble (ssl1 + from-scratch baseline, detection-level NMS fusion), exactly
as backbone_curation/ensemble_eval.py and diagnostics/label_completeness.py. Operating point,
matcher and preprocessing are identical to those probes so the numbers are comparable.

Runs on both eval gates (organic + 41). GPU, ~15 min. See tmp_diag/run_prominence_probe.sbatch.
"""
import os, sys, json
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CUR = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
CKPT_A = f"{CUR}/detector_runs/dino_ssl1/checkpoint.pth"                                                # ssl1
CKPT_B = "/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth"  # baseline
CONFIG = os.path.join(_REPO, "config/DINO/DINO_4scale_swin.py")
DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41",      "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
ST = 0.30           # operating score — same as the faint/high-q recall probe and label_completeness
BASE_SCORE = 0.10   # postprocessing floor before the ST cut (identical to label_completeness)
OUT = "/mnt/lustre/work/schreiber/szb389/tmp_diag"


# ----------------------------------------------------------------------------- prominence core
def superlevel_prominence(img, valid=None, max_levels=512):
    """0-dim superlevel-set persistence of every local maximum. Returns y, x, birth, prom."""
    img = np.asarray(img, dtype=np.float64)
    if valid is None:
        valid = np.ones(img.shape, bool)
    vals = img[valid]
    empty = dict(y=np.zeros(0, int), x=np.zeros(0, int), birth=np.zeros(0), prom=np.zeros(0))
    if vals.size == 0 or vals.max() <= vals.min():
        return empty
    vmax, vmin = float(vals.max()), float(vals.min())

    uv = np.unique(vals)
    if uv.size <= max_levels:
        levels = uv[::-1]                                    # exact when the image is quantised
    else:
        levels = np.quantile(vals, np.linspace(0, 1, max_levels))
        levels = np.unique(np.concatenate([[vmin], levels, [vmax]]))[::-1]

    struct = ndi.generate_binary_structure(2, 2)             # 8-connectivity
    birth, death, py, px = [], [], [], []
    old_lab = np.zeros(img.shape, np.int32); old_root = np.zeros(1, np.int64); n_old = 0

    for lev in levels:
        new_lab, n_new = ndi.label(valid & (img >= lev), structure=struct)
        if n_new == 0:
            continue
        new_root = np.full(n_new + 1, -1, np.int64)
        if n_old:
            # levels descend => every old component lies inside exactly one new component
            mapping = np.zeros(n_old + 1, np.int32)
            mapping[old_lab.ravel()] = new_lab.ravel()
            tgt = mapping[1:]
            order = np.argsort(tgt, kind='stable')
            uniq, first, counts = np.unique(tgt[order], return_index=True, return_counts=True)
            one = counts == 1
            if one.any():                                    # carried over, no merge
                new_root[uniq[one]] = old_root[order[first[one]] + 1]
            for u, i0, c in zip(uniq[~one], first[~one], counts[~one]):   # merges: elder rule
                roots = old_root[order[i0:i0 + c] + 1]
                elder = int(roots[np.argmax([birth[r] for r in roots])])
                for r in roots:
                    if int(r) != elder:
                        death[int(r)] = float(lev)
                new_root[u] = elder
        fresh = np.nonzero(new_root == -1)[0]; fresh = fresh[fresh > 0]
        if fresh.size:
            mx = np.atleast_1d(ndi.maximum(img, new_lab, index=fresh))
            pos = ndi.maximum_position(img, new_lab, index=fresh)
            if not isinstance(pos, list):
                pos = [pos]
            for k, f in enumerate(fresh):
                new_root[f] = len(birth)
                birth.append(float(mx[k])); death.append(vmin)
                py.append(int(pos[k][0])); px.append(int(pos[k][1]))
        old_lab, old_root, n_old = new_lab, new_root, n_new

    birth = np.array(birth); death = np.array(death)
    return dict(y=np.array(py, int), x=np.array(px, int), birth=birth, prom=birth - death)


def noise_sigma_per_q(img, valid, nbins=32):
    """Robust local noise sigma per q-band (MAD of the high-pass residual). Noise falls with q,
    so a single per-image sigma would understate SNR at high q."""
    resid = img - ndi.uniform_filter(img, size=5)
    W = img.shape[1]
    edges = np.linspace(0, W, nbins + 1).astype(int)
    sig = np.full(nbins, np.nan)
    for i in range(nbins):
        r = resid[:, edges[i]:edges[i + 1]][valid[:, edges[i]:edges[i + 1]]]
        if r.size > 50:
            sig[i] = 1.4826 * np.median(np.abs(r - np.median(r)))
    good = np.isfinite(sig) & (sig > 0)
    if good.any():                                            # fill empty bands with the median
        sig[~good] = np.median(sig[good])
    else:
        sig[:] = 1.0
    return sig, edges


def box_prominence(pk, box, valid):
    """Max prominence among local maxima inside `box` (xyxy, x=q/col, y=angle/row).

    Also returns the fraction of the box that is real data: a GT peak sitting inside a detector
    gap / masked region is invisible to ANY model and must be reported separately."""
    H, W = valid.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = np.clip(min(x1, x2), 0, W - 1), np.clip(max(x1, x2), 0, W - 1)
    y1, y2 = np.clip(min(y1, y2), 0, H - 1), np.clip(max(y1, y2), 0, H - 1)
    sub = valid[int(np.floor(y1)):int(np.ceil(y2)) + 1, int(np.floor(x1)):int(np.ceil(x2)) + 1]
    vf = float(sub.mean()) if sub.size else 0.0
    m = (pk['x'] >= x1) & (pk['x'] <= x2) & (pk['y'] >= y1) & (pk['y'] <= y2)
    if not m.any():
        return 0.0, 0.0, -1, vf
    j = np.argmax(pk['prom'][m])
    idx = np.nonzero(m)[0][j]
    return float(pk['prom'][idx]), float(pk['birth'][idx]), int(pk['x'][idx]), vf


# ------------------------------------------------------------------------------------ model
def build_model_from_ckpt(config_file, ckpt_path, device):
    parser = get_args_parser()
    a = parser.parse_args(['-c', config_file, '--output_dir', '/tmp/prom_probe', '--eval'])
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
          f"missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}", flush=True)
    model.to(device).eval()
    return model, a


def topk_dets(config, gc, model, img):
    out = model(img)
    raw = [out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
    onnx_to_xyxy(config, gc, raw)
    return gc.boxes.clone(), gc.scores.clone(), gc.pred_labels.clone()


def run_dataset(tag, path, modelA, modelB, a, device):
    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.POSTPROCESSING_SCORE = BASE_SCORE
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    config.INPUT_DATASET = path
    if detect_dataset_type(path) == 'pygid':
        ds = PyGIDDataset(config, path=path, preprocess_func=standard_preprocessing,
                          buffer_size=5, load_labels=True)
    else:
        ds = H5GIWAXSDataset(config, path=path, preprocess_func=standard_preprocessing,
                             buffer_size=5)
    matcher = get_matcher('q', min_iou=0.1)

    G = dict(prom=[], snr=[], birth=[], qn=[], conf=[], ring=[], det=[], frame=[],
             nomax=[], vfrac=[])
    F = dict(prom=[], snr=[], qn=[], score=[], vfrac=[])          # unmatched detections (candidate real peaks)
    nf = 0
    with torch.no_grad():
        for gc in ds.iter_images():
            img_np = np.asarray(gc.converted_polar_image[0, 0], dtype=np.float64)   # model input
            H, W = img_np.shape
            valid = img_np > 1e-6                                                   # 0 = no-data
            pk = superlevel_prominence(img_np, valid)
            sig, edges = noise_sigma_per_q(img_np, valid)

            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(device)
            img = img.repeat(1, a.num_channels, 1, 1)
            bA, sA, lA = topk_dets(config, gc, modelA, img)
            bB, sB, lB = topk_dets(config, gc, modelB, img)
            gc.boxes = torch.cat([bA, bB]); gc.scores = torch.cat([sA, sB])
            gc.pred_labels = torch.cat([lA, lB])
            filter_boxes(config, gc)                       # production class-aware NMS + score cut
            keep = gc.scores > ST
            pred, sc = gc.boxes[keep], gc.scores[keep]

            L = gc.polar_labels
            gt = (torch.tensor(np.array(L.boxes), dtype=torch.float32)
                  if len(L.boxes) else torch.zeros((0, 4)))
            ring = np.asarray(L.is_ring if len(L.is_ring) else
                              getattr(gc.reciprocal_labels, 'is_ring', []) or [False] * len(gt))
            conf = np.asarray(L.confidences if len(L.confidences) else [np.nan] * len(gt), float)

            row = np.array([], int); col = np.array([], int)
            if len(gt) and len(pred):
                try:
                    _, row, col = matcher(gt, pred)
                except IndexError:
                    pass
            mset, cset = set(row.tolist()), set(col.tolist())

            def snr_at(xpix, prom):
                b = int(np.clip(np.searchsorted(edges, xpix, 'right') - 1, 0, len(sig) - 1))
                return prom / sig[b] if sig[b] > 0 else np.nan

            for i in range(len(gt)):
                p, b, xp, vf = box_prominence(pk, gt[i].numpy(), valid)
                qc = float((gt[i, 0] + gt[i, 2]) / 2)
                G['prom'].append(p); G['birth'].append(b)
                G['snr'].append(snr_at(xp if xp >= 0 else qc, p))
                G['qn'].append(qc / W); G['conf'].append(float(conf[i]) if i < len(conf) else np.nan)
                G['ring'].append(bool(ring[i]) if i < len(ring) else False)
                G['det'].append(i in mset); G['frame'].append(nf); G['nomax'].append(xp < 0)
                G['vfrac'].append(vf)
            for j in range(len(pred)):
                if j in cset:
                    continue
                p, b, xp, vf = box_prominence(pk, pred[j].numpy(), valid)
                qc = float((pred[j, 0] + pred[j, 2]) / 2)
                F['prom'].append(p); F['snr'].append(snr_at(xp if xp >= 0 else qc, p))
                F['qn'].append(qc / W); F['score'].append(float(sc[j]))
                F['vfrac'].append(vf)
            nf += 1
            print(f"  [{tag}] frame {nf}: GT={len(gt)} det={len(pred)} maxima={len(pk['prom'])}",
                  flush=True)
    if hasattr(ds, 'close'):
        ds.close()
    G = {k: np.array(v) for k, v in G.items()}
    F = {k: np.array(v) for k, v in F.items()}
    print(f"[{tag}] {nf} frames, {len(G['prom'])} GT peaks, {len(F['prom'])} unmatched detections",
          flush=True)
    return G, F


# ----------------------------------------------------------------------------------- analysis
def auc_sep(det_vals, miss_vals):
    """P(random detected peak is more prominent than a random missed peak). 0.5 = prominence
    explains nothing about who gets missed; 1.0 = misses are entirely explained by faintness."""
    if len(det_vals) == 0 or len(miss_vals) == 0:
        return float('nan')
    allv = np.concatenate([det_vals, miss_vals])
    r = rankdata(allv)
    n1, n2 = len(det_vals), len(miss_vals)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def decile_table(prom, det, nbins=10, label=""):
    if len(prom) < nbins * 3:
        return []
    qs = np.quantile(prom, np.linspace(0, 1, nbins + 1))
    qs[-1] += 1e-12
    rows = []
    for i in range(nbins):
        m = (prom >= qs[i]) & (prom < qs[i + 1])
        if m.sum() == 0:
            continue
        rows.append(dict(lo=float(qs[i]), hi=float(qs[i + 1]), n=int(m.sum()),
                         rate=float(det[m].mean())))
    if label:
        print(f"\n  {label}")
        print(f"    {'prominence range':>22s} {'n':>5s} {'detected':>9s}")
        for r in rows:
            bar = '#' * int(round(r['rate'] * 30))
            print(f"    [{r['lo']:.4f}, {r['hi']:.4f}] {r['n']:5d} {r['rate']:8.3f}  {bar}")
    return rows


def report(tag, G, F):
    prom, det, qn = G['prom'], G['det'].astype(bool), G['qn']
    N = len(prom)
    print("\n" + "=" * 92)
    print(f"{tag.upper()}   deployed ensemble @score>{ST}   |  {N} labeled peaks   "
          f"recall={det.mean():.3f}")
    print("=" * 92)
    if N == 0:
        return {}
    dv, mv = prom[det], prom[~det]
    print(f"  prominence (model-input units, image range [0,1]):")
    print(f"    detected (n={len(dv):4d})  median={np.median(dv):.4f}  "
          f"q25={np.quantile(dv,.25):.4f}  q75={np.quantile(dv,.75):.4f}")
    print(f"    MISSED   (n={len(mv):4d})  median={np.median(mv) if len(mv) else float('nan'):.4f}  "
          f"q25={np.quantile(mv,.25) if len(mv) else float('nan'):.4f}  "
          f"q75={np.quantile(mv,.75) if len(mv) else float('nan'):.4f}")
    A = auc_sep(dv, mv)
    print(f"    separation AUC = {A:.3f}   "
          f"(0.5 = faintness explains nothing; 1.0 = misses are purely the faint ones)")
    sd, sm = G['snr'][det], G['snr'][~det]
    print(f"  SNR (prominence / local noise sigma):  detected median={np.nanmedian(sd):.1f}   "
          f"MISSED median={np.nanmedian(sm) if len(sm) else float('nan'):.1f}")
    nm = int(G['nomax'].sum())
    if nm:
        print(f"  note: {nm} GT boxes contained no local maximum at all (prominence set to 0)")
    dead = G['vfrac'] < 0.5
    if dead.any():
        print(f"  note: {int(dead.sum())} GT boxes are >50% detector-mask / no-data "
              f"(recall there = {det[dead].mean():.3f}) -- invisible to ANY model.")
        print(f"        excluding them, recall = {det[~dead].mean():.3f} over {int((~dead).sum())} peaks")

    decile_table(prom, det, label="detection rate by prominence decile (all peaks)")

    # headroom: misses that are as prominent as a typical DETECTED peak are genuine model failures
    P50 = float(np.median(dv))
    obvious = int(((~det) & (prom > P50)).sum())
    print(f"\n  HEADROOM ESTIMATE")
    print(f"    median prominence of detected peaks P50_det = {P50:.4f}")
    print(f"    missed peaks with prominence > P50_det      = {obvious} "
          f"({obvious / N:.3f} of all GT)")
    print(f"    -> recall is {det.mean():.3f}; at most +{obvious / N:.3f} is reachable by fixing")
    print(f"       misses that are as visible as a typical successful detection.")
    below = int(((~det) & (prom <= np.quantile(prom, 0.2))).sum())
    print(f"    missed peaks in the faintest 20% of all GT  = {below} "
          f"({below / max(1, (~det).sum()):.2f} of all misses)")

    strat = {}
    for lab, lo, hi in (('q_lo', 0.0, 1/3), ('q_mid', 1/3, 2/3), ('q_hi', 2/3, 1.0001)):
        m = (qn >= lo) & (qn < hi)
        if m.sum() < 10:
            continue
        dm, mm = prom[m & det], prom[m & ~det]
        strat[lab] = dict(n=int(m.sum()), recall=float(det[m].mean()),
                          med_prom=float(np.median(prom[m])),
                          med_det=float(np.median(dm)) if len(dm) else float('nan'),
                          med_miss=float(np.median(mm)) if len(mm) else float('nan'),
                          auc=auc_sep(dm, mm))
    print(f"\n  BY q-STRATUM   (the hires lever failed specifically at high q)")
    print(f"    {'stratum':8s} {'n':>5s} {'recall':>7s} {'med prom':>9s} {'med det':>9s} "
          f"{'med miss':>9s} {'AUC':>6s}")
    for k, v in strat.items():
        print(f"    {k:8s} {v['n']:5d} {v['recall']:7.3f} {v['med_prom']:9.4f} "
              f"{v['med_det']:9.4f} {v['med_miss']:9.4f} {v['auc']:6.3f}")

    ring = G['ring'].astype(bool)
    if ring.any() and (~ring).any():
        print(f"\n  BY CLASS   ring: n={int(ring.sum())} recall={det[ring].mean():.3f} "
              f"med_prom={np.median(prom[ring]):.4f}   |   "
              f"segment: n={int((~ring).sum())} recall={det[~ring].mean():.3f} "
              f"med_prom={np.median(prom[~ring]):.4f}")

    cf = G['conf']
    if np.isfinite(cf).any():
        print(f"\n  BY ANNOTATOR CONFIDENCE")
        for c in np.unique(cf[np.isfinite(cf)]):
            m = cf == c
            print(f"    conf={c:<4} n={int(m.sum()):4d} recall={det[m].mean():.3f} "
                  f"med_prom={np.median(prom[m]):.4f}")

    if len(F['prom']):
        fp = F['prom']
        pctile = float((prom[:, None] < fp[None, :]).mean()) if N else float('nan')
        print(f"\n  UNMATCHED DETECTIONS (n={len(fp)})   median prominence={np.median(fp):.4f}")
        print(f"    vs labeled peaks median={np.median(prom):.4f}; a typical unmatched detection "
              f"is more prominent than {pctile:.2f} of labeled peaks")
        print(f"    -> high values support the label-completeness finding (these look like real "
              f"peaks, not noise)")

    return dict(n=N, recall=float(det.mean()), auc=float(A), P50_det=P50,
                obvious_misses=obvious, obvious_frac=obvious / N,
                med_prom_det=float(np.median(dv)),
                med_prom_miss=float(np.median(mv)) if len(mv) else None,
                strata=strat, n_fp=len(F['prom']),
                med_prom_fp=float(np.median(F['prom'])) if len(F['prom']) else None)


def make_plot(res_all, path):
    n = len(res_all)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.2 * n), squeeze=False)
    for r, (tag, G, F) in enumerate(res_all):
        prom, det = G['prom'], G['det'].astype(bool)
        eps = max(1e-4, np.quantile(prom[prom > 0], 0.01) if (prom > 0).any() else 1e-4)
        bins = np.logspace(np.log10(eps), np.log10(max(prom.max(), eps * 10)), 30)

        ax = axes[r][0]
        ax.hist(np.clip(prom[det], eps, None), bins=bins, alpha=.6, label=f'detected ({det.sum()})',
                color='tab:blue')
        ax.hist(np.clip(prom[~det], eps, None), bins=bins, alpha=.6,
                label=f'MISSED ({(~det).sum()})', color='tab:red')
        ax.set_xscale('log'); ax.set_xlabel('prominence'); ax.set_ylabel('labeled peaks')
        ax.set_title(f'{tag}: prominence of found vs missed peaks'); ax.legend(fontsize=8)

        ax = axes[r][1]
        qs = np.quantile(prom, np.linspace(0, 1, 11)); qs[-1] += 1e-12
        cx, cy, cn = [], [], []
        for i in range(10):
            m = (prom >= qs[i]) & (prom < qs[i + 1])
            if m.sum():
                cx.append(np.median(prom[m])); cy.append(det[m].mean()); cn.append(m.sum())
        ax.plot(cx, cy, 'o-', color='tab:purple')
        for x, y, k in zip(cx, cy, cn):
            ax.annotate(str(k), (x, y), fontsize=7, xytext=(0, 5), textcoords='offset points',
                        ha='center')
        ax.set_xscale('log'); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel('prominence (decile median)'); ax.set_ylabel('fraction detected')
        ax.axhline(det.mean(), ls=':', c='gray', label=f'overall recall {det.mean():.3f}')
        ax.set_title(f'{tag}: detection rate vs prominence'); ax.legend(fontsize=8)

        ax = axes[r][2]
        for lab, lo, hi, c in (('low q', 0, 1/3, 'tab:green'), ('mid q', 1/3, 2/3, 'tab:orange'),
                               ('high q', 2/3, 1.0001, 'tab:red')):
            m = (G['qn'] >= lo) & (G['qn'] < hi)
            if m.sum() < 10:
                continue
            q2 = np.quantile(prom[m], np.linspace(0, 1, 6)); q2[-1] += 1e-12
            xx, yy = [], []
            for i in range(5):
                mm = m & (prom >= q2[i]) & (prom < q2[i + 1])
                if mm.sum():
                    xx.append(np.median(prom[mm])); yy.append(det[mm].mean())
            ax.plot(xx, yy, 'o-', color=c, label=f'{lab} (n={m.sum()}, R={det[m].mean():.2f})')
        ax.set_xscale('log'); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel('prominence'); ax.set_ylabel('fraction detected')
        ax.set_title(f'{tag}: detection rate vs prominence, by q'); ax.legend(fontsize=8)
    fig.suptitle('Prominence probe — are the missed peaks the ones that do not stand out?',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches='tight')
    print(f"\nsaved {path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)
    print("building model A (ssl1)");     modelA, a = build_model_from_ckpt(CONFIG, CKPT_A, device)
    print("building model B (baseline)"); modelB, _ = build_model_from_ckpt(CONFIG, CKPT_B, device)

    res_all, summary = [], {}
    for tag, path in DSETS:
        print(f"\n########## {tag}  ({path}) ##########", flush=True)
        G, F = run_dataset(tag, path, modelA, modelB, a, device)
        res_all.append((tag, G, F))
        summary[tag] = report(tag, G, F)
        np.savez(os.path.join(OUT, f'prominence_{tag}.npz'),
                 **{f'gt_{k}': v for k, v in G.items()},
                 **{f'fp_{k}': v for k, v in F.items()})

    make_plot(res_all, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'prominence_probe.png'))
    json.dump(summary, open(os.path.join(OUT, 'prominence_probe.json'), 'w'), indent=2, default=str)
    print("\nwrote prominence_probe.json")
    print("PROBE DONE")


if __name__ == '__main__':
    main()
