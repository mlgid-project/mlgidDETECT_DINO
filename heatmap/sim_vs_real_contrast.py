"""Does the real-background simulator (realsim) show the network the same kind of rings / segments / mask edges as the real validation frames? Step 0 of the realsim E9 diagnosis.
Measures, on N simulated frames (E9 settings, box convention 3,3, boxes as the network trains on them) and on the real organic + 41 frames (dump_dets pickle, GT scaled to 3,3), with the
SAME feature as heatmap/miss_analysis.py: contrast = (core mean - surrounding annulus mean) / annulus std of a GT box in the image the network sees.
  python heatmap/sim_vs_real_contrast.py --real <dump.pkl> --out <file.txt> [--n 150] [--opt E9|E12a ...]
Prints: per class (ring / segment) the contrast quantiles, split by q column quarter; box width / height quantiles; boxes per frame; image level vs distance to the nearest invalid pixel (edge rim).
GPU + about 13 GB RAM (one simulator in-process); run it only when no training is running."""
import os, sys, ast, pickle, argparse, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

E9 = ["amplitude_mode='fitted'", 'intensity_decades=None', 'ring_gain=0.4', 'ring_contrast_min=0.3', 'ring_chi_texture=0.05', 'ring_chi_corr=(8.0,64.0)', 'seg_sc_scale=0.6', 'ring_sq_scale=1.0',
      'ring_w_coef=2.2', 'ring_no_elongate=True', 'ring_cap=4.0', 'ring_chi_floor=0.5', 'ring_follow_bkg=True', 'width_jitter=(0.2,0.832)', 'bkg_noise_blur=(0.0,1.2)', 'bkg_noise_power=1.5',
      'mask_keep=[0,3,6,7]', 'ring_x_share=(0.106,0.179,0.257,0.238,0.220)', 'seg_wide_frac=0.0', 'p_ring=0.85', 'n_powder=(1,2)', 'rings_cap=(6,30)', 'n_oriented=(2,3)', 'spots_cap=(25,120)',
      'sq_q_slope=0.6', 'mask_wedge_only_frac=0.2']
Q = (5, 25, 50, 75, 95)


def contrast(img, mask, b):
    H, W = img.shape
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, x1 = max(x0, 0), min(max(x1, x0 + 1), W); y0, y1 = max(y0, 0), min(max(y1, y0 + 1), H)
    core = img[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    if core.size == 0:
        return np.nan
    pad = 6
    X0, X1, Y0, Y1 = max(x0 - pad, 0), min(x1 + pad, W), max(y0 - pad, 0), min(y1 + pad, H)
    ann = np.ones((Y1 - Y0, X1 - X0), bool); ann[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    ann &= mask[Y0:Y1, X0:X1]
    a = img[Y0:Y1, X0:X1][ann]
    if a.size < 8:
        return np.nan
    return float((core.mean() - a.mean()) / max(a.std(), 1e-3))


def collect(frames):
    """frames: iterable of (img [H,W] float, mask bool, gt xyxy [N,4], is_ring bool [N]) -> dict of per-box lists + edge profile + counts"""
    rows, edge, nr, ns = [], [], [], []
    for img, mask, gt, ring in frames:
        c = [contrast(img, mask, b) for b in gt]
        for b, r, ct in zip(gt, ring, c):
            rows.append((bool(r), 0.5 * (b[0] + b[2]), b[2] - b[0], b[3] - b[1], ct))
        nr.append(int(np.sum(ring))); ns.append(int(len(gt) - np.sum(ring)))
        d = distance_transform_edt(mask); v = img[mask]; ref = np.median(img[mask & (d > 40)]) if (mask & (d > 40)).any() else np.nan
        sd = np.std(img[mask & (d > 40)]) if (mask & (d > 40)).any() else np.nan
        prof = [(img[mask & (d > lo) & (d <= hi)].mean() - ref) / max(sd, 1e-6) if (mask & (d > lo) & (d <= hi)).any() else np.nan for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 40))]
        edge.append(prof)
    return np.array(rows, float), np.array(edge, float), nr, ns


def q(v):
    v = v[np.isfinite(v)]
    return ' / '.join(f'{x:.2f}' for x in np.percentile(v, Q)) if len(v) else 'n/a'


def report(name, rows, edge, nr, ns, out):
    p = lambda s: (print(s), out.write(s + '\n'))
    p(f'== {name}: {len(nr)} frames, {int(rows[:, 0].sum())} ring boxes, {int((1 - rows[:, 0]).sum())} segment boxes; per frame rings {np.mean(nr):.1f} (median {np.median(nr):.0f}), segments {np.mean(ns):.1f} (median {np.median(ns):.0f})')
    for cls, lab in ((1, 'RING'), (0, 'SEGMENT')):
        s = rows[rows[:, 0] == cls]
        p(f'  {lab}: contrast p5/25/50/75/95 {q(s[:, 4])} | width px {q(s[:, 2])} | height px {q(s[:, 3])}')
        for k, (lo, hi) in enumerate(((0, 256), (256, 512), (512, 768), (768, 1025))):
            t = s[(s[:, 1] >= lo) & (s[:, 1] < hi)]
            p(f'     q column [{lo},{hi}): n={len(t):5d} contrast p5/25/50/75/95 {q(t[:, 4])}')
    p('  image level (in std of the frame interior) at distance d px from the nearest invalid pixel, mean over frames; bins (0,1] (1,2] (2,4] (4,8] (8,16] (16,40]: '
      + ' '.join(f'{np.nanmean(edge[:, i]):+.2f}' for i in range(edge.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--opt', nargs='*', default=None, help='realsim options (default = E9 list); e.g. bkg_noise_edge_fix=True ring_cap=8.0 (added to / override the E9 list)')
    ap.add_argument('--box_coef', default='3.0,3.0'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = open(a.out, 'w')
    # real frames
    real = pickle.load(open(a.real, 'rb'))['sets']
    for ds in ('41', 'organic'):
        fr = []
        for f in real[ds]:
            g = np.asarray(f['gt']).reshape(-1, 4); m = np.asarray(f['mask']).astype(bool)
            rows = np.where(m.any(axis=1))[0]; span = rows.max() - rows.min() + 1
            fr.append((np.asarray(f['img'], np.float32), m, g, (g[:, 3] - g[:, 1]) >= 0.7 * span))
        report('REAL ' + ds + ' (GT scaled to ' + a.box_coef + ')', *collect(fr), out)
    # simulated frames
    os.environ.setdefault('GIWAXS_WORK', '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data')
    from realsim.factory import make_realbkg_sim
    items = list(E9)
    for o in (a.opt or []):
        k = o.split('=', 1)[0]; items = [i for i in items if i.split('=', 1)[0] != k] + [o]
    ov = {k: ast.literal_eval(v) for k, v in (o.split('=', 1) for o in items)}
    coef = tuple(float(v) for v in a.box_coef.split(','))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    sim = make_realbkg_sim(device='cuda' if torch.cuda.is_available() else 'cpu', box_coef=coef, overrides=ov)
    fr = []
    while len(fr) < a.n:
        try:
            img, boxes, mask, is_ring = sim.simulate_img()
        except Exception:
            continue
        img = img if img.dim() == 3 else img[None]
        fr.append((img[0].float().cpu().numpy(), np.asarray(mask.cpu() if torch.is_tensor(mask) else mask).astype(bool), boxes.float().cpu().numpy().reshape(-1, 4), np.asarray(is_ring.cpu()).astype(bool)))
    out.write('SIM options: ' + ' '.join(items) + '\n')
    report(f'SIM ({a.n} frames, {a.box_coef})', *collect(fr), out)
    out.close()


if __name__ == '__main__':
    main()
