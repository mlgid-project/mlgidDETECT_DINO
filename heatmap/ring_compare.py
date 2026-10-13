"""How do rings LOOK in different image sources, in the processed (HE) image the network sees? CPU only; input = dump_dets / probe_frames pickles.

  python heatmap/ring_compare.py <name>=<dump.pkl>:<set> [<name>=<dump.pkl>:<set> ...] [--max_frames 60]
e.g. real41=hm_runs/plots/dumps/dets_he_tf32_seed42.pkl:41  legacy=hm_runs/plots/dumps/dets_he_tf32_seed42.pkl:sim  realsim=hm_runs/plots/dumps/dets_realsim_single_ep7_on_simframes.pkl:sim_bench2_single
For every ring (the frame's own 'gt_ring' flag if present, else the geometric criterion) the cross-section across the ring is averaged over the valid chi rows of its column:
  amplitude   peak of the profile minus the median of the flanks (|dq| 25-40 px), image units 0-1
  fwhm        full width at half maximum of that profile, px
  flank sd    sd of the single pixels in the flanks (the noise the ring has to stand out from), per frame region
  chi cv      patch-scale variation of the ring along chi: sd/mean of the ring-column values after averaging +-2 px across, in 8-row blocks (rows with 4+ valid pixels)
  snr         amplitude / (flank sd / sqrt(valid rows)) : how visible the ring is once integrated along chi
  gap         distance to the nearest other ring, px
  box w, h    ring box width / height (px)
Medians and quartiles are printed per source."""
import os, sys, types, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
from heatmap import evaluation as E


LABS = (('amp', 'amplitude (0-1 image units)'), ('fwhm', 'FWHM px'), ('sd', 'flank noise sd'), ('cv', 'chi variation cv'), ('snr', 'integrated snr'), ('gap', 'gap to nearest ring px'), ('w', 'box width px'), ('h', 'box height px'))


def ring_stats(f):
    img = np.asarray(f['img'], np.float32); mask = np.asarray(f['mask'], bool)
    gt = np.asarray(f['gt'], np.float32).reshape(-1, 4)
    if len(gt) == 0:
        return []
    ring = np.asarray(f['gt_ring'], bool) if 'gt_ring' in f else E.is_ring_geom(gt, mask)
    xs = np.sort((gt[ring, 0] + gt[ring, 2]) / 2)
    out = []
    for b in gt[ring]:
        xc = int(round((b[0] + b[2]) / 2)); W = img.shape[1]
        if xc < 45 or xc > W - 46:
            continue
        sl = (slice(None), slice(xc - 40, xc + 41)); m = mask[sl]; a = img[sl]
        rows = m[:, 36:45].all(1) & m[:, 40]                      # rows valid across the ring core
        if rows.sum() < 40:
            continue
        P = np.where(m[rows], a[rows], np.nan)
        prof = np.nanmean(P, 0)
        flank = np.r_[prof[:16], prof[-16:]]
        base = np.nanmedian(flank); amp = float(np.nanmax(prof[36:45]) - base)
        if amp <= 0:
            out.append(dict(xc=xc, amp=amp, fwhm=np.nan, sd=np.nan, cv=np.nan, snr=np.nan, gap=np.nan, w=b[2] - b[0], h=b[3] - b[1])); continue
        half = base + amp / 2; pk = 36 + int(np.nanargmax(prof[36:45]))
        l = pk
        while l > 0 and prof[l] > half: l -= 1
        r = pk
        while r < 80 and prof[r] > half: r += 1
        sd = float(np.nanstd(np.r_[P[:, :16].ravel(), P[:, -16:].ravel()]))
        col = np.nanmean(P[:, 38:43], 1)
        n8 = len(col) // 8
        blk = col[:n8 * 8].reshape(n8, 8).mean(1)
        cv = float(blk.std() / max(abs(blk.mean()), 1e-6)) if n8 >= 4 else np.nan
        others = np.delete(xs, np.searchsorted(xs, (b[0] + b[2]) / 2)) if len(xs) > 1 else np.array([])
        gap = float(np.min(np.abs(others - (b[0] + b[2]) / 2))) if len(others) else np.nan
        out.append(dict(xc=xc, amp=amp, fwhm=float(r - l), sd=sd, cv=cv, snr=amp / max(sd / np.sqrt(rows.sum()), 1e-6), gap=gap, w=float(b[2] - b[0]), h=float(b[3] - b[1])))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('src', nargs='+'); p.add_argument('--max_frames', type=int, default=60); p.add_argument('--x_split', type=float, default=0, help='if > 0: print the statistics separately for rings left / right of this q column')
    a = p.parse_args()
    cache = {}
    for s in a.src:
        name, rest = s.split('=', 1); path, ds = rest.rsplit(':', 1)
        if path not in cache:
            cache[path] = pickle.load(open(path, 'rb'))
        frames = cache[path]['sets'][ds][:a.max_frames]
        R = [r for f in frames for r in ring_stats(f)]
        groups = [('all', R)] if a.x_split <= 0 else [('all', R), (f'x < {a.x_split:g}', [r for r in R if r['xc'] < a.x_split]), (f'x >= {a.x_split:g}', [r for r in R if r['xc'] >= a.x_split])]
        for gname, RR in groups:
            print(f'\n== {name} [{gname}]: {len(RR)} rings in {len(frames)} frames')
            for k, lab in LABS:
                v = np.array([r[k] for r in RR], float); v = v[np.isfinite(v)]
                if len(v):
                    print(f'  {lab:30s} p25 {np.percentile(v, 25):8.3f}  median {np.median(v):8.3f}  p75 {np.percentile(v, 75):8.3f}')


if __name__ == '__main__':
    main()
