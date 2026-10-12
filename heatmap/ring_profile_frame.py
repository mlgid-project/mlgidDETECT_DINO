"""Look at the rings of ONE simulated frame along chi: is a ring patchy, partly saturated white, cut by the mask, or fading? CPU only.

  python heatmap/ring_profile_frame.py <bench dir>/sim_frames.npz <frame index> [--out profile.png]
Rings are listed left to right (q column), as a person reads the picture. For each ring the processed image is averaged over +-2 columns around the ring centre, row by row
(valid rows only) and the printout gives: valid rows, mean / min / p10 / p90 / max of that profile, the ring level relative to the flank median (amplitude per row),
share of rows that are white (>= 0.99), share of rows where the ring is not visible (amplitude < 25 % of its median amplitude), the longest such run of rows, the
slope of the amplitude along chi, and the 8-row cv. A text strip shows the profile (. low ... # white). --out draws one panel per ring (profile along chi, invalid rows shaded)."""
import os, sys, argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('npz'); p.add_argument('frame', type=int); p.add_argument('--out', default=None)
    a = p.parse_args()
    z = np.load(a.npz); k = a.frame
    img = z[f'{k}_img'].astype(np.float32); mask = z[f'{k}_mask'].astype(bool); boxes = z[f'{k}_boxes']; ring = z[f'{k}_ring'].astype(bool)
    rb = boxes[ring]
    order = np.argsort((rb[:, 0] + rb[:, 2]) / 2)
    rb = rb[order]
    print(f'frame {k}: {len(rb)} rings, {int((~ring).sum())} segments; valid share of the frame {mask.mean():.2f}')
    profs = []
    for n, b in enumerate(rb):
        xc = int(round((b[0] + b[2]) / 2)); W = img.shape[1]
        lo, hi = max(xc - 2, 0), min(xc + 3, W)
        rows = mask[:, lo:hi].all(1)
        col = np.where(rows, img[:, lo:hi].mean(1), np.nan)
        fl = np.concatenate([img[rows][:, max(xc - 40, 0):max(xc - 25, 1)].ravel(), img[rows][:, min(xc + 25, W - 1):min(xc + 41, W)].ravel()])
        base = float(np.median(fl)) if fl.size else np.nan
        amp = col - base
        v = amp[rows]
        med = float(np.median(v)) if v.size else np.nan
        white = float(np.mean(col[rows] >= 0.99)) if rows.any() else np.nan
        dark = amp[rows] < 0.25 * max(med, 1e-6)
        run = best = 0
        for d in dark:
            run = run + 1 if d else 0; best = max(best, run)
        yy = np.flatnonzero(rows)
        slope = float(np.polyfit(yy, v, 1)[0] * 512) if len(yy) > 10 else np.nan
        nb = len(v) // 8
        blk = v[:nb * 8].reshape(nb, 8).mean(1) if nb >= 4 else np.array([np.nan])
        cv = float(np.nanstd(blk) / max(abs(np.nanmean(blk)), 1e-6))
        strip = ''.join(' ' if not r else ('#' if c >= 0.99 else '.:-=+*%'[min(int(max(c, 0) * 7), 6)]) for r, c in zip(rows[::8], np.nan_to_num(col[::8])))
        print(f'  ring {n + 1} (x={xc}, box {b[2] - b[0]:.1f} x {b[3] - b[1]:.0f}): valid rows {int(rows.sum())}, level mean {np.nanmean(col):.2f} min {np.nanmin(col):.2f} p10 {np.nanpercentile(col, 10):.2f} '
              f'p90 {np.nanpercentile(col, 90):.2f} max {np.nanmax(col):.2f}; flank {base:.2f}; amplitude median {med:.2f}; white rows {white:.0%}; invisible rows {float(dark.mean()):.0%} (longest run {best}); '
              f'slope along chi {slope:+.2f} per frame height; cv8 {cv:.2f}')
        print('      chi profile (every 8th row, top = chi 0): ' + strip)
        profs.append((xc, rows, col, base))
    if a.out and profs:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(profs), 1, figsize=(10, 1.6 * len(profs)), sharex=True)
        ax = np.atleast_1d(ax)
        for i, (xc, rows, col, base) in enumerate(profs):
            ax[i].plot(np.arange(len(col)), col, lw=0.8); ax[i].axhline(base, color='gray', lw=0.5, ls='--'); ax[i].set_ylim(0, 1.05)
            ax[i].fill_between(np.arange(len(col)), 0, 1.05, where=~rows, color='red', alpha=0.15)
            ax[i].set_ylabel(f'ring {i + 1}\nx={xc}', fontsize=7)
        ax[-1].set_xlabel('chi row (red = invalid rows)'); fig.tight_layout(); fig.savefig(a.out, dpi=110)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
