"""Atlas (one self-contained HTML page) of converted SciData GIWAXS frames with the boxes of the detector (mlgidgui env).

  ~/miniconda3/envs/mlgidgui/bin/python heatmap/scidata_atlas.py --root <giwaxs_scidata_raw> --out <atlas.html> [--per 16 16 8 16] [--score 0.3] [--seed 0]

Frames are taken from <root>/converted/run*.h5 (detect / fit / match done by scidata_pipeline.py) and <root>/eval_both/match_frames.csv. Per run, a random sample of
  PVK called (mlgidMATCH perovskite probability >= 0.5), PVK missed, SnO2-only class, OPV / other  (--per gives the four counts).
Each card: left the image (log scale, q_xy / q_z 0 - 2.8 A^-1), right the same image with the boxes of detected_peaks with score >= --score
  (segments red, rings cyan, drawn as annular sectors: radius +- width/2, angle +- width/2 measured from the q_xy axis), fitted peaks orange circles, alpha-FAPbI3 rings thin green.
The boxes are those of the original mlgidDETECT detector, not of a trained heatmap / DINO model."""
import argparse, base64, csv, io, json, random
from pathlib import Path
import h5py, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

p = argparse.ArgumentParser()
p.add_argument('--root', required=True); p.add_argument('--out', required=True); p.add_argument('--per', type=int, nargs=4, default=[16, 16, 8, 16]); p.add_argument('--score', type=float, default=0.3); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
root = Path(a.root); rows = list(csv.DictReader(open(root / 'eval_both/match_frames.csv')))
PERO = [0.986, 1.395, 1.708, 1.973, 2.206, 2.417, 2.79]
LIM = 2.8
random.seed(a.seed)
groups = ['PVK called', 'PVK missed', 'SnO2 only', 'OPV / other']
pick = []
for run in sorted({r['run'] for r in rows}):
    R = [r for r in rows if r['run'] == run]
    pools = [[r for r in R if r['family'] == 'PVK' and float(r['p_perovskite']) >= 0.5], [r for r in R if r['family'] == 'PVK' and float(r['p_perovskite']) < 0.5],
             [r for r in R if r['family'] == 'SnO2 (no PVK)'], [r for r in R if r['family'] == 'OPV/other']]
    for g, pool, k in zip(groups, pools, a.per):
        for r in random.sample(pool, min(k, len(pool))):
            pick.append((g, r))
t = np.linspace(0, np.pi / 2, 120)
cards = []
for run in sorted({r['run'] for _, r in pick}):
    with h5py.File(root / f'converted/{run}.h5', 'r') as f:
        for g, r in [x for x in pick if x[1]['run'] == run]:
            d = f[r['entry']]['data']; img = np.asarray(d['img_gid_q'])[0]; qxy = np.asarray(d['q_xy']); qz = np.asarray(d['q_z'])
            v = np.log10(np.where(img > 0, img, np.nan) + 1); lo, hi = np.nanpercentile(v, 10), np.nanpercentile(v, 99.8)
            fr = d['analysis']; fr = fr[list(fr.keys())[0]] if len(fr) else None
            dp = fr['detected_peaks'][()] if fr is not None and 'detected_peaks' in fr else np.zeros(0)
            fp = fr['fitted_peaks'][()] if fr is not None and 'fitted_peaks' in fr else np.zeros(0)
            dp = dp[dp['score'] >= a.score] if len(dp) else dp
            fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
            for k, ax in enumerate(axs):
                ax.imshow(v, origin='lower', extent=[qxy.min(), qxy.max(), qz.min(), qz.max()], cmap='gray', vmin=lo, vmax=hi); ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect('equal')
                ax.tick_params(labelsize=7); ax.set_xlabel('q_xy (1/A)', fontsize=7)
            ax = axs[1]
            for q0 in PERO:
                ax.plot(q0 * np.cos(t), q0 * np.sin(t), color='#1baf7a', lw=0.5, alpha=0.6)
            for b in dp:
                r0, rw = float(b['radius']), float(b['radius_width']); ang, aw = (45.0, 90.0) if bool(b['is_ring']) else (float(b['angle']), float(b['angle_width']))
                a0, a1 = max(ang - aw / 2, 0), min(ang + aw / 2, 90)
                ax.add_patch(Wedge((0, 0), r0 + rw / 2, a0, a1, width=max(rw, 0.004), fill=False, ec='#00d5ff' if bool(b['is_ring']) else '#ff3b3b', lw=0.9))
            if len(fp):
                ok = fp['score'] >= 0.3
                ax.scatter(fp['q_xy'][ok], fp['q_z'][ok], s=22, facecolors='none', edgecolors='#ff9a2e', lw=0.9)
            axs[0].set_title(f"{r['cls']}  ai {float(r['ai']):g}", fontsize=8); axs[1].set_title(f"{len(dp)} boxes ({int(dp['is_ring'].sum()) if len(dp) else 0} rings), {int((fp['score'] >= 0.3).sum()) if len(fp) else 0} fits", fontsize=8)
            fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format='jpeg', dpi=78, pil_kwargs=dict(quality=82)); plt.close(fig)
            cards.append(dict(group=g, run=run, entry=r['entry'], cls=r['cls'], ai=float(r['ai']), raw=r['raw'], p=float(r['p_perovskite']), img=base64.b64encode(b.getvalue()).decode()))
            print(len(cards), '/', len(pick), flush=True)
cards.sort(key=lambda c: (groups.index(c['group']), c['run'], c['cls']))
out = Path(a.out); imgdir = out.with_suffix('').name + '_img'; (out.parent / imgdir).mkdir(exist_ok=True)
html = ['<!doctype html><html><head><meta charset="utf-8"><title>SciData GIWAXS atlas</title><style>body{font-family:sans-serif;background:#16181c;color:#ddd;margin:12px}a{color:#8cf}'
        '.c{display:inline-block;margin:6px;background:#22252b;padding:6px;border-radius:4px;vertical-align:top}.c img{display:block;width:700px;max-width:96vw}.m{font-size:12px;margin-top:3px}</style></head><body>',
        '<h2>SciData GIWAXS atlas</h2><div style="max-width:900px;font-size:13px">Left: converted frame (log). Right: boxes of the mlgidDETECT detector (score &gt;= %.1f; red = segments, cyan = rings), fitted peaks (orange circles), alpha-FAPbI3 rings (green). '
        'Groups by the mlgidMATCH perovskite call (probability &gt;= 0.5) against the class name. Random samples, seed %d.</div><p>%s</p>' % (a.score, a.seed, ' | '.join(f'<a href="#g{k}">{g}</a>' for k, g in enumerate(groups)))]
for k, g in enumerate(groups):
    html.append(f'<h3 id="g{k}">{g}</h3>')
    for c in [c for c in cards if c['group'] == g]:
        fn = f"{imgdir}/{c['run']}_{c['entry']}.jpg"; (out.parent / fn).write_bytes(base64.b64decode(c['img']))
        html.append(f"<div class=c><img src=\"{fn}\"><div class=m>{c['run']} {c['entry']} | {c['cls']} | ai {c['ai']:g} | p(perovskite) {c['p']}<br>{c['raw']}</div></div>")
html.append('</body></html>')
out.write_text('\n'.join(html)); print('written', out, len(cards), 'cards, images in', imgdir)
