"""Review figures for the background-diversity work: real donor vs current vs A+B2 vs B1.

Four columns, so the question "does it still look like a real background" can be answered by eye
against an actual donor frame in the same row treatment:
    REAL        a clean donor frame, the ground truth for what a background looks like
    CURRENT     what training sees today: flat mosaic crop x ONE OF ~90 donor envelopes
    A+B2        fresh per-frame crop x an envelope SAMPLED from the donor PCA
    B1          random-phase spectral surrogate of the texture, same envelope treatment as A+B2

Both a raw log-scale view and the preprocessed view the network actually receives, plus radial
profiles so the large-scale shape can be checked numerically rather than by impression.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from realbkg_sim.mosaic_background import MosaicBackground
from realbkg_sim.detector_masks import MaskBank
from realbkg_sim.background_v2 import BackgroundV2, HEIGHT, WIDTH, _orient_low_q_bright
from realbkg_simulation import apply_contrast, CHAIN

OUT = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/09_background_v2'
os.makedirs(OUT, exist_ok=True)
N = 5
# dataviz default categorical slots 1-4, in fixed order (node unavailable on the cluster, so the
# validator could not be run here; these are the reference instance's pre-validated values)
C = {'REAL': '#2a78d6', 'CURRENT': '#eb6834', 'A+B2': '#1baf7a', 'B1': '#eda100'}

mb = MosaicBackground(seed=7)
mk = MaskBank(seed=7, keep='default')
bv = BackgroundV2(mb, canvas=(1536, 3072), n_pc=6, seed=7)
print(f'{bv.n_donors} donor envelopes, {bv.env_var_explained:.1%} of their log-variance in 6 PCs',
      flush=True)

def real_donor(i):
    g = np.nan_to_num(np.asarray(mb.frames[i], np.float32), nan=0.0)
    g = cv2.resize(g, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    g = _orient_low_q_bright(g)
    lv = float(mb.med[i])
    m = float(np.median(g[g > 0])) if (g > 0).any() else 1.0
    return (np.maximum(g, 0)/max(m, 1e-6)*lv).astype(np.float32)

rows = []
di = np.random.default_rng(7).choice(bv.n_donors, N, replace=False)
for k in range(N):
    msk, _ = mk.draw()
    i = int(di[k])
    # EXPOSURE-MATCH THE ROW. Without this each column draws its own exposure class and the levels
    # differ by up to 17x, so the eye compares brightness instead of texture. Matching by POOL
    # rather than by rescaling keeps the level coming from the tiles, which is what ties the
    # graininess to the brightness -- an imposed level would give a frame noise it cannot have.
    lv = float(mb.med[i])
    pool = np.nonzero((mb.med >= lv/1.5) & (mb.med <= lv*1.5))[0]
    if len(pool) < 4:
        pool = np.argsort(np.abs(mb.med - lv))[:8]
    bv.new_canvas(pool=pool)
    cur, _ = mb.background(mask=msk, pool=pool)           # today's pipeline
    ab, _ = bv.frame(mask=msk, envelope='pca')            # A + B2
    b1, _ = bv.surrogate(mask=msk, envelope='pca')        # B1
    rd = real_donor(i)
    rows.append((('REAL', rd*msk), ('CURRENT', cur), ('A+B2', ab), ('B1', b1), msk))
    print(f'  row {k+1}/{N} built (donor level {lv:.0f} counts, pool {len(pool)})', flush=True)

def panel(fname, mode):
    fig, ax = plt.subplots(N, 4, figsize=(19, 2.9*N))
    for r, row in enumerate(rows):
        msk = row[4]
        for c, (tag, img) in enumerate(row[:4]):
            a = ax[r, c]
            if mode == 'raw':
                v = np.log10(np.maximum(img, 1.0))
                good = v[msk]
                a.imshow(v, cmap='magma', aspect='auto',
                         vmin=np.percentile(good, 1), vmax=np.percentile(good, 99.5))
            else:
                a.imshow(apply_contrast(img.astype(np.float64), msk, CHAIN),
                         cmap='magma', aspect='auto', vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            mv = img[msk]; mv = mv[mv > 0]
            if len(mv):
                a.text(0.015, 0.03, f'median {np.median(mv):.0f} counts', transform=a.transAxes,
                       fontsize=8, color='#ffffff', va='bottom')
            for s in a.spines.values():
                s.set_edgecolor(C[tag]); s.set_linewidth(2.0)
            if r == 0:
                a.set_title(tag, color=C[tag], fontsize=13, fontweight='bold', pad=8)
            if c == 0:
                a.set_ylabel(f'example {r+1}', fontsize=10, color='#52514e')
    fig.suptitle('Background sources — ' + ('raw counts, log scale' if mode == 'raw'
                 else 'after the contrast chain, i.e. what the network sees'),
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(f'{OUT}/{fname}', dpi=105)
    plt.close(fig)
    print('wrote', fname, flush=True)

panel('panels_raw.png', 'raw')
panel('panels_preprocessed.png', 'pre')

# ---- radial profiles: mean over chi per q column, normalised to each frame's median
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
prof = {k: [] for k in C}
for row in rows:
    msk = row[4]
    for tag, img in row[:4]:
        d = np.where(msk, img, np.nan)
        with np.errstate(invalid='ignore'):
            p = np.nanmean(d, axis=0)
        prof[tag].append(p/max(np.nanmedian(p), 1e-9))
# left: every example, thin; right: the spread
for tag in C:
    P = np.vstack(prof[tag])
    x = np.arange(WIDTH)
    ax[0].plot(x, np.nanmedian(P, 0), color=C[tag], lw=2.0, label=tag)
    ax[1].fill_between(x, np.nanmin(P, 0), np.nanmax(P, 0), color=C[tag], alpha=0.22, lw=0)
    ax[1].plot(x, np.nanmedian(P, 0), color=C[tag], lw=2.0, label=tag)
for a, t in ((ax[0], 'median profile over the 5 examples'),
             (ax[1], 'spread across the 5 examples (min–max band)')):
    a.set_yscale('log'); a.set_title(t, fontsize=12)
    a.set_xlabel('q (pixel column)'); a.set_ylabel('mean intensity over χ  /  frame median')
    a.grid(True, alpha=0.18, lw=0.6); a.set_axisbelow(True)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.legend(frameon=False, fontsize=10)
fig.suptitle('Large-scale shape. CURRENT draws its envelope from bkg.npy — the REJECTED 189-donor\n'
             'bank with unremoved diffraction — not from the 90 clean donors the tiles come from.',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT}/radial_profiles.png', dpi=110)
plt.close(fig)
print('wrote radial_profiles.png')

# ---- numbers to go with the pictures
print('\nSTATS (median over the 5 examples, valid pixels only)')
print(f"  {'source':10} {'median':>10} {'max/med':>9} {'noise/sqrt(I)':>14}")
for tag in C:
    md, dr, cf = [], [], []
    for row in rows:
        msk = row[4]
        img = dict((t, i) for t, i in row[:4])[tag]
        v = img[msk]; v = v[v > 0]
        if not len(v): continue
        md.append(np.median(v)); dr.append(v.max()/max(np.median(v), 1e-9))
        sm = cv2.GaussianBlur(img, (0, 0), 16.0)
        res = (img-sm)[msk]
        cf.append(np.std(res)/max(np.sqrt(np.median(v)), 1e-9))
    print(f'  {tag:10} {np.median(md):10.1f} {np.median(dr):9.1f} {np.median(cf):14.2f}')
print('\n(noise/sqrt(I) ~ 1 means Poisson-like graininess; the REAL column is the target)')
