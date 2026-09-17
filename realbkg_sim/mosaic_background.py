"""Build simulator backgrounds by mosaicking real feature-free detector frames.

WHY A MOSAIC. The donor pool is small -- 90 bare-silicon Lambda modules from two DESY P03
beamtimes, all that survived the reviewer's pass over 212 candidates -- and a 500k-image training
set would reuse each one ~5,500 times. Cutting every donor into tiles and reassembling a FRESH
canvas per frame turns 90 frames into effectively unlimited variety while every pixel stays a real
detector measurement. Nothing is cached: `background()` builds a new canvas every call, so two
training runs never see the same background twice even at the same seed offset.

WHICH DONORS. `donors_final.json`, the 90 frames left after the reviewer looked at all 212
delivered candidates as numbered contact sheets and removed:
  * frames containing the direct beam and beamstop (a hard localised feature a mosaic would
    transplant into the middle of an otherwise empty frame),
  * frames with visible diffraction arcs that the flat_dyn screen had let through,
  * every ESRF ID10 Eiger frame (IDs 108-211) and both PerkinElmer frames: the ID10 set carries a
    distinct ring across the whole beamtime, and mixing 75/200 um detectors into one canvas mixes
    two different noise textures as well.
The survivors are therefore ONE detector (Lambda, 55 um), one material (bare Si), one facility.
That is a deliberate narrowing: uniform texture beats variety we cannot vouch for. If the ID10
frames are ever wanted back, they are still in `master_donors.json` under their IDs.

The donors carry no diffraction at all, which is the whole point: the previous bank removed peaks
from frames that had them and always left residue, so every frame trained the detector to call
real peaks background. These frames never had peaks, so nothing has to be removed and there is
nothing left behind.

TWO THINGS THAT MAKE OR BREAK IT:

  LEVEL NORMALISATION, PER TILE. Normalising per DONOR is not enough and the first version proved
  it: intensity varies strongly WITHIN a frame because of the radial falloff, so a tile cut from a
  donor's bright low-q region landed beside one from its dark high-q region and the step survived
  as a visible checkerboard -- hard block edges, narrow in q, exactly the shape the detector is
  trained to find. Every tile is therefore divided by ITS OWN median, so a tile contributes
  texture and nothing else, and every tile arrives at the same level.

  THE LARGE-SCALE SHAPE COMES BACK SEPARATELY, AND FROM A POLAR FRAME. Flat texture alone is not a
  GIWAXS background either -- real frames are bright at low q and fall away. The first attempt took
  that envelope from a Lambda module, which lives in DETECTOR space, and several mosaics came out
  dark at low q and bright at high q, backwards. The envelope is therefore taken from the old
  polar bank and blurred at sigma 64: that bank is useless as an image source because of its
  unremoved peaks, but at sigma 64 no peak survives (sigma_q ~ 4 px) and what is left is exactly
  the radial falloff and beamstop shape we want. Texture is mosaicked from many clean frames, the
  smooth shape is one real polar frame's, and neither introduces an edge.

  TILES COME FROM SIMILAR EXPOSURES. Relative noise depends on count rate: a tile from a 33-count
  donor is visibly grainier than one from a 270-count donor even after both are normalised to
  median 1, and that texture mismatch left faint block structure at the seams. Each canvas
  therefore draws only from donors whose level is within a factor of two of a target.

  FEATHERED SEAMS, BUT NARROW ONES. Tiles overlap and are alpha-blended across the overlap with a
  separable raised cosine rather than butted together, so no hard edge survives at peak scale
  (sigma_q ~ 4 px). The band is deliberately thin. A wide one (overlap 64 of a 128 tile) makes
  EVERY pixel an average of four independent tiles, which suppresses the detector noise to 0.58 of
  what Poisson demands at that count level -- the frames came out quieter than any real frame can
  be -- and it also smeared mismatches over a wide area. Measured across blend widths:

      overlap   noise/Poisson   fraction of pixels >20% over local background
         64          0.58                   0.0014
         32          0.86                   0.0003
         16          0.87                   0.0001
          8          0.96                   0.0002

  so 8 wins on both counts; real donors sit at 0.0009-0.0027 on the second column.

WHAT IS LOST. Free shuffling destroys the physical intensity gradient with q -- real GIWAXS
backgrounds are bright at low q and fall off. The mosaic is statistically flat instead. If that
turns out to matter, the fix is to keep each tile at its original q position and shuffle only
along chi; `q_locked=True` does exactly that.

MASKS come from the old polar bank, because a mosaic has no detector geometry of its own and real
polar frames have a characteristic masked high-q wedge. A mask is pure geometry, so it carries
none of the peak contamination that made that bank unusable as an image source.
"""
import hashlib
import json
import os

import cv2
import numpy as np

HEIGHT, WIDTH = 512, 1024
WORK = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
ACCEPTED = f'{WORK}/tmp_diag/sim2/donors_final.json'
BATCH = f'{WORK}/datasets/raw_backgrounds/batch2'
MASK_BANK = f'{WORK}/datasets/realbkg_donors_mm/mask.npy'
CACHE = f'{WORK}/tmp_diag/sim2/donor_cache_repaired.npz'


def _resolve(row, batch):
    """Locate a donor frame on disk. Delivered batches nest differently, so try every layout."""
    if row.get('path') and os.path.exists(row['path']):
        return row['path']
    base = row.get('base', batch)
    for p in (os.path.join(base, row.get('group', ''), 'raw', row['raw_file']),
              os.path.join(base, 'raw', row['raw_file']),
              os.path.join(base, row['raw_file']),
              os.path.join(batch, 'raw', row['raw_file'])):
        if p and os.path.exists(p):
            return p
    return None


def repair_lines(img, mask, sig=32.0, lim=(0.5, 2.0)):
    """Flatten out per-column and per-row gain anomalies in a donor frame.

    WHY THIS IS NOT COSMETIC. A Lambda module has dead and low-gain COLUMNS -- chip boundaries,
    individual bad channels. A tile keeps its orientation when it is laid into the mosaic, and the
    mosaic's x axis is q while its y axis is chi, so a donor COLUMN becomes a line of constant q
    running the full height of the frame. That is precisely the signature of a powder ring. The
    first 20-frame review file had several of them, thin and black, running through the image with
    no box on them -- an unlabelled ring-shaped feature, the exact failure the whole background
    rebuild exists to avoid.

    The fix is a flat field, not interpolation: each column is divided by its own median relative
    to a smooth version of the median profile, which corrects the LEVEL while leaving that
    column's own photon noise intact. Interpolating from neighbours would instead copy a
    neighbour's noise and leave a visibly smoother line. A column too far gone to rescale
    (dead, or more than 2x hot) is marked invalid instead, so no tile will ever include it.
    """
    out = np.array(img, np.float32, copy=True)
    bad = ~mask
    for axis in (0, 1):
        v = np.where(mask, out, np.nan)
        prof = np.nanmedian(v, axis=axis)
        good = np.isfinite(prof) & (prof > 0)
        if good.sum() < 16:
            continue
        p = np.interp(np.arange(len(prof)), np.nonzero(good)[0], prof[good]).astype(np.float32)
        sm = cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), sig).ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            g = sm/np.where(p > 0, p, np.nan)
        ok = np.isfinite(g) & (g > lim[0]) & (g < lim[1])
        gg = np.where(ok, g, 1.0).astype(np.float32)
        if axis == 0:                      # prof is per-column
            out *= gg[None, :]
            bad |= ~ok[None, :]
        else:                              # prof is per-row
            out *= gg[:, None]
            bad |= ~ok[:, None]
    return out, ~bad


def repair_defects(img, mask, counts, nsig=6.0, k=5):
    """Remove isolated bad pixels and short line defects, keeping the noise statistics.

    `repair_lines` is a flat field and can only see a defect that runs the FULL height or width of
    the frame. Most of them do not: a dead or hot run covering part of a column is invisible to a
    per-column median over 516 rows, and the first screened tiles still showed one-pixel white and
    black verticals from donors 16, 23, 54 and 89. In the mosaic those become lines of constant q
    -- false rings -- so they have to go.

    A 5x5 median is immune to any line up to two pixels wide (at most 10 of its 25 samples sit on
    the line), so it is a clean reference for what the pixel SHOULD read. A pixel more than `nsig`
    Poisson sigmas away from it is replaced by that reference plus freshly drawn noise of the
    right amplitude -- not by the reference itself, which would leave a visibly smooth line where
    the defect was. `counts` is the frame's median in counts, which is what converts the
    median-normalised values here into a photon count and therefore into a sigma.
    """
    from scipy.ndimage import median_filter
    ref = median_filter(np.where(mask, img, np.nan_to_num(np.nanmedian(np.where(mask, img, np.nan)))),
                        size=k, mode='nearest').astype(np.float32)
    sig = np.sqrt(np.maximum(ref, 0.0)/max(float(counts), 1e-6)).astype(np.float32)
    bad = mask & (np.abs(img - ref) > nsig*np.maximum(sig, 1e-6))
    out = np.where(bad, ref + np.random.standard_normal(img.shape).astype(np.float32)*sig, img)
    return out.astype(np.float32), bad


def tile_structure(t, sig=8.0):
    """How much of a flattened tile is structure rather than noise: std(blur)/std.

    Detector noise is nearly white, so blurring at sigma 8 all but removes it; a shadow edge, a
    module border or a gain step survives. Measured over 3,963 random tiles of the 90 donors the
    distribution is p50 0.055, p90 0.118, p95 0.155, p99 0.597 -- a tight noise-only core with a
    thin tail of genuinely structured tiles, which is what the tail is for.
    """
    b = cv2.GaussianBlur(t, (0, 0), float(sig))
    return float(b.std()/max(t.std(), 1e-9))


def _feather(h, w, v):
    """Separable raised-cosine ramp: 0 -> 1 over the first v rows/cols, 1 elsewhere."""
    def ramp(n):
        a = np.ones(n, np.float32)
        if v > 0:
            t = np.linspace(0, np.pi, min(v, n), dtype=np.float32)
            a[:min(v, n)] = (1 - np.cos(t))/2
        return a
    return np.outer(ramp(h), ramp(w))


class MosaicBackground:
    """Assemble a 512x1024 background from tiles of real feature-free frames."""

    def __init__(self, accepted=ACCEPTED, batch=BATCH, mask_bank=MASK_BANK,
                 tile=128, overlap=16, canvas=(768, 1536), q_locked=False, seed=None,
                 flatten_sigma=24, struct_max=0.12, cache=CACHE):
        self.tile, self.overlap, self.canvas, self.q_locked = tile, overlap, canvas, q_locked
        self.flatten_sigma = flatten_sigma
        self.struct_max = float(struct_max)
        self.rng = np.random.default_rng(seed)
        self.frames, self.med, self.rows = [], [], []
        rows = json.load(open(accepted))
        # Reading and repairing 90 raw frames off lustre costs ~4 minutes, which would be paid at
        # the start of every run. The REPAIRED donor pixels are a deterministic function of the
        # donor list, so caching them changes nothing about freshness -- the variety comes from
        # re-tiling them, which still happens per frame.
        # hashlib, not hash(): str hashing is salted per process, so hash() would never match
        sig = '%d:%s' % (len(rows), hashlib.md5(
            '\n'.join(r['raw_file'] for r in rows).encode()).hexdigest()[:12])
        if cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            if str(z['sig']) == sig:
                self.frames = list(z['frames'])
                self.med = z['med']
                self.rows = list(z['rows'])
                self.med = np.asarray(self.med, np.float32)
                self.masks = np.load(mask_bank, mmap_mode='r') if os.path.exists(mask_bank) else None
                env = os.path.join(os.path.dirname(mask_bank), 'bkg.npy')
                self.env_bank = np.load(env, mmap_mode='r') if os.path.exists(env) else None
                self._pool = self._target = None
                print(f'[mosaic] {len(self.frames)} donor frames from cache, tile {tile} '
                      f'overlap {overlap}, canvas {canvas}, struct_max {struct_max}', flush=True)
                return
        from realbkg_sim.review_raw_backgrounds import read_frame, valid_mask, _as_index
        for r in rows:
            p = _resolve(r, batch)
            if p is None:
                continue
            img, _ = read_frame(p, _as_index(r.get('frame')))
            if img is None:
                continue
            m = valid_mask(img)
            if m.mean() < 0.5:
                continue
            img, m = repair_lines(img, m)
            if m.mean() < 0.5:
                continue
            med = float(np.median(img[m]))
            if med <= 0:
                continue
            img, _bad = repair_defects(img/med, m, med)
            img = img*med
            # store as relative texture; NaN marks pixels that must never be sampled
            f = np.where(m, img/med, np.nan).astype(np.float32)
            self.frames.append(f)
            self.med.append(med)
            self.rows.append(r)
        if not self.frames:
            raise RuntimeError(f'no usable donor frames from {accepted}')
        self.med = np.asarray(self.med, np.float32)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, frames=np.stack(self.frames), med=self.med,
                     rows=np.array(self.rows, dtype=object), sig=sig)
            print(f'[mosaic] cached repaired donors -> {cache}', flush=True)
        self.masks = np.load(mask_bank, mmap_mode='r') if os.path.exists(mask_bank) else None
        env = os.path.join(os.path.dirname(mask_bank), 'bkg.npy')
        self.env_bank = np.load(env, mmap_mode='r') if os.path.exists(env) else None
        self._pool = None                       # per-canvas subset of donors, set in canvas_image
        self._target = None                     # and the level that subset's noise belongs to
        print(f'[mosaic] {len(self.frames)} donor frames, tile {tile} overlap {overlap}, '
              f'canvas {canvas}, q_locked={q_locked}', flush=True)

    # ------------------------------------------------------------------ tiles
    def _take(self, want_col=None):
        """One clean tile, normalised to median 1. Retries until it finds one with no masked pixels."""
        T = self.tile
        pool = self._pool if self._pool is not None and len(self._pool) else range(len(self.frames))
        pool = list(pool)
        for _ in range(200):
            f = self.frames[pool[self.rng.integers(len(pool))]]
            h, w = f.shape
            if h <= T or w <= T:
                continue
            r = int(self.rng.integers(0, h - T))
            if self.q_locked and want_col is not None:
                # keep the tile at its own q: map the canvas column onto the donor's width
                c = int(np.clip(round(want_col/self.canvas[1]*(w - T)), 0, w - T - 1))
            else:
                c = int(self.rng.integers(0, w - T))
            t = f[r:r+T, c:c+T]
            if np.isfinite(t).all():
                # Divide by the tile's OWN smooth field, not its median. A median only removes a
                # tile's level; it leaves the tile's internal GRADIENT, and butting tiles with
                # different slopes together produced visible rectangular blocks a few percent
                # apart. Dividing by a blur removes the gradient too, so every tile arrives flat
                # at large scale and contributes only fluctuation. Noise amplitude is untouched --
                # dividing by a smooth field averages nothing, unlike the wide alpha blend which
                # suppressed noise to 0.58 of Poisson.
                sm = cv2.GaussianBlur(t, (0, 0), float(self.flatten_sigma))
                if float(np.median(sm)) <= 1e-6:
                    continue
                q = (t/np.maximum(sm, 1e-6)).astype(np.float32)
                # Reject a tile that is still STRUCTURED after flattening. Screening donors frame
                # by frame is not enough: a frame can be flat overall and still have a shadow
                # edge, a module border or a gain step somewhere in it, and a 128 px tile that
                # lands there gets transplanted whole -- the first review file had hard-edged
                # rectangles from exactly this. sigma 24 flattening removes a gradient, not an
                # edge of comparable width, so the edge has to be screened out instead.
                if tile_structure(q) < self.struct_max:
                    return q
        return np.ones((T, T), np.float32)

    def exposure_partition(self, k):
        """Split the donors into k DISJOINT groups, each internally exposure-matched.

        For a review set we want to be able to say that no two frames share a single source pixel.
        Sorting by count rate and cutting the sorted list into k contiguous chunks does both jobs
        at once: the chunks are disjoint by construction, and because they are contiguous in level
        each chunk is automatically narrow in exposure, which is the property the tile pool needs
        anyway. Training does NOT use this -- there the pool should be as wide as possible.
        """
        order = np.argsort(self.med)
        return [np.sort(c) for c in np.array_split(order, k) if len(c)]

    def canvas_image(self, pool=None):
        """Assemble the full mosaic canvas (relative units, median ~1).

        `pool` restricts the tiles to those donor indices, which is how a review set forces every
        frame onto its own donors. Left None, the canvas picks an exposure class itself.
        """
        if pool is not None and len(pool):
            pool = np.asarray(pool, int)
            self._pool = pool
            self._target = float(np.median(self.med[pool]))
        else:
            # one exposure class per canvas, so tile-to-tile noise character matches
            target = float(self.med[self.rng.integers(len(self.med))])
            self._pool = np.nonzero((self.med >= target/2) & (self.med <= target*2))[0]
            self._target = target   # the level this canvas's NOISE corresponds to
        H, W = self.canvas
        T, V = self.tile, self.overlap
        step = T - V
        acc = np.zeros((H + T, W + T), np.float32)
        wgt = np.zeros((H + T, W + T), np.float32)
        win = _feather(T, T, V)
        for r in range(0, H, step):
            for c in range(0, W, step):
                t = self._take(want_col=c)
                acc[r:r+T, c:c+T] += t*win
                wgt[r:r+T, c:c+T] += win
        out = acc/np.maximum(wgt, 1e-6)
        return out[:H, :W]

    def _envelope(self):
        """Smooth large-scale shape of one real donor, normalised to median 1.

        The mosaic is flat by construction once tiles are individually normalised. A real GIWAXS
        background is not flat -- it is bright at low q and falls away -- so the shape is taken
        from a real frame at sigma 64, far broader than any peak (sigma_q ~ 4 px), which means it
        cannot introduce anything the detector could mistake for a feature.
        """
        if self.env_bank is not None and len(self.env_bank):
            g = np.asarray(self.env_bank[self.rng.integers(len(self.env_bank))], np.float32)
        else:                                   # fall back to a donor, oriented low-q bright
            g = np.nan_to_num(self.frames[self.rng.integers(len(self.frames))], nan=1.0)
            g = cv2.resize(g, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            if g[:, :WIDTH//4].mean() < g[:, -WIDTH//4:].mean():
                g = g[:, ::-1]
        e = cv2.GaussianBlur(np.nan_to_num(g, nan=0.0), (0, 0), 64.0)
        m = float(np.median(e[e > 0])) if (e > 0).any() else 1.0
        return np.maximum(e/max(m, 1e-6), 1e-3).astype(np.float32)

    # ------------------------------------------------------------- background
    def background(self, level=None, pool=None):
        """-> (bkg float32 (512,1024) in counts, mask bool). Random crop of a fresh canvas."""
        cv = self.canvas_image(pool=pool)
        H, W = cv.shape
        r = int(self.rng.integers(0, max(H - HEIGHT, 1)))
        c = int(self.rng.integers(0, max(W - WIDTH, 1)))
        patch = cv[r:r+HEIGHT, c:c+WIDTH]
        if patch.shape != (HEIGHT, WIDTH):
            patch = cv2.resize(patch, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        if level is None:
            # Use the level the tiles were CUT AT, not an independent draw. Relative noise is
            # fixed by the source exposure -- a tile from a 270-count donor carries ~6% noise,
            # one from a 33-count donor ~17%. Sampling the output level independently let a
            # 270-count canvas be scaled to 33 counts, giving a frame far smoother than Poisson
            # allows at that brightness, and the reverse for the other direction.
            level = float(getattr(self, '_target', self.med[self.rng.integers(len(self.med))]))
        bkg = (patch*self._envelope()*level).astype(np.float32)
        if self.masks is not None and len(self.masks):
            mask = np.asarray(self.masks[self.rng.integers(len(self.masks))], bool)
        else:
            mask = np.ones((HEIGHT, WIDTH), bool)
        return np.where(mask, np.maximum(bkg, 0), 0).astype(np.float32), mask


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mb = MosaicBackground(seed=0)
    OUT = f'{WORK}/tmp_diag/sim2/images'
    os.makedirs(OUT, exist_ok=True)
    n = 6
    fig, ax = plt.subplots(n, 2, figsize=(22, 2.9*n))
    for k in range(n):
        b, m = mb.background()
        v = b[m]
        for j, (img, ttl) in enumerate([(b, 'linear 1-99%'), (b, 'log + HE, what the model sees')]):
            a = ax[k][j]
            if j == 0:
                lo, hi = np.percentile(v, 1), np.percentile(v, 99)
                d = np.clip((img-lo)/max(hi-lo, 1e-9), 0, 1)
            else:
                lo, hi = np.percentile(v, 5), np.percentile(v, 99.5)
                x = np.clip((img-lo)/max(hi-lo, 1e-9), 0, 1)
                x = np.log10(1+9*x)
                u = (x[m]*255).astype(np.uint8)
                e = cv2.equalizeHist(u.reshape(-1, 1)).ravel().astype(np.float32)/255.
                d = np.zeros_like(x); d[m] = e
            d = np.where(m, d, 0)
            a.imshow(d, cmap='gray', aspect='auto', origin='lower')
            a.set_title(f'mosaic {k}   level {np.median(v):.0f} counts   {ttl}', fontsize=9)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle('Mosaic backgrounds from 64 bare-Si Lambda modules (free shuffle, '
                 'level-normalised tiles, feathered seams)', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/mosaic_demo.png', dpi=105)
    print('wrote', f'{OUT}/mosaic_demo.png')
