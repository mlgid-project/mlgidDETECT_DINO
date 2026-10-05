"""More diverse real-like backgrounds, built only from the 90 clean donor frames.

WHY. Every realbkg run decays with training while every synthetic-background run improves
(conv3: organic 0.594 at epochs 60-100 -> 0.496 at plateau, train loss falling 37% the whole way).
Randomisation was checked and is fine -- 160/160 distinct frames over four epochs, zero repeats.
The cause is background reuse: a 48-slot pool of finished backgrounds, each seen ~21 times per
epoch, all drawn from 90 donors.

WHAT IS ACTUALLY MEMORISABLE. `MosaicBackground` runs with `q_locked=False` and divides every
tile by its own blur, so the canvas is a FLAT, STATIONARY fluctuation field with no q information
in it. All of the q and chi structure is in `_envelope()`, applied after the crop, and that is one
of ~90 fixed smooth shapes. So what repeats across a run is not the texture -- it is the global
configuration (envelope shape, exposure level, mask). That is a small set, and it is what a
network can latch onto.

Measured stage costs (diagnostics/mosaic_cost.py), against ~300 ms for one simulated frame:
    canvas_image()  143 ms     _envelope()  183 ms     mask draw  18 ms
    random crop     0 ms       noise map     36 ms     full entry 469 ms
The crop is free and was being cached; the envelope is the single most expensive stage and was
being recomputed per entry. Both backwards.

THIS MODULE
  A  cache the flat canvas, take a FRESH CROP per frame, with an optional chi flip. Free.
     Costs nothing and gives every frame its own noise realisation.
  B2 sample the envelope from a PCA fitted to the donors' own smooth shapes, instead of picking
     one of 90. Clipped to the per-pixel range the donors actually span, so a sampled envelope
     stays inside the real family rather than extrapolating out of it. This is the axis that
     re-cropping does NOT touch, and on the analysis above it is the one that matters most.
  B1 a random-phase spectral surrogate of the texture, for COMPARISON ONLY. It reproduces the
     measured two-point spatial statistics exactly and never repeats, but it discards
     higher-order structure -- hot pixels, line defects, panel seams -- which is the structure we
     want a detector to learn to ignore. Rendered so the realism can be judged by eye before any
     decision to blend it in.

Nothing here uses the legacy simulator or any synthetic noise model: every pixel of A and B2
traces back to a real donor frame, and B1 is built from a real donor's own power spectrum.
"""
import numpy as np
import cv2

HEIGHT, WIDTH = 512, 1024
ENV_H, ENV_W = 64, 128          # envelopes are sigma-64 smooth, so this loses nothing
ENV_SIGMA = 64.0


def _orient_low_q_bright(g):
    """Real GIWAXS backgrounds are bright at low q. Mirror in q if this donor is the other way."""
    if g[:, :WIDTH//4].mean() < g[:, -WIDTH//4:].mean():
        return g[:, ::-1]
    return g


class BackgroundV2:
    def __init__(self, mb, canvas=(1536, 3072), n_pc=6, seed=None):
        self.mb = mb
        self.rng = np.random.default_rng(seed)
        self.canvas_shape = tuple(canvas)
        self._fit_envelopes(n_pc)
        self._canvas = None
        self._level = None

    # ------------------------------------------------------------------ B2
    def _fit_envelopes(self, n_pc):
        """PCA over the donors' smooth shapes, in log space so a sample stays positive."""
        E = []
        for f in self.mb.frames:
            g = np.nan_to_num(np.asarray(f, np.float32), nan=0.0)
            g = cv2.resize(g, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            g = _orient_low_q_bright(g)
            e = cv2.GaussianBlur(g, (0, 0), ENV_SIGMA)
            m = float(np.median(e[e > 0])) if (e > 0).any() else 1.0
            e = np.maximum(e/max(m, 1e-6), 1e-3)
            E.append(cv2.resize(e, (ENV_W, ENV_H), interpolation=cv2.INTER_AREA))
        E = np.log(np.maximum(np.stack(E).reshape(len(E), -1), 1e-6))
        self._E = E                      # keep them: donor_envelope() needs no recomputation
        self.n_donors = len(E)
        self._mu = E.mean(0)
        X = E - self._mu
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = int(min(n_pc, Vt.shape[0]))
        self._V = Vt[:k]
        self._sd = (U[:, :k]*S[:k]).std(0)
        self._lo, self._hi = E.min(0), E.max(0)
        var = (S**2)/max(float((S**2).sum()), 1e-12)
        self.env_var_explained = float(var[:k].sum())

    def sample_envelope(self):
        """A new smooth shape from inside the donor family. ~2 ms."""
        c = self.rng.normal(0.0, self._sd)
        v = np.clip(self._mu + c @ self._V, self._lo, self._hi)
        e = np.exp(v).reshape(ENV_H, ENV_W)
        e = cv2.resize(e, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        e = cv2.GaussianBlur(e, (0, 0), 8.0)        # remove any upsampling seam
        return (np.maximum(e, 1e-3)/max(float(np.median(e)), 1e-6)).astype(np.float32)

    def donor_envelope(self, i=None):
        """The CURRENT behaviour, for comparison: one donor's own shape, unaltered."""
        i = int(self.rng.integers(self.n_donors)) if i is None else int(i)
        e = np.exp(self._E[i]).reshape(ENV_H, ENV_W)
        e = cv2.resize(e, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        return (np.maximum(e, 1e-3)/max(float(np.median(e)), 1e-6)).astype(np.float32)

    # ------------------------------------------------------------------ A
    def new_canvas(self, pool=None):
        """Assemble one big flat fluctuation canvas and remember the level its tiles were cut at.

        `pool` restricts the tiles to those donor indices, which is how a caller matches the
        exposure class -- the level has to come from the tiles, not be imposed afterwards, or the
        frame gets a graininess its brightness does not allow.
        """
        old = self.mb.canvas
        self.mb.canvas = self.canvas_shape
        try:
            self._canvas = self.mb.canvas_image(pool=pool)
        finally:
            self.mb.canvas = old
        self._level = float(getattr(self.mb, '_target',
                                    self.mb.med[self.rng.integers(len(self.mb.med))]))
        return self._canvas

    def crop(self, flip=None):
        """A fresh 512x1024 view of the cached canvas. Free."""
        if self._canvas is None:
            self.new_canvas()
        H, W = self._canvas.shape
        r = int(self.rng.integers(0, max(H-HEIGHT, 1)))
        c = int(self.rng.integers(0, max(W-WIDTH, 1)))
        p = self._canvas[r:r+HEIGHT, c:c+WIDTH]
        if p.shape != (HEIGHT, WIDTH):
            p = cv2.resize(p, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        f = bool(self.rng.integers(2)) if flip is None else bool(flip)
        if f:
            p = p[::-1]                      # chi mirror only -- q must not move
        return np.ascontiguousarray(p, np.float32)

    def frame(self, mask=None, envelope='pca'):
        """One background in counts: fresh crop x sampled envelope x the tiles' own level."""
        p = self.crop()
        env = self.sample_envelope() if envelope == 'pca' else self.donor_envelope()
        bkg = (p*env*self._level).astype(np.float32)
        if mask is None:
            mask = np.ones((HEIGHT, WIDTH), bool)
        mask = np.asarray(mask, bool)
        return np.where(mask, np.maximum(bkg, 0), 0).astype(np.float32), mask

    # ------------------------------------------------------------------ B1
    def surrogate(self, mask=None, envelope='pca'):
        """Random-phase surrogate of the texture: same power spectrum, new phases. Comparison only."""
        p = self.crop(flip=False)
        x = p - float(p.mean())
        F = np.fft.rfft2(x)
        ph = self.rng.uniform(0.0, 2.0*np.pi, F.shape)
        y = np.fft.irfft2(np.abs(F)*np.exp(1j*ph), s=x.shape)
        s = float(x.std())/max(float(y.std()), 1e-12)
        t = (1.0 + y*s).astype(np.float32)
        env = self.sample_envelope() if envelope == 'pca' else self.donor_envelope()
        bkg = (np.maximum(t, 0)*env*self._level).astype(np.float32)
        if mask is None:
            mask = np.ones((HEIGHT, WIDTH), bool)
        mask = np.asarray(mask, bool)
        return np.where(mask, np.maximum(bkg, 0), 0).astype(np.float32), mask
