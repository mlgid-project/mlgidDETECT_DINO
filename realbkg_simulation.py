"""Simulator that puts pygidSIM peaks on REAL, feature-stripped GIWAXS backgrounds.

Built in giwaxs_sim.ipynb; this module is the training-side port, with the same numbers.

WHY IT EXISTS. Every previous simulator modelled the whole frame: a background field, Poisson,
read noise, a correlated-noise term, each piece fitted to a real statistic. The frames still did
not look real, because a real GIWAXS frame carries things a small parameter set does not produce --
detector panel seams, the low-q halo, radial streaks, beam-stop geometry, gain structure, and the
particular way intensity falls off with q for that sample and that beamline. So this one stops
modelling the background and uses a real one, and takes from pygidSIM only what pygidSIM knows:
raw peak POSITIONS and raw INTENSITIES.

    real frame (linear, reciprocal) -> polar -> every diffraction feature removed  -> DONOR
    pygidSIM (q, chi, I)            -> widths/shape/brightness from real 2-D fits  -> PEAKS
    DONOR + counting-noise(PEAKS)   -> clip 5/99.5 -> log10 -> HE                  -> model input
    PEAKS only                      -> visibility gate -> boxes                    -> ground truth

GROUND TRUTH IS SOUND because the donor's own peaks are removed before use: every feature the
detector is asked to find is one we put there. The visibility gate then drops simulated peaks too
faint to see, so the labels match what a human would mark.

PEAK PARAMETERS ARE MEASURED, not guessed -- mlgidFIT's own 2-D Gaussian fits of 1,926 real peaks
in 57 real frames (`fitted_peaks/parameters_peak`). Two results shaped this:
  * real peaks are ARCS, not blobs: median sigma_q 3.9 px vs sigma_chi 20.3 px, and the two are
    uncorrelated per peak (r = 0.00);
  * real peaks are FAINT: median amplitude 3.1x the local noise, p10 1.5x.
Widths also vary WITHIN a frame -- sd(log sigma_q) = 0.53, sd(log sigma_chi) = 0.83.

LEAKAGE. Donors come from ekaterina_aftermlgidFIT. Max azimuthal-I(q) cosine against any
organic_labeled.h5 frame is 0.888, below the 0.949 same-material baseline and far below the 0.999
same-frame value. No evaluation frame is in the donor pool.

KNOWN GAPS, both open and both measured (see the notebook's section 6):
  * background lag-1 autocorrelation 0.11 against real organic's 0.31;
  * selected donors still score leftover-z ~15 against a detection threshold of 3, so a few frames
    carry a faint unremoved real feature with no box on it -- an unlabelled positive.
"""
import math
import os
import random

import cv2
import numpy as np
import torch

from simulation import HEIGHT, WIDTH, SimulationConfig
from util.exp_preprocess import apply_contrast

CHAIN = {'clip': (5, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}


class RealBkgSimulation:
    """Drop-in for FastSimulation/PhysicsSimulation: `simulate_img()` -> (img, boxes, mask, is_ring)."""

    def __init__(self, bank_path, donor_path, stats_path, sim_config=None, device='cuda',
                 n_oriented=(1, 3), p_ring=0.15, n_powder=(1, 1),
                 contrast_min=1.5, snr_min=6.0, ring_iou_max=0.10,
                 unified_labels=False, seg_iou_max=None, max_peaks=None,
                 spots_cap=None, rings_cap=None,
                 voigt_eta=(0.6, 1.0), voigt_cut=3.0, donor_keep_frac=0.40,
                 elongate=None, max_donors=None,
                 mosaic=False, mosaic_pool=48, mosaic_refresh=64, mosaic_seed=None,
                 intensity_decades=None, amplitude_mode='fitted',
                 mask_bank=True, mask_keep='default'):
        from physics_simulation import PhysicsSimulation, RINGS_PER_POWDER, SPOTS_PER_ORIENTED
        self.device = device
        self.sim_config = sim_config or SimulationConfig()
        self.w_coef = float(self.sim_config.w_coef)
        self.a_coef = float(self.sim_config.a_coef)
        self.n_oriented, self.p_ring, self.n_powder = n_oriented, p_ring, n_powder
        #: reflections drawn per ORIENTED entry, and rings per POWDER entry. These were module
        #: constants in physics_simulation.py with no way to reach them from a config, so the
        #: whole peak-count axis was unreachable from a training run: the 2026-09-21 label
        #: review ran at (2, 200) while training was pinned at (8, 60), and the dynamic-range
        #: result depends on which one is used.
        self.spots_cap = tuple(spots_cap) if spots_cap else SPOTS_PER_ORIENTED
        self.rings_cap = tuple(rings_cap) if rings_cap else RINGS_PER_POWDER
        self.contrast_min, self.snr_min, self.ring_iou_max = contrast_min, snr_min, ring_iou_max
        #: RENDER IFF LABELLED. Off, the frame carries three tiers -- labelled, rendered but
        #: unlabelled, and not rendered -- so a visible peak can sit in the image with no box.
        #: Real labelled frames have no such tier (organic_labeled.h5 has no unlabelled peaks),
        #: so that tier teaches the model to suppress peaks that look exactly like real ones.
        #: On, one decision governs both: a peak is drawn if and only if it gets a box.
        self.unified_labels = bool(unified_labels)
        #: IoU ceiling for SEGMENT-SEGMENT pairs. None keeps the historical behaviour, which is
        #: no segment suppression at all. Rings have always had one (ring_iou_max).
        self.seg_iou_max = seg_iou_max
        #: ceiling on reflections drawn per frame, across all entries, before any gate
        self.max_peaks = max_peaks
        self.voigt_eta, self.voigt_cut = voigt_eta, voigt_cut
        self.elongate = elongate or dict(p_frame=0.30, p_peak=0.30, factor=(2.0, 5.0),
                                         p_chi=0.8, conserve_flux=True)
        self.sig_clip = ((0.7, 25.0), (2.0, 160.0))

        self.phys = PhysicsSimulation(bank_path, sim_config=self.sim_config, device=device,
                                      unify_contrast=True)
        S = np.load(stats_path)
        self.R = {k: (float(S[k]) if S[k].ndim == 0 else S[k]) for k in S.files}
        self.mosaic_refresh = int(mosaic_refresh)
        self.intensity_decades = (None if intensity_decades is None
                                  else (float(intensity_decades[0]), float(intensity_decades[1])))
        self.last_gain = 1.0
        self.next_intensity_target = None   # set to force one frame's peak scale, else drawn
        self.amplitude_mode = str(amplitude_mode)
        self.mask_bank, self.mask_keep = bool(mask_bank), mask_keep
        self._frames_made = 0
        if mosaic:
            self._load_mosaic_donors(mosaic_pool, mosaic_seed)
        else:
            self._load_donors(donor_path, donor_keep_frac, max_donors)

    # ------------------------------------------------------------------ mosaic
    def _mosaic_entry(self, pool=None):
        """One mosaic background plus the per-donor quantities `simulate_img` needs.

        `noise` is the local fluctuation amplitude measured on the background itself, and `coef`
        turns that into a sqrt(I) law so peak photons get the same graininess as the pixels they
        land on. Both are measured on THIS background, exactly as for a real donor -- the mosaic is
        made of real detector pixels, so its noise is real detector noise and nothing is assumed.
        """
        # The mask is drawn FIRST and carries the geometry: a real converted detector mask AND a
        # freshly generated missing wedge at a random incidence angle. Its q_max is then the
        # frame's q_max -- the mask's gaps sit where they do because of that geometry, so a
        # borrowed q axis would put them at the wrong q relative to the peaks.
        if self._masks is not None:
            m, md = self._masks.draw()
            qm = float(md['q_max'])
        else:
            m, md, qm = None, {}, None
        b, m = self._mb.background(pool=pool, mask=m)
        nz = self._noise_map(b.astype(np.float64), m)
        cf = nz/np.sqrt(np.maximum(cv2.GaussianBlur(b, (0, 0), 16.0), 1e-6))
        if qm is None:
            qm = float(self._qsrc[np.random.randint(len(self._qsrc))]) if len(self._qsrc) else 4.45
        return b.astype(np.float32), m, nz.astype(np.float32), cf.astype(np.float32), qm

    def _load_mosaic_donors(self, n_pool, seed):
        """Build a FRESH pool of mosaic backgrounds for this run.

        Nothing is read from a pre-built bank. Every run re-cuts the 90 clean donor frames into
        new tiles and new canvases, so no two runs -- and, with `mosaic_refresh`, no two parts of
        the same run -- train on the same background pixels in the same arrangement.
        """
        from realbkg_sim.mosaic_background import MosaicBackground
        self._mb = MosaicBackground(seed=seed)
        self._masks = None
        if self.mask_bank:
            from realbkg_sim.detector_masks import MaskBank
            self._masks = MaskBank(seed=seed, keep=self.mask_keep)
        qp = os.path.join(os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389'),
                          'datasets/realbkg_donors_mm/qmax.npy')
        self._qsrc = np.load(qp) if os.path.exists(qp) else np.array([], np.float32)
        e = [self._mosaic_entry() for _ in range(int(n_pool))]
        self.bkg = np.stack([x[0] for x in e])
        self.mask = np.stack([x[1] for x in e])
        self.noise = np.stack([x[2] for x in e])
        self.coef = np.stack([x[3] for x in e])
        self.qmax = np.asarray([x[4] for x in e], np.float32)
        self.meta = [dict(source='mosaic') for _ in e]
        print(f"[realbkg] mosaic mode: {len(self.bkg)} fresh backgrounds from "
              f"{len(self._mb.frames)} clean donor frames, one slot replaced every "
              f"{self.mosaic_refresh} frames", flush=True)

    def _refresh_mosaic_slot(self, d):
        """Replace one pool slot with a newly assembled mosaic.

        A fixed pool of N backgrounds would be reused 500k/N times even though the tiles behind it
        are unlimited. Rebuilding one slot every `mosaic_refresh` frames costs ~1 s spread over
        that many frames (~15 ms/frame against ~300 ms of simulation) and means the pool is fully
        turned over every N*refresh frames, thousands of times across a full run.
        """
        b, m, nz, cf, qm = self._mosaic_entry()
        self.bkg[d], self.mask[d], self.noise[d], self.coef[d], self.qmax[d] = b, m, nz, cf, qm

    # ------------------------------------------------------------------ donors
    def _load_donors(self, path, keep_frac, max_donors):
        import json
        import h5py
        with h5py.File(path, 'r') as f:
            bkg = f['background'][()].astype(np.float32)
            msk = f['mask'][()].astype(bool)
            meta = [json.loads(s) for s in f['meta'][()].astype(str)]
        noise = np.stack([self._noise_map(bkg[i].astype(np.float64), msk[i])
                          for i in range(len(bkg))])
        bn = np.array([np.median(bkg[i][msk[i]])/max(np.median(noise[i][msk[i]]), 1e-9)
                       for i in range(len(bkg))])
        ac = np.array([self._ac1(bkg[i].astype(np.float64), msk[i]) for i in range(len(bkg))])
        left = np.array([m['leftover'] for m in meta])
        remv = np.array([m['removed'] for m in meta])
        # Ranked, not thresholded. The terms fight each other on purpose: `leftover` is
        # amplitude-over-noise, so a WHITE background hides its own residual features and scores
        # well -- selecting on it alone gave a pool with background/noise 1.4 against real's 4.1
        # and lag-1 autocorrelation 0.07 against 0.31, i.e. visibly sandblasted frames. Hard cuts
        # also collapsed the pool to two source files.
        score = (np.log10(np.maximum(left, 1))
                 + np.abs(np.log(np.maximum(bn, 1e-3)/4.1))
                 + 1.5*np.abs(np.maximum(ac, 1e-3) - 0.35)/0.35
                 + 2.0*np.maximum(remv - 0.35, 0))
        byfile = {}
        for i, m in enumerate(meta):
            byfile.setdefault(m['path'], []).append(i)
        per = max(1, int(round(keep_frac*len(left)/max(len(byfile), 1))))
        sel = sorted(j for idx in byfile.values()
                     for j in sorted(idx, key=lambda i: score[i])[:per])
        if max_donors:
            sel = sel[:max_donors]
        sel = np.array(sel)
        self.bkg, self.mask, self.noise = bkg[sel], msk[sel], noise[sel]
        self.coef = np.stack([self.noise[i]/np.sqrt(np.maximum(
            cv2.GaussianBlur(self.bkg[i], (0, 0), 16.0), 1e-6)) for i in range(len(sel))])
        self.qmax = np.array([meta[i]['qmax'] for i in sel], dtype=np.float32)
        self.meta = [meta[i] for i in sel]
        print(f"[realbkg] {len(sel)} donors of {len(left)} from "
              f"{len({self.meta[i]['path'] for i in range(len(sel))})} source files | "
              f"leftover p50 {np.median(left[sel]):.1f} | bkg/noise p50 {np.median(bn[sel]):.2f} "
              f"(real 4.1) | autocorr p50 {np.median(ac[sel]):.3f} (real 0.31)", flush=True)

    @staticmethod
    def _noise_map(bkg, mask, sig=16.0, T=64):
        num = cv2.GaussianBlur((bkg*mask).astype(np.float32), (0, 0), sig)
        den = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sig)
        res = np.where(mask, bkg - num/np.maximum(den, 1e-6), 0.0)
        out = np.empty_like(res, dtype=np.float32)
        for r in range(0, HEIGHT, T):
            for c in range(0, WIDTH, T):
                blk = res[r:r+T, c:c+T][mask[r:r+T, c:c+T]]
                out[r:r+T, c:c+T] = (1.4826*np.median(np.abs(blk-np.median(blk)))
                                     if blk.size > 48 else np.nan)
        v = out[np.isfinite(out)]
        out[~np.isfinite(out)] = np.median(v) if v.size else 1.0
        return np.maximum(cv2.GaussianBlur(out, (0, 0), T/2), 1e-6)

    @staticmethod
    def _ac1(b, m, T=48):
        out = []
        for r in range(0, HEIGHT-T, 2*T):
            for c in range(0, WIDTH-T, 2*T):
                if not m[r:r+T, c:c+T].all():
                    continue
                t = b[r:r+T, c:c+T]
                d = t - t.mean(); v = (d*d).mean()
                if v > 0:
                    out.append((d[:, :-1]*d[:, 1:]).mean()/v)
        return float(np.median(out)) if out else 0.0

    # ------------------------------------------------------------------ peaks
    @staticmethod
    def _pick(I, k, n_bright=5, rng=np.random):
        """k reflections drawn at RANDOM from the entry, not the k brightest.

        Taking the top k collapses the frame's intensity range: measured over 3,000 oriented
        entries, the brightest 8 span 0.53 decades at the median and the brightest 60 span 1.15,
        while the entry's stored list spans 1.61 and real labelled peaks span 2.7 (organic) to 3.6
        (41). Sampling across the whole list instead lets one frame carry both ends of the
        structure's own distribution, which is the point -- the ratios stay exactly the physics',
        only which subset is shown changes.

        One of the `n_bright` brightest is always forced in, so no frame ends up made entirely of
        faint reflections and every frame has a proper bright peak to anchor it.
        """
        n = len(I)
        k = min(k, n)
        idx = rng.choice(n, size=k, replace=False)
        bright = np.argsort(-I)[:min(n_bright, n)]
        if not np.intersect1d(idx, bright).size:
            idx[rng.randint(k)] = bright[rng.randint(len(bright))]
        return idx

    def _draw_peaks(self, qmax):
        xs, ys, ii, rg = [], [], [], []
        for _ in range(random.randint(*self.n_oriented)):
            q, chi, I = self.phys._entry(int(np.random.choice(self.phys.oriented_ids)))
            q, chi, I = (np.asarray(v.cpu() if torch.is_tensor(v) else v, float)
                         for v in (q, chi, I))
            v = q < qmax*0.995
            if v.sum() < 1:
                continue
            q, chi, I = q[v], chi[v], I[v]
            k = min(len(q), random.randint(*self.spots_cap))
            top = self._pick(I, k)
            k = len(top)
            xs.append(q[top]/qmax*WIDTH); ys.append(chi[top]/90.0*HEIGHT)
            ii.append(I[top]);            rg.append(np.zeros(k, bool))
        if random.random() < self.p_ring:
            for _ in range(random.randint(*self.n_powder)):
                q, _c, I = self.phys._entry(int(np.random.choice(self.phys.powder_ids)))
                q, I = (np.asarray(v.cpu() if torch.is_tensor(v) else v, float) for v in (q, I))
                v = q < qmax*0.995
                if v.sum() < 1:
                    continue
                q, I = q[v], I[v]
                k = min(len(q), random.randint(*self.rings_cap))
                top = self._pick(I, k)
                k = len(top)
                xs.append(q[top]/qmax*WIDTH); ys.append(np.full(k, HEIGHT/2.0))
                ii.append(I[top]);            rg.append(np.ones(k, bool))
        if not xs:
            return None
        x, y = np.concatenate(xs), np.concatenate(ys)
        I, r = np.maximum(np.concatenate(ii), 1e-12), np.concatenate(rg)
        if self.max_peaks is not None and len(x) > self.max_peaks:
            # Subsample UNIFORMLY, not by intensity: keeping the brightest would bias the frame
            # toward its own top end and undo what _pick's random draw is for. Real labelled
            # frames top out at 168 boxes (organic) and 65 (41), so a few hundred is already
            # past anything measured.
            k = np.random.choice(len(x), self.max_peaks, replace=False)
            x, y, I, r = x[k], y[k], I[k], r[k]
        return x, y, I/I.max(), r

    def _draw_widths(self, n, is_ring):
        R = self.R
        ln = lambda mu, sd, k=None: np.exp(np.random.normal(mu, sd, k))
        s_q = ln(R['log_sq_mu'], R['log_sq_sd'])*ln(0.0, R['w_sq'], n)
        s_c = ln(R['log_sc_mu'], R['log_sc_sd'])*ln(0.0, R['w_sc'], n)
        el = dict(on=False, k=1.0, f=None)
        E = self.elongate
        if random.random() < E['p_frame']:
            el['on'] = True
            el['k'] = random.uniform(*E['factor'])
            sel = np.random.random(n) < E['p_peak']
            f = np.where(sel, el['k']*ln(0.0, 0.1, n), 1.0)
            if random.random() < E['p_chi']:
                s_c = s_c*f
            else:
                s_q = s_q*f
            el['f'] = f
        s_q = np.clip(s_q, *self.sig_clip[0])
        s_c = np.clip(s_c, *self.sig_clip[1])
        return s_q, np.where(is_ring, 1e4, s_c), el

    def _assign_amplitudes(self, rel, noise_at, n_label):
        """pygidSIM's ORDERING, the real distribution's VALUES.

        The brightest `n_label/keep_frac` peaks are rank-matched onto the lognormal fitted to the
        real amp/local-noise distribution, so the labelled set reproduces the real spread AND the
        real count once the gate has cut the part below CONTRAST_MIN. Everything fainter is placed
        under the gate, where pygidSIM's long tail of weak reflections belongs: rendered, because
        those photons are really there, but unlabelled, because nobody could mark them.
        """
        from scipy.special import ndtr, ndtri
        R = self.R
        n = len(rel)
        order = np.argsort(-rel)
        an = np.empty(n)
        keep_frac = max(1.0 - ndtr((np.log(self.contrast_min) - R['log_an_mu'])/R['log_an_sd']), .05)
        k = int(min(n, round(n_label/keep_frac)))
        u = 1.0 - (np.arange(k) + 0.5)/k
        an[order[:k]] = np.exp(R['log_an_mu'] + R['log_an_sd']*ndtri(np.clip(u, 1e-6, 1-1e-6)))
        if n > k:
            an[order[k:]] = self.contrast_min*np.exp(-np.abs(np.random.normal(0, 0.8, n-k)))
        return an*noise_at

    #: render each peak only within this many sigma of its centre. The taper is
    #: exp(-(u^2/uc^2)^2) with uc^2 = voigt_cut^2 * 2ln2, so at u = 2*voigt_cut = 6 the Lorentzian
    #: term is 1/(1+26) = 0.037 and the taper is exp(-8.3) = 2.4e-4: together under 1e-5 of the
    #: peak amplitude. Truncating there is invisible and turns a full-frame evaluation into a
    #: window roughly 45x smaller for a typical peak (sigma_q 4, sigma_chi 20).
    PATCH_SIGMA = 2.0

    def _render(self, x, y, s_q, s_c, amp, eta):
        """pseudo-Voigt: Gaussian core, Lorentzian wings, both normalised to the SAME FWHM so the
        mixing weight cannot move the peak's visible width -- which is what keeps the box
        convention exact for any eta. A pure Gaussian stops abruptly; real peaks at 2 HWHM are
        4-5x brighter than a Gaussian predicts. Wings tapered past `voigt_cut` half-widths, because
        a 2-D Lorentzian falls only as 1/u^2 and dozens of untapered tails sum into a pedestal.

        Evaluated per peak over a local window rather than the whole frame -- see PATCH_SIGMA.
        Full-frame evaluation cost 4.16 s per simulated frame, which at 8 dataloader workers is
        1.9 images/s, exactly what the detector consumes and therefore no margin at all.
        """
        dev = self.device
        img = torch.zeros(HEIGHT, WIDTH, device=dev, dtype=torch.float32)
        uc2 = (float(self.voigt_cut)**2)*2*math.log(2)
        ln2 = 2*math.log(2)
        R = float(self.PATCH_SIGMA)*float(self.voigt_cut)
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
        s_q = np.asarray(s_q, np.float64); s_c = np.asarray(s_c, np.float64)
        amp = np.asarray(amp, np.float64)
        for i in range(len(x)):
            rx, ry = R*s_q[i], R*s_c[i]
            c0 = max(int(math.floor(x[i] - rx)), 0); c1 = min(int(math.ceil(x[i] + rx)) + 1, WIDTH)
            r0 = max(int(math.floor(y[i] - ry)), 0); r1 = min(int(math.ceil(y[i] + ry)) + 1, HEIGHT)
            if c1 <= c0 or r1 <= r0:
                continue
            X = torch.arange(c0, c1, device=dev, dtype=torch.float32).view(1, -1)
            Y = torch.arange(r0, r1, device=dev, dtype=torch.float32).view(-1, 1)
            u2 = ((X - float(x[i]))/float(s_q[i]))**2 + ((Y - float(y[i]))/float(s_c[i]))**2
            prof = eta/(1.0 + u2/ln2) + (1-eta)*torch.exp(-u2/2)
            img[r0:r1, c0:c1] += float(amp[i])*prof*torch.exp(-(u2/uc2)**2)
        return img.cpu().numpy().astype(np.float64)

    def _visibility(self, amp, noise_at, s_q, s_c, is_ring, mask, x):
        """per-pixel contrast and matched-filter SNR, both from the peak's OWN amplitude, so a
        bright neighbour cannot lend a faint peak its signal"""
        contrast = amp/np.maximum(noise_at, 1e-9)
        col = np.clip(np.round(x).astype(int), 0, WIDTH-1)
        L = mask[:, col].sum(0).astype(float)
        n_eff = np.where(is_ring, L*s_q*np.sqrt(np.pi), np.pi*s_q*np.minimum(s_c, HEIGHT))
        return contrast, contrast*np.sqrt(np.clip(n_eff, 0, None))

    def _nms(self, bx, sel, amp, iou_max):
        """Greedy suppression by amplitude within the subset `sel`, at `iou_max`.

        Measured on the real labelled sets, this is what boxes a human actually leaves
        overlapping. organic: 27,734 pairs, 11 overlap at all, worst 0.154. 41: 10,705 segment
        pairs, 19 overlap, worst 0.400, and 2,220 ring pairs of which exactly ONE overlaps
        (at 0.989, which looks like a duplicated annotation). So real boxes essentially do not
        overlap, and rings never do.
        """
        keep = np.ones(len(bx), bool)
        if iou_max is None:
            return keep
        idx = np.flatnonzero(sel)
        if len(idx) < 2 or iou_max >= 1.0:
            return keep
        b = bx[idx]
        x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
        ar = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
        iou = inter/np.maximum(ar[:, None]+ar[None, :]-inter, 1e-9)
        alive = np.ones(len(idx), bool)
        for n in np.argsort(-amp[idx]):
            if not alive[n]:
                continue
            hit = (iou[n] > iou_max) & alive; hit[n] = False
            alive[hit] = False
        keep[idx[~alive]] = False
        return keep

    def _ring_nms(self, bx, rg, amp):
        return self._nms(bx, rg, amp, self.ring_iou_max)

    def _compose(self, bkg, peaks, mask, coef):
        """Apply the per-frame intensity scale, add counting noise to the PEAK photons, and lay
        the peaks on the donor background. Factored out so the unified-label path and the
        historical path cannot drift apart."""
        if self.amplitude_mode != 'pygid':
            self.last_gain = 1.0
        tgt = self.next_intensity_target
        if self.amplitude_mode != 'pygid' and (tgt is not None
                                               or self.intensity_decades is not None):
            pmax = float(peaks[mask].max()) if mask.any() else 0.0
            if pmax > 0:
                R = float(tgt) if tgt is not None else 10.0**random.uniform(*self.intensity_decades)
                self.last_gain = R/pmax
                peaks = peaks*self.last_gain
        # Counting noise on the PEAK photons only: the donor already carries its own noise, and
        # adding it again would double-count what the real frame already has. c = n/sqrt(B) is
        # measured on THIS donor, so peaks are exactly as grainy as the background they sit on.
        peaks = peaks + coef*np.sqrt(np.maximum(peaks, 0))*np.random.standard_normal(peaks.shape)
        return np.where(mask, np.maximum(bkg + peaks, 0.0), 0.0)

    # ------------------------------------------------------------------ frame
    def simulate_img(self):
        d = np.random.randint(len(self.bkg))
        if getattr(self, '_mb', None) is not None and self.mosaic_refresh > 0:
            self._frames_made += 1
            if self._frames_made % self.mosaic_refresh == 0:
                self._refresh_mosaic_slot(d)
        bkg = self.bkg[d].astype(np.float64); mask = self.mask[d]
        noise = self.noise[d].astype(np.float64); coef = self.coef[d].astype(np.float64)
        qmax = float(self.qmax[d])

        p = self._draw_peaks(qmax)
        if p is None:
            return None
        x, y, rel, rg = p
        iy = np.clip(np.round(y).astype(int), 0, HEIGHT-1)
        ix = np.clip(np.round(x).astype(int), 0, WIDTH-1)
        inside = mask[iy, ix]
        if inside.sum() < 3:
            return None
        x, y, rel, rg, iy, ix = x[inside], y[inside], rel[inside], rg[inside], iy[inside], ix[inside]

        s_q, s_c, el = self._draw_widths(len(x), rg)

        if self.amplitude_mode == 'pygid':
            # PYGIDSIM INTENSITIES, UNTOUCHED. `rel` is I/I_max straight out of the structure
            # factors; every peak in the frame is multiplied by ONE scale, so the relative
            # intensity distribution the physics produced is exactly what gets rendered. This is
            # the whole point of the mode: the 'fitted' path below reduces `rel` to a RANK and
            # redraws the values from a lognormal fitted to real labelled peaks, which reproduces
            # the real amplitude-over-noise distribution but discards the physics' own ratios.
            #
            # Elongation's flux division is skipped here for the same reason -- dividing an
            # elongated peak's amplitude by its stretch factor would make amplitude no longer
            # proportional to `rel`.
            #
            # NOTE the scale is applied to AMPLITUDE, i.e. peak height. If pygidSIM's intensities
            # are integrated intensities rather than peak heights, a wide peak and a narrow peak
            # with the same I should NOT get the same height; that conversion is not done here.
            tgt = self.next_intensity_target
            R = (float(tgt) if tgt is not None
                 else 10.0**random.uniform(*(self.intensity_decades or (3.0, 6.0))))
            amp = np.asarray(rel, np.float64)*R
            self.last_gain = R
        else:
            n_label = int(np.random.choice(self.R['npf']))
            amp = self._assign_amplitudes(rel, noise[iy, ix], n_label)
            if el['on'] and self.elongate['conserve_flux'] and el['f'] is not None:
                amp = amp/el['f']          # an arc is the same reflection over a longer footprint

        eta = random.uniform(*self.voigt_eta)
        if self.unified_labels:
            # ONE DECISION for drawing and labelling. The gate runs BEFORE the render, so the
            # peaks that fail it are never painted into the image at all -- there is no tier of
            # visible-but-unlabelled structure for the model to learn to ignore.
            #
            # Note this also fixes the suppression path: previously _ring_nms removed a ring's
            # BOX while the ring stayed in the image. Here suppression removes the peak.
            con, snr = self._visibility(amp, noise[iy, ix], s_q, s_c, rg, mask, x)
            keep = (con >= self.contrast_min) & (snr >= self.snr_min)
            hw, hh = self.w_coef*s_q/2.0, self.a_coef*s_c/2.0
            bx = np.stack([x-hw, y-hh, x+hw, y+hh], 1).astype(np.float32)
            bx[rg, 1] = 0.0; bx[rg, 3] = float(HEIGHT)
            # Clip BEFORE the overlap test, not after. Two boxes that overhang the same frame
            # edge become more alike once clipped, so suppressing on unclipped geometry lets
            # pairs through above the threshold -- measured, 6 pairs up to 0.479 under a 0.40
            # cap. Clipping first makes the threshold mean what it says.
            bx[:, 0::2] = np.clip(bx[:, 0::2], 0, WIDTH-1)
            bx[:, 1::2] = np.clip(bx[:, 1::2], 0, HEIGHT-1)
            keep &= self._nms(bx, rg & keep, amp, self.ring_iou_max)
            keep &= self._nms(bx, (~rg) & keep, amp, self.seg_iou_max)
            if keep.sum() < 3:
                return None
            x, y, s_q, s_c, amp, rg = (v[keep] for v in (x, y, s_q, s_c, amp, rg))
            bx = bx[keep]
            peaks = self._render(x, y, s_q, s_c, amp, eta)
            total = self._compose(bkg, peaks, mask, coef)
            ok = (bx[:, 0] < bx[:, 2]) & (bx[:, 1] < bx[:, 3])
            bx, rgk = bx[ok], rg[ok]
            if len(bx) == 0:
                return None
            img = apply_contrast(total, mask, CHAIN)
            if not np.isfinite(img).all():
                return None
            dev = self.device
            return (torch.as_tensor(img, dtype=torch.float32, device=dev),
                    torch.as_tensor(bx, dtype=torch.float32, device=dev),
                    torch.as_tensor(mask, device=dev),
                    torch.as_tensor(rgk, device=dev))

        # Skip peaks that cannot be seen at all. `_visibility`'s contrast is amp over the local
        # noise, so amp < 0.2*noise puts the peak's BRIGHTEST pixel a fifth of a sigma above the
        # background -- nothing a render would show and nothing the gate would ever keep. Peaks
        # merely below the labelling threshold are still drawn, because those are real faint
        # structure the frame should contain; only the invisible ones are dropped.
        vis = amp >= 0.2*np.maximum(noise[iy, ix], 1e-9)
        peaks = self._render(x[vis], y[vis], s_q[vis], s_c[vis], amp[vis], eta)

        # PEAK INTENSITY SCALE. pygidSIM returns NORMALISED structure-factor intensities, which
        # only become "counts" once multiplied by some chosen range; real pyGID frames carry
        # maxima from 1e3 to 1e6. The scale is applied to the RENDERED PEAKS ONLY -- the donor
        # background keeps the real counts it was measured at, untouched.
        #
        # Applied BEFORE the peak counting noise below, so a peak scaled to 1e6 gets the grain a
        # 1e6-count measurement would have rather than its old grain multiplied up.
        #
        # NOTE what this does to the labels. `contrast` and the matched-filter SNR in the gate are
        # computed from the PRE-scale amplitude against the background's own noise, so the boxes
        # are exactly the ones the calibrated gate chooses and do not move. But every peak,
        # including the sub-threshold ones the gate deliberately leaves unlabelled, is lifted by
        # the same factor relative to the background -- so at a large scale a faint unlabelled
        # peak can look obvious while carrying no box. That is acceptable for LOOKING at frames
        # and is not acceptable for training; see the dynamic-range work before any run.
        total = self._compose(bkg, peaks, mask, coef)

        con, snr = self._visibility(amp, noise[iy, ix], s_q, s_c, rg, mask, x)
        keep = (con >= self.contrast_min) & (snr >= self.snr_min)
        hw, hh = self.w_coef*s_q/2.0, self.a_coef*s_c/2.0
        bx = np.stack([x-hw, y-hh, x+hw, y+hh], 1).astype(np.float32)
        bx[rg, 1] = 0.0; bx[rg, 3] = float(HEIGHT)
        keep &= self._ring_nms(bx, rg & keep, amp)
        bx, rgk = bx[keep], rg[keep]
        bx[:, 0::2] = np.clip(bx[:, 0::2], 0, WIDTH-1)
        bx[:, 1::2] = np.clip(bx[:, 1::2], 0, HEIGHT-1)
        ok = (bx[:, 0] < bx[:, 2]) & (bx[:, 1] < bx[:, 3])
        bx, rgk = bx[ok], rgk[ok]
        if len(bx) == 0:
            return None

        img = apply_contrast(total, mask, CHAIN)
        if not np.isfinite(img).all():
            return None
        dev = self.device
        return (torch.as_tensor(img, dtype=torch.float32, device=dev),
                torch.as_tensor(bx, dtype=torch.float32, device=dev),
                torch.as_tensor(mask, device=dev),
                torch.as_tensor(rgk, device=dev))
