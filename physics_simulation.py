"""Physics-based training simulation from CIF structures (MODIFICATIONS.md section I).

Renders training images whose PEAK CONFIGURATION comes from real crystallography -- structures,
orientations and structure-factor intensities simulated offline from CIF files with pygidsim
(physics_sim/generate_bank.py -> bank npz) -- instead of the standard sim's random q positions and
random intensities. The point of the track is the INTENSITIES: the standard sim draws them
UNIFORMLY in a bounded range (gen_intensities, simulation.py:1161, ring (2,50) / segment (10,50)),
whereas real diffraction has a few strong reflections and a long weak tail spanning orders of
magnitude, correlated with q through the structure and form factors.

Ported from branch `development` (git show b8f220b:physics_simulation.py) with four changes:
  1. `sim_config` is threaded through, so the box_coef_override the run is training under
     actually reaches the internal FastSimulation. The original built FastSimulation() with
     default coefficients, which silently ignored the override.
  2. Boxes are built as centre +/- sigma*coef using the config's own width ranges, so the label
     convention and the box-size statistics match the base recipe instead of the ad-hoc
     half-widths the original sampled.
  3. Q_MAX_RANGE covers both eval detectors (organic 4.95, 41 3.82); the original (2.5, 4.5)
     sat below organic's real q_max.
  4. Optional `unify_contrast`: run the REAL contrast pipeline (clip -> log -> HE, mask-aware,
     simulation.contrast_like_real) instead of the sim's log -> HE -> clip chain.

ISOLATION: a SIBLING of simulation.py; the standard generator is untouched. Appearance is kept
consistent by REUSING the standard machinery -- an internal FastSimulation supplies the detector
mask, dark areas and img_from_labels, and the module-level chain runs in the same order as
FastSimulation.simulate_img. A physics image differs ONLY in where the peaks are and how bright
they are relative to each other.

Geometry: bank peaks are (|q| A^-1, chi deg from the q_xy axis); per image a q_max is sampled and
mapped x = q/q_max*WIDTH, y = chi/90*HEIGHT -- the same transform util/pygidloader.py:146 uses for
real labels.

Contract: simulate_img() returns (image (H,W) float32 [0,1] cuda, boxes xyxy pixel, mask bool,
is_ring bool) -- identical to FastSimulation.simulate_img.
"""
import random

import numpy as np
import torch

from simulation import (FastSimulation, SimulationConfig, HEIGHT, WIDTH, normalize, mul_perlin,
                        add_glass, add_linear_background, apply_poisson_noise,
                        apply_salt_pepper_noise, apply_stretch, apply_log, apply_he,
                        apply_clip_img, apply_kernel, digitalize_img, flip_image, clamp_boxes,
                        contrast_like_real)

# per-image detector q_max [A^-1]. Real: organic 4.95 (every entry), 41 sqrt(2*1350^2)/500 = 3.82.
Q_MAX_RANGE = (3.5, 5.0)
N_POWDER = (0, 1)               # entries composed into one image -- OVERRIDABLE, see below
N_ORIENTED = (1, 2)
RINGS_PER_POWDER = (3, 15)      # cap rings (top by intensity)
SPOTS_PER_ORIENTED = (8, 60)    # cap spots (top by intensity)
# Per-image DYNAMIC RANGE in decades: the faintest RENDERED peak sits 10^-d below the brightest.
# Calibrated to the eval sets, not invented: measured per-pattern I/Imax of real labeled peaks puts
# p5 at 0.0019 for organic (2.71 decades) and 0.00024 for 41 (3.63 decades). Peaks fainter than the
# floor are DROPPED, not clamped -- real label files contain the peaks a human could see, so
# labeling a peak that is invisible in the rendered image would train the detector to hallucinate.
DYN_RANGE_DECADES = (2.4, 3.4)
INTENSITY_RANGE = (2.0, 50.0)   # only the UPPER bound is used; see _compress
ENTRY_SCALE_RANGE = (0.08, 1.0)  # per-entry scale: minor/major phases, trains faint-phase recall


def _usable(img):
    """True only if the image is FINITE and has contrast. Both halves are load-bearing, and a
    min/max test alone is not enough: `img.min() == img.max()` is False for an all-NaN image
    (NaN != NaN), so the old guard waved NaN straight through. See _attempt for how NaN gets in.
    """
    return bool(torch.isfinite(img).all()) and bool(img.min() != img.max())


class PhysicsSimulation(object):
    """`n_powder` overrides N_POWDER, the per-image count of powder (ring) entries.

    This is the ring:segment lever. Measured 2026-09-11 over 120 frames with the geometric ring
    criterion (box spans >= 70% of the valid chi rows at its radius), current box convention:

        real 41        16.90 rings/frame   24.0 segs   ring:seg 0.704
        real organic    3.50               98.6        ring:seg 0.035
        sim legacy     17.05               30.7        ring:seg 0.555
        sim physics     4.38               37.2        ring:seg 0.118   <- N_POWDER = (0, 1)

    At the default the physics sim lands on ORGANIC's composition and misses 41's, which is what
    dino_physics3_2 does on the gates: +0.030 organic over dino_lr4e5_1 (33/35 shared epochs) and
    -0.189 on 41 (34/35). Rings are the easy class (41 ring recall 0.856 vs 0.713 segments) and
    41% of 41's objects, so starving them costs 41 disproportionately. Widening the range -- NOT
    raising it to a compromise value -- lets single frames span both regimes.

    Default None keeps N_POWDER, so every run up to and including dino_physics3_2 is unchanged:
    `random.randint(*self.n_powder)` draws from the RNG identically for the same range.
    """

    def __init__(self, bank_path, sim_config=None, device='cuda', unify_contrast=False,
                 n_powder=None):
        d = np.load(bank_path, allow_pickle=False)
        self.q = torch.from_numpy(d['q']).float()
        self.chi = torch.from_numpy(d['chi']).float()
        self.intensity = torch.from_numpy(d['intensity']).float()
        self.entry_start = d['entry_start']
        self.entry_count = d['entry_count']
        kinds = d['entry_kind']
        self.powder_ids = np.nonzero(kinds == 'powder')[0]
        self.oriented_ids = np.nonzero(kinds == 'oriented')[0]
        if len(self.oriented_ids) == 0:
            raise ValueError(f'bank {bank_path} has no oriented entries')
        self.device = device
        self.unify_contrast = unify_contrast
        self.n_powder = N_POWDER if n_powder is None else (int(n_powder[0]), int(n_powder[1]))
        if self.n_powder[0] < 0 or self.n_powder[1] < self.n_powder[0]:
            raise ValueError(f'n_powder must be a non-negative (lo, hi) with hi >= lo, '
                             f'got {self.n_powder}')
        self.sim_config = sim_config or SimulationConfig()
        # the SAME config object, so box_coef_override reaches img_from_labels' sigma recovery
        self.fast = FastSimulation(sim_config=self.sim_config, device=device)

    def _entry(self, idx):
        s, c = int(self.entry_start[idx]), int(self.entry_count[idx])
        return self.q[s:s + c], self.chi[s:s + c], self.intensity[s:s + c]

    def _scale(self, inten, dyn):
        """Map structure-factor intensities into the renderer LINEARLY, preserving true relative
        contrast, and report which peaks clear the visibility floor.

        Returns (values, keep_mask). `dyn` is the per-image dynamic range (max/floor).

        WHY NOT COMPRESS. The previous version applied (I/I_max)^gamma with gamma 0.3-0.6 into a
        (2, 50) range. Two things made that destroy the point of this whole track:
          - the FLOOR. 2/50 caps the rendered dynamic range at 25x, the same cap the uniform
            standard sim has. Real labeled peaks span 2.7 (organic) to 3.6 (41) decades. No gamma
            can put a peak at I/Imax = 0.007 through a 25x mapping.
          - the GAMMA. At 0.45 a peak at 0.1% of max comes out at 5.6% -- weak peaks are pulled up
            hard. Measured on the bank, after compression: med I/Imax 0.286 and only 2.3% of peaks
            below 0.1, against the uniform sim's 0.363 / 5.7% and reality's ~0.008 / ~0.89. The
            compressed physics distribution was WORSE than the uniform sim on the weak-peak
            fraction, i.e. the intensity advantage was erased before the detector ever saw it.
        The many-decade range does need handling, but the contrast pipeline is what handles it --
        percentile clip -> log10 -> HE, the same chain applied to real frames (unify_contrast), and
        apply_log's own log in the legacy chain. Compressing first does that job twice.

        NOTE the absolute scale is irrelevant: add_glass/add_linear_background normalize() their
        input (simulation.py:331 says so), so only the RATIOS between peaks survive into the image.
        """
        rel = inten / inten.max().clamp(min=1e-12)
        keep = rel >= (1.0 / dyn)
        return rel * INTENSITY_RANGE[1], keep

    @torch.no_grad()
    def simulate_img(self):
        """Bounded retries: a draw can end up with no renderable peaks (all outside q_max, all
        clipped by the dark area, degenerate image). Recursion would risk a RecursionError deep
        into a multi-day run, so retry in a loop and fail loudly if the bank is unusable."""
        for _ in range(20):
            out = self._attempt()
            if out is not None:
                return out
        raise RuntimeError('PhysicsSimulation: 20 consecutive draws produced no renderable peaks '
                           '-- check the bank (q coverage) and Q_MAX_RANGE.')

    @torch.no_grad()
    def _attempt(self):
        dev = self.device
        sc = self.sim_config
        q_max = random.uniform(*Q_MAX_RANGE)
        dyn = 10.0 ** random.uniform(*DYN_RANGE_DECADES)
        boxes_l, inten_l, ring_l = [], [], []

        # ---- powder entries (rings) ----
        for _ in range(random.randint(*self.n_powder)):
            qs, _, ii = self._entry(int(np.random.choice(self.powder_ids)))
            vis = qs < q_max * 0.99
            if int(vis.sum()) < 1:
                continue
            qs, ii = qs[vis], ii[vis]
            vals, keep = self._scale(ii, dyn)          # drop peaks below the visibility floor
            if int(keep.sum()) < 1:
                continue
            qs, ii, vals = qs[keep], ii[keep], vals[keep]
            k = min(len(qs), random.randint(*RINGS_PER_POWDER))
            top = torch.argsort(ii, descending=True)[:k]
            x = qs[top] / q_max * WIDTH
            # box edge at +/- sigma*w_coef, exactly as _boxes_from_positions builds it, so
            # img_from_labels recovers the same sigma it would for a standard image
            sigma_q = torch.empty(k).uniform_(*sc.ring_width_central)
            w = sigma_q * sc.w_coef
            b = torch.stack([x - w, torch.zeros(k), x + w, torch.full((k,), float(HEIGHT))], -1)
            scale = random.uniform(*ENTRY_SCALE_RANGE)
            boxes_l.append(b)
            inten_l.append(vals[top] * scale)
            ring_l.append(torch.ones(k, dtype=torch.bool))

        # ---- oriented entries (spots / arcs) ----
        for _ in range(random.randint(*N_ORIENTED)):
            qs, cs, ii = self._entry(int(np.random.choice(self.oriented_ids)))
            vis = qs < q_max * 0.99
            if int(vis.sum()) < 1:
                continue
            qs, cs, ii = qs[vis], cs[vis], ii[vis]
            vals, keep = self._scale(ii, dyn)          # drop peaks below the visibility floor
            if int(keep.sum()) < 1:
                continue
            qs, cs, ii, vals = qs[keep], cs[keep], ii[keep], vals[keep]
            k = min(len(qs), random.randint(*SPOTS_PER_ORIENTED))
            top = torch.argsort(ii, descending=True)[:k]
            x = qs[top] / q_max * WIDTH
            y = cs[top] / 90.0 * HEIGHT
            sigma_q = torch.empty(k).uniform_(*sc.width_central)
            sigma_chi = torch.empty(k).uniform_(*sc.a_seg_widths_central)
            w, aw = sigma_q * sc.w_coef, sigma_chi * sc.a_coef
            b = torch.stack([x - w, y - aw, x + w, y + aw], -1)
            scale = random.uniform(*ENTRY_SCALE_RANGE)
            boxes_l.append(b)
            inten_l.append(vals[top] * scale)
            ring_l.append(torch.zeros(k, dtype=torch.bool))

        if not boxes_l:
            return None
        boxes = torch.cat(boxes_l).to(dev)
        intensities = torch.cat(inten_l).to(dev)
        is_ring = torch.cat(ring_l).to(dev)

        # ---- reuse the standard pipeline (mirrors FastSimulation.simulate_img order) ----
        f = self.fast
        f.background_img = None            # filter_dark_area reads it (quazipolar branch)
        f.detector_mask = False
        f.create_detector_mask()
        f.angle_limits.update_params()

        boxes = clamp_boxes(boxes)
        # spots inside a detector gap are dropped, as in simulate_img; rings are kept (a ring
        # survives its gap crossing). Index by mask so peak ORDER is irrelevant here.
        idx_gap = torch.ones(len(boxes), dtype=torch.bool, device=dev)
        seg = ~is_ring
        if bool(seg.any()):
            idx_gap[seg] = f.filter_peaks_detector_gap(boxes[seg])
        pos = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes, idx_dark = f.filter_dark_area(pos, boxes)
        keep = idx_gap & idx_dark
        boxes, intensities, is_ring = boxes[keep], intensities[keep], is_ring[keep]
        if len(boxes) == 0:
            return None

        # Drop inverted/degenerate boxes, then re-clamp -- EXACTLY as the standard sim does at
        # simulation.py:416-420. The polar/quazipolar dark-area clamp inside filter_dark_area can
        # push y2 below y1. The DINO matcher asserts x2>=x1 & y2>=y1 on target boxes
        # (util/box_ops.py:53), so an unfiltered inverted box crashes training on the first batch
        # that draws one -- this guard is why the original physics runs died on epoch 0.
        valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
        boxes, intensities, is_ring = boxes[valid], intensities[valid], is_ring[valid]
        if len(boxes) == 0:
            return None
        clamp_boxes(boxes)

        img = f.img_from_labels(boxes, intensities, is_ring)
        if not _usable(img):
            return None

        img = mul_perlin(img)
        img = add_glass(img, f.x, f.y)
        img = add_linear_background(img)
        img = apply_poisson_noise(img, f.sim_config.poisson_range)
        if f.polar_dark_area:
            img = apply_stretch(img, (int(.04 * WIDTH), int(.1 * WIDTH)), (7, 10))
        if not _usable(img):
            return None

        img, mask = f.add_dark_area(img, boxes)
        img, mask = f.apply_detector_gaps(img, mask)
        img = apply_salt_pepper_noise(img, f.sim_config.p_ps_noise)

        if self.unify_contrast:
            # the real pipeline's order and log argument, mask-aware (see contrast_like_real)
            img = contrast_like_real(img, mask)
            img = apply_kernel(img, f.kernel1)
            img = digitalize_img(img)
            img = normalize(img)
        else:
            img = apply_log(img)
            img = apply_he(img)
            img = apply_clip_img(img)
            img = apply_kernel(img, f.kernel1)
            img = digitalize_img(img)
            img = normalize(img)

        # THE contrast chain can produce NaN, and this is where the dino_phys3 crash came from
        # (jobs 2862172 and 2865413, both at util/box_ops.py:52 with boxes1 = the PREDICTIONS,
        # i.e. a NaN model output in an amp=False run, i.e. a NaN input image). Traced on the
        # offending draw (seed 42, draw 15,850 -- 1 image in ~20,000, which is why 41 epochs of
        # 1000 images ran first):
        #     apply_salt_pepper_noise   min +0  max +1     const=False
        #     contrast_like_real        min +0  max +0     const=True     <- every VALID pixel equal
        #     apply_kernel              min +0  max +0     const=True
        #     digitalize_img            min +0  max +0     const=True
        #     normalize                 nan=524288 (512x1024, i.e. all of them)
        # contrast_like_real returns all-zero when the valid region carries no signal at all --
        # here mask_valid was 0.459 and both peaks fell in the masked-out region, so the clip
        # quantiles coincide, log10 gives one constant, and the final where(m, img, 0) is zero
        # everywhere. normalize() is then (img - min) / (max - min) = 0/0. Retrying costs one
        # draw; letting it through costs the run.
        if not _usable(img):
            return None

        img, boxes, mask = flip_image(img, boxes, mask)
        return img, boxes, mask, is_ring
