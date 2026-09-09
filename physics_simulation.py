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
N_POWDER = (0, 1)               # entries composed into one image
N_ORIENTED = (1, 2)
RINGS_PER_POWDER = (3, 15)      # cap rings (top by intensity)
SPOTS_PER_ORIENTED = (8, 60)    # cap spots (top by intensity)
GAMMA_RANGE = (0.3, 0.6)        # intensity compression I' = (I/I_max)^gamma
INTENSITY_RANGE = (2.0, 50.0)   # final range fed to img_from_labels (matches SimulationConfig)
ENTRY_SCALE_RANGE = (0.08, 1.0)  # per-entry scale: minor/major phases, trains faint-phase recall


class PhysicsSimulation(object):
    def __init__(self, bank_path, sim_config=None, device='cuda', unify_contrast=False):
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
        self.sim_config = sim_config or SimulationConfig()
        # the SAME config object, so box_coef_override reaches img_from_labels' sigma recovery
        self.fast = FastSimulation(sim_config=self.sim_config, device=device)

    def _entry(self, idx):
        s, c = int(self.entry_start[idx]), int(self.entry_count[idx])
        return self.q[s:s + c], self.chi[s:s + c], self.intensity[s:s + c]

    def _compress(self, inten, gamma):
        """Map structure-factor intensities into the renderer's range, PRESERVING their ordering
        and relative contrast. gamma < 1 compresses the many-decade physical range into something
        the log+HE chain can still show, which is the whole point of using physics intensities."""
        rel = (inten / inten.max().clamp(min=1e-12)) ** gamma
        lo, hi = INTENSITY_RANGE
        return lo + rel * (hi - lo)

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
        gamma = random.uniform(*GAMMA_RANGE)
        boxes_l, inten_l, ring_l = [], [], []

        # ---- powder entries (rings) ----
        for _ in range(random.randint(*N_POWDER)):
            qs, _, ii = self._entry(int(np.random.choice(self.powder_ids)))
            vis = qs < q_max * 0.99
            if int(vis.sum()) < 1:
                continue
            qs, ii = qs[vis], ii[vis]
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
            inten_l.append(self._compress(ii[top], gamma) * scale)
            ring_l.append(torch.ones(k, dtype=torch.bool))

        # ---- oriented entries (spots / arcs) ----
        for _ in range(random.randint(*N_ORIENTED)):
            qs, cs, ii = self._entry(int(np.random.choice(self.oriented_ids)))
            vis = qs < q_max * 0.99
            if int(vis.sum()) < 1:
                continue
            qs, cs, ii = qs[vis], cs[vis], ii[vis]
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
            inten_l.append(self._compress(ii[top], gamma) * scale)
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
        if img.min() == img.max() or torch.any(img.isnan()):
            return None

        img = mul_perlin(img)
        img = add_glass(img, f.x, f.y)
        img = add_linear_background(img)
        img = apply_poisson_noise(img, f.sim_config.poisson_range)
        if f.polar_dark_area:
            img = apply_stretch(img, (int(.04 * WIDTH), int(.1 * WIDTH)), (7, 10))
        if img.min() == img.max():
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

        img, boxes, mask = flip_image(img, boxes, mask)
        return img, boxes, mask, is_ring
