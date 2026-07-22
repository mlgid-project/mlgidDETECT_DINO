"""Physics-based training simulation from CIF structures (Track B; docs/PHYSICS_SIM_INVESTIGATION.md).

Renders training images whose PEAK CONFIGURATION comes from real crystallography -- structures,
orientations, and structure-factor intensities simulated offline from CIF files with pygidsim
(physics_sim/generate_bank.py -> bank.npz) -- instead of the standard sim's random q positions and
random intensities. Composition per image (user spec): 0-1 powder pattern + 1-2 oriented patterns.

ISOLATION: this file is a SIBLING of simulation.py and does not modify it. It maximizes appearance
consistency by REUSING the standard machinery unmodified:
  * an internal FastSimulation instance supplies the detector mask, dark-area configuration/masking
    (create_detector_mask / filter_dark_area / add_dark_area / apply_detector_gaps) and the peak
    renderer img_from_labels(boxes, intensities, is_ring);
  * the module-level appearance chain is applied in the SAME ORDER as FastSimulation.simulate_img:
    mul_perlin -> add_glass -> add_linear_background -> poisson -> stretch -> dark areas ->
    detector gaps -> salt&pepper -> log -> HE -> clip -> kernel -> digitalize -> normalize -> flip.
So a physics image differs from a standard image ONLY in where the peaks are and how bright they
are relative to each other -- the appearance statistics are identical by construction.

Geometry: bank peaks are (|q| A^-1, chi deg from the q_xy axis); per image a q_max is sampled from
the eval-geometry range and mapped x = q/q_max*WIDTH, y = chi/90*HEIGHT (both axes randomly
flipped downstream by flip_image, as in the standard sim).

Contract: simulate_img() returns (image (H,W) float32 [0,1] cuda, boxes xyxy pixel, mask bool,
is_ring bool) -- identical to FastSimulation.simulate_img. Consumed by SimulationDataset via the
use_physics_sim / physics_sim_fraction flags (training-only; eval/ONNX untouched).
"""
import random

import numpy as np
import torch

from simulation import (FastSimulation, HEIGHT, WIDTH, normalize, mul_perlin, add_glass,
                        add_linear_background, apply_poisson_noise, apply_salt_pepper_noise,
                        apply_stretch, apply_log, apply_he, apply_clip_img, apply_kernel,
                        digitalize_img, flip_image, clamp_boxes)

Q_MAX_RANGE = (2.5, 4.5)        # per-image q_max [A^-1]; eval q_max = sqrt(qz^2+qxy^2) ~ 2.5-4.5
N_POWDER = (0, 1)               # per image (user spec)
N_ORIENTED = (1, 2)
RINGS_PER_POWDER = (3, 15)      # cap rings (top by intensity)
SPOTS_PER_ORIENTED = (8, 60)    # cap spots (top by intensity)
GAMMA_RANGE = (0.3, 0.6)        # intensity compression I' = (I/I_max)^gamma
INTENSITY_RANGE = (2.0, 50.0)   # final range fed to img_from_labels (matches SimulationConfig)
ENTRY_SCALE_RANGE = (0.08, 1.0) # per-entry overall scale (minor/major phases; trains faint phases)
RADIAL_W_RANGE = (1.0, 5.0)     # radial half-width px (matches width_central)
CHI_W_SPOT_RANGE = (2.0, 40.0)  # chi half-width px for oriented spots (mosaic spread)


class PhysicsSimulation(object):
    def __init__(self, bank_path, device='cuda'):
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
            raise ValueError(f"bank {bank_path} has no oriented entries")
        self.device = device
        self.fast = FastSimulation(device=device)   # reused, never modified

    def _entry(self, idx):
        s, c = int(self.entry_start[idx]), int(self.entry_count[idx])
        return self.q[s:s + c], self.chi[s:s + c], self.intensity[s:s + c]

    def _compress(self, inten, gamma):
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
        raise RuntimeError("PhysicsSimulation: 20 consecutive draws produced no renderable "
                           "peaks -- check the bank (q coverage) and Q_MAX_RANGE.")

    @torch.no_grad()
    def _attempt(self):
        dev = self.device
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
            x = (qs[top] / q_max * WIDTH)
            w = torch.empty(k).uniform_(*RADIAL_W_RANGE)
            b = torch.stack([x - w, torch.zeros(k), x + w, torch.full((k,), float(HEIGHT))], -1)
            scale = random.uniform(*ENTRY_SCALE_RANGE)
            boxes_l.append(b); inten_l.append(self._compress(ii[top], gamma) * scale)
            ring_l.append(torch.ones(k, dtype=torch.bool))

        # ---- oriented entries (spots/arcs) ----
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
            w = torch.empty(k).uniform_(*RADIAL_W_RANGE)
            aw = torch.empty(k).uniform_(*CHI_W_SPOT_RANGE)
            b = torch.stack([x - w, y - aw, x + w, y + aw], -1)
            scale = random.uniform(*ENTRY_SCALE_RANGE)
            boxes_l.append(b); inten_l.append(self._compress(ii[top], gamma) * scale)
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
        # spots falling inside a detector gap are dropped, as in simulate_img (rings are kept:
        # a ring survives its gap crossing). Index by mask, not by concat, so peak ORDER is
        # irrelevant here.
        idx_gap = torch.ones(len(boxes), dtype=torch.bool, device=dev)
        seg = ~is_ring
        if bool(seg.any()):
            idx_gap[seg] = f.filter_peaks_detector_gap(boxes[seg])
        # filter_dark_area applies the >1.6 size filter itself and clamps boxes to the dark-area
        # limits; it also SETS the dark-area state used by add_dark_area below.
        pos = (boxes[:, 0] + boxes[:, 2]) / 2
        boxes, idx_dark = f.filter_dark_area(pos, boxes)
        keep = idx_gap & idx_dark
        boxes, intensities, is_ring = boxes[keep], intensities[keep], is_ring[keep]
        if len(boxes) == 0:
            return None

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

        img = apply_log(img)
        img = apply_he(img)
        img = apply_clip_img(img)
        img = apply_kernel(img, f.kernel1)
        img = digitalize_img(img)
        img = normalize(img)

        img, boxes, mask = flip_image(img, boxes, mask)
        return img, boxes, mask, is_ring
