"""Phase AA.2 — how often does training actually show the model a PEAK SITTING ON A RING?

Hypothesis (user, 2026-08-25): the simulator rarely generates proper peaks-on-rings, so the model
never learns what one looks like, and at inference it guesses that a bright stretch of ring is a peak.
That would explain the dominant failure mode on the precision side — **59% of ssl1's false positives
sit on a ring** (49 of 83, phase AA), and **74% of the HIGH-confidence ones** (label_completeness) —
which phase AA showed is mostly NOT the close-pair weakness (only ~24% of FPs are both on-ring and
within 10 px in χ).

The code says the rate is low. `simulation.py`, `add_peaks_on_rings`, first line:

    if random.random() > .1:
        return None, None, None, None, None      # declines on 90% of frames

and when it does fire, three more filters cut it down: `randint(0, 4)` peaks per ring so a quarter of
eligible rings get none; `no_peaks = max_a_width < 100` excludes every ring whose angular half-extent
is under 100 px (in a 512 px polar image, every arc under ~200 px of χ); and the whole body sits in a
bare `try/except` that silently returns None on any error.

But a rate read off the code is not a dose, and "the model therefore cannot discriminate" is an
inference. This measures the dose on both sides with the SAME criterion.

MEASURED, simulated (the deployed ssl1 training config, `use_peak_clusters=False`, and again with it
on for completeness):
  - fraction of frames containing at least one peak-on-ring
  - fraction of all labelled SEGMENTS that are peaks-on-rings
  - rings per frame
MEASURED, real organic labels: the same three.

TWO CRITERIA, deliberately, so neither is taken on trust:
  - TAGGED: the boxes `add_peaks_on_rings` actually returned, matched into the final label list by
    exact centre (`_boxes_from_positions` centres a box on `pos`/`a_pos`, so the match is exact).
    Simulated only — the ground truth of what the generator intended.
  - GEOMETRIC: a segment whose centre falls inside a labelled RING's box. Applies to BOTH sides, so
    the sim/real comparison is like-for-like. On the simulated side it also cross-checks the tag: a
    geometric count far above the tagged count means segments land on rings by coincidence too, which
    is itself worth knowing.

READING. If simulated peaks-on-rings are an order of magnitude rarer than real ones, that is a
distribution hole sitting directly under the dominant failure mode — and unlike anything in the
close-pair line (V-AA, nine mechanisms refuted, no lever worth a run), the knob is one character:
`random.random() > .1`.

No model, no training. ~5 min. See tmp_diag/run_onring.sbatch.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

import simulation as S
from simulation import FastSimulation
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset

DSET = "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"
N_SIM = 300
TOL = 0.5          # px; centre match against what add_peaks_on_rings returned (exact in principle)


def inside_ring(seg_c, ring_boxes):
    """GEOMETRIC criterion, applied identically to simulated and real labels."""
    if not len(ring_boxes) or not len(seg_c):
        return np.zeros(len(seg_c), bool)
    q, c = seg_c[:, 0], seg_c[:, 1]
    hit = np.zeros(len(seg_c), bool)
    for b in ring_boxes:
        hit |= (q >= b[0]) & (q <= b[2]) & (c >= b[1]) & (c <= b[3])
    return hit


def simulate(dev, clusters, n):
    sim = FastSimulation(device=dev)
    sim.sim_config.use_peak_clusters = bool(clusters)
    _orig = FastSimulation.add_peaks_on_rings
    rec = {}

    def wrapped(self, x_position, widths, boxes, ring_intensities):
        out = _orig(self, x_position, widths, boxes, ring_intensities)
        if out[0] is None:
            rec['pk'] = np.zeros((0, 2))
        else:
            rec['pk'] = np.stack([out[0].detach().cpu().numpy(),
                                  out[2].detach().cpu().numpy()], 1)   # (pos, a_pos) centres
        return out
    FastSimulation.add_peaks_on_rings = wrapped

    fired = tagged = geo = n_seg = n_ring = n_frames = 0
    fr_with_tag = fr_with_geo = 0
    for k in range(n):
        _sd = 90000 + k
        random.seed(_sd); torch.manual_seed(_sd); np.random.seed(_sd)
        rec['pk'] = np.zeros((0, 2))
        try:
            img, bx, mask, isr = sim.simulate_img()
        except Exception:
            continue
        b = bx.detach().cpu().numpy()
        r = isr.detach().cpu().numpy().astype(bool)
        if not len(b):
            continue
        n_frames += 1
        cen = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        seg_m = ~r
        n_seg += int(seg_m.sum()); n_ring += int(r.sum())
        pk = rec['pk']
        if len(pk):
            fired += 1
        t = np.zeros(len(b), bool)
        if len(pk) and seg_m.any():
            d = np.abs(cen[:, None, :] - pk[None, :, :]).max(-1)
            t = (d < TOL).any(1) & seg_m
        g = inside_ring(cen[seg_m], b[r])
        tagged += int(t.sum()); geo += int(g.sum())
        fr_with_tag += int(t.any()); fr_with_geo += int(g.any())

    FastSimulation.add_peaks_on_rings = _orig
    return dict(frames=n_frames, seg=n_seg, ring=n_ring,
                fired_frac=fired / max(n_frames, 1),
                tagged=tagged, tagged_frac_of_seg=tagged / max(n_seg, 1),
                frames_with_tagged=fr_with_tag / max(n_frames, 1),
                geo=geo, geo_frac_of_seg=geo / max(n_seg, 1),
                frames_with_geo=fr_with_geo / max(n_frames, 1),
                rings_per_frame=n_ring / max(n_frames, 1),
                segs_per_frame=n_seg / max(n_frames, 1))


def real():
    config = Config()
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.INPUT_DATASET = DSET
    ds = PyGIDDataset(config, path=DSET, preprocess_func=standard_preprocessing,
                      buffer_size=5, load_labels=True)
    n_frames = n_seg = n_ring = geo = fr_with_geo = 0
    for gc in ds.iter_images():
        b = np.array(gc.polar_labels.boxes, dtype=np.float64)
        if not len(b):
            continue
        r = np.array(list(gc.polar_labels.is_ring), dtype=bool)
        n_frames += 1
        cen = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        seg_m = ~r
        n_seg += int(seg_m.sum()); n_ring += int(r.sum())
        g = inside_ring(cen[seg_m], b[r])
        geo += int(g.sum()); fr_with_geo += int(g.any())
    if hasattr(ds, 'close'):
        ds.close()
    return dict(frames=n_frames, seg=n_seg, ring=n_ring, geo=geo,
                geo_frac_of_seg=geo / max(n_seg, 1),
                frames_with_geo=fr_with_geo / max(n_frames, 1),
                rings_per_frame=n_ring / max(n_frames, 1),
                segs_per_frame=n_seg / max(n_frames, 1))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  N_SIM={N_SIM}", flush=True)
    out = {}
    for tag, cl in (('sim, clusters OFF (ssl1 config)', False), ('sim, clusters ON', True)):
        out[tag] = simulate(dev, cl, N_SIM)
        print(f"  {tag}: done", flush=True)
    out['REAL organic labels'] = real()

    print("\n" + "=" * 98)
    print("  How often does a PEAK SITTING ON A RING appear?")
    print("=" * 98)
    print(f"  {'':<34s}{'frames':>8s}{'segs/fr':>9s}{'rings/fr':>9s}"
          f"{'on-ring segs':>14s}{'% of segs':>11s}{'% of frames':>12s}")
    for k, v in out.items():
        print(f"  {k:<34s}{v['frames']:8d}{v['segs_per_frame']:9.1f}{v['rings_per_frame']:9.1f}"
              f"{v['geo']:14d}{100 * v['geo_frac_of_seg']:11.2f}{100 * v['frames_with_geo']:12.1f}")
    print("\n  (geometric criterion: a labelled segment whose centre falls inside a labelled ring box;"
          "\n   identical on both sides, so the comparison is like-for-like)")

    print("\n  simulated only — what add_peaks_on_rings actually produced (TAGGED):")
    for k, v in out.items():
        if 'tagged' not in v:
            continue
        print(f"    {k:<32s} fired on {100 * v['fired_frac']:5.1f}% of frames;  "
              f"{v['tagged']} tagged peaks = {100 * v['tagged_frac_of_seg']:.2f}% of segments;  "
              f"present in {100 * v['frames_with_tagged']:.1f}% of frames")

    rs = out['REAL organic labels']['geo_frac_of_seg']
    ss = out['sim, clusters OFF (ssl1 config)']['geo_frac_of_seg']
    if ss > 0:
        print(f"\n  REAL / SIM ratio on the geometric criterion: {rs / ss:.1f}x")
    else:
        print(f"\n  SIM produces ZERO on-ring segments; real rate is {100 * rs:.2f}% of segments.")
    json.dump(out, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/onring_dose.json', 'w'),
              indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
