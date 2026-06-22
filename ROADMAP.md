# Roadmap — diagnostics, ideas & opportunities (mlgidDETECT_DINO)

Forward-looking companion to `MODIFICATIONS.md` (which is the change log). Captures where the model
stands, *why* it's stuck, and the candidate directions with honest assessments. Updated 2026-06-07.

## Current state
- Best model: 2-class ring/segment DINO (swin_L 48×6), run `ringseg_2class_20260603-142434`,
  organic ap_total ~0.55 / 41 ~0.76 by ep360. Pre/post matched to mlgidDETECT; class-aware NMS.
- Tried & rejected: #3 Path A (realistic simulation masking) — no AP gain (masking gap wasn't the
  bottleneck). #3 Path B (fine-tune on real) — not viable (all labeled real data = 41 + organic is
  used for eval; no held-out set).
- AP has plateaued at ~0.55 (organic) across the 91-class baseline, 2-class, and Path A.

## Diagnostic C — where the AP is actually lost
Run on the best checkpoint vs `organic` (8 imgs, 817 GT peaks, q-matcher, operating point score>0.3):
overall **recall 0.49, precision 0.81**. Breakdown (`diagnostics/diagnose_C.py`, `diagnose_C.png`):

| axis | finding |
|---|---|
| **visibility** | recall high(3)=0.66, med(2)=0.45, **low(1)=0.28** — most misses are faint peaks |
| **type** | ring=0.79 (n=24) vs **segment=0.48 (n=793)** — segments are the bulk and the hard part |
| **q-position** | recall ~0.55 for q<682 but **0.34 for q>682** — misses concentrate at high q |
| **false positives** | ~12 FP/image, **~48% high-confidence (>0.5)**, 97% predicted "segment" |

**Interpretation:** this is a *representation/sensitivity* ceiling, not a preprocessing/masking one
(consistent with Path A failing). The model misses faint, high-q segments and simultaneously emits
confident spurious segments. Both symptoms point at backbone feature quality and the lack of any
physical/contextual prior — exactly what the two ideas below target. Caveat: low-visibility peaks
(vis=1) may be partly an irreducible label-difficulty ceiling; and the eval set is small (8 frames),
so treat magnitudes as indicative.

## Idea A — self-supervised backbone on real + simulated GIWAXS (then freeze)
Pretrain the swin backbone with SSL on a large pool of **unlabeled** GIWAXS images (real +
simulated), freeze (or low-LR fine-tune), and reuse for all detection runs.
- **Why it fits the diagnosis:** the current backbone is trained from scratch on synthetic only (no
  ImageNet, no SSL). A backbone that has seen lots of real diffraction texture/noise/ring structure
  should be markedly more sensitive to faint/high-q peaks and less prone to confident background FPs
  — i.e. it attacks the representation ceiling directly. This is the principled way to inject
  real-domain knowledge **without labeled real data** (no detection-label leakage).
- **Approach:** Masked Autoencoder (MAE) is a natural fit (GIWAXS already has large masked
  regions); alternatives: DINO/iBOT. Must use the same swin_L(48×6) arch so weights transfer to the
  detector backbone. Then load as `pretrain_model_path` (the repo already supports loading a
  pretrained backbone + `finetune_ignore`).
- **Pure-simulated SSL is redundant:** the backbone already learns the simulated distribution via the
  supervised detection loss, and the detector converges fine from scratch, so SSL-on-sim adds no new
  info and isn't needed as an init. **The entire value of SSL is the real data**; sim is mixed in only
  for volume / sim↔real bridging.
- **Data requirement (real, unlabeled):** SSL is data-hungry. Rough: min useful ~2k–5k *diverse* real
  frames (mixed with sim); comfortable ~10k–50k; strong 100k+. Diversity (many samples/materials/
  geometries) matters far more than raw frame count (1 in-situ scan of 10k near-identical frames is
  low value). The current **labeled** real set is only ~49 frames (8 organic + 41) — far too few, and
  it's the eval set; SSL needs a *separate* unlabeled archive (beamline raw archives usually have this
  even when labels are scarce).
- **48×6 window blocks ImageNet warm-start:** the elongated window is incompatible with ImageNet
  swin-large weights (same relative-position mismatch we hit loading checkpoints), so SSL must run
  **from scratch** → pushes the data requirement to the higher end. Middle option: switching the
  detector to a *standard* window would unlock ImageNet warm-start + much cheaper domain-adaptive SSL
  (separate architecture decision; 48×6 was chosen for the elongated polar aspect ratio).
- Using eval frames' pixels for SSL is a gray area (no label leak, but backbone "sees" eval pixels);
  cleanest is to exclude eval frames. "Frozen" is efficient/reusable but may cap fine-tuning (try
  frozen first, then low-LR unfreeze). Highest setup cost of any option, but highest ceiling-raise —
  **only if the unlabeled real corpus exists.**

## ⚠️ KEY FINDING (2026-06-07) — the eval is likely label-limited, and physics wins don't apply
Before implementing any physics post-processor, validated the hypotheses cheaply on the best
checkpoint (organic). Results (`diagnostics/` — diagnose_rings, sweep_nms, viz_fp (+PNGs)):
- **Symmetry TTA: not applicable** — the polar image is a single 0–90° quadrant, so the qxy↔−qxy
  mirror symmetry is out of frame.
- **Ring-aware FP rejection: rejected** — the FPs are *on* rings, not off them. I(q) percentile at
  the detection q: TP 0.61 vs FP 0.67 (FPs skew *more* on-ring); **~93% of FPs are within ~8px-q of a
  real GT peak**. An I(q)/ring gate would reject real-ring detections.
- **NMS tuning: no help** — sweeping segment-NMS IoU 0.4→0.1 leaves ap_total flat (0.554→0.552) and
  trades recall for the few FPs removed → the FPs are *not* overlapping duplicates.
- The FPs are confident detections on real rings at *unlabeled angles*. **Expert review (2026-06-07)
  confirmed the labels are complete/authoritative → these are GENUINE HALLUCINATIONS**, not missed
  labels. So precision (0.81) is real: the model confidently fires "segment" boxes on real ring /
  diffuse intensity where there is no actual peak. `viz_fp.png` = GT(green)/TP(blue)/FP(red).

**Implications for strategy:** the model has two real failure modes, and **neither is fixable by
physics/NMS/preprocessing** (all tested above):
1. **Precision:** confident hallucinations on real ring/diffuse structure (model can't tell a real
   localized peak from generic ring intensity).
2. **Recall:** misses faint / high-q peaks.
Both are classic **synthetic→real *content* gaps** — the model trained on synthetic doesn't truly
know what real peaks vs real rings/background/diffuse scattering look like. That's why every
synthetic-only change (91→2 class, Path A masking) plateaued at ~0.55. The principled fix is **real
data in the representation** → SSL backbone on real+sim (Idea A, data-gated). Speculative
no-real-data alternatives: simulation *content* realism (ring/diffuse/background appearance — distinct
from Path A masking, but uncertain given Path A precedent) or precision-oriented/hard-negative
training (e.g. enrich "ring-without-peak" negatives so the model stops firing on ring intensity).
Physics post-processing remains **shelved**.

## Idea B — physics-informed model (SHELVED — see KEY FINDING above; validated as not applicable)
Inject GIWAXS physics so predictions are constrained to physically plausible configurations —
targets both the high-confidence-FP (precision) and faint/high-q-miss (recall) problems.
- **Mirror symmetry (qxy=0 meridian):** GIWAXS patterns are symmetric; real peaks come in mirror
  pairs. Cheap win: **symmetry TTA** (run on image + its mirror, fuse by consensus) — recovers
  missed mirror peaks and rejects asymmetric FPs. (mlgidDETECT already has flip-TTA to extend.)
- **Debye–Scherrer rings (constant q):** segments often lie on rings. Detect ring q-values (e.g.
  peaks in the angle-integrated 1-D I(q) profile, or a Hough/Radon in q) and use them to (a) propose
  missed peaks along a detected ring → recall, (b) reject FPs far from any ring → precision
  (directly hits the ~12 FP/img, mostly segments).
- **q-awareness / form-factor decay:** add an explicit q-coordinate input channel (or broadcast the
  1-D I(q) profile) so the model can expect faint high-q peaks (the high-q recall cliff). Cheap.
- **Lattice / q-series consistency:** crystalline peaks fall on lattice-consistent q-series; enforcing
  this (auxiliary loss or post-hoc indexing) is the most powerful but also the hardest (needs
  indexing) — defer.
- **Assessment:** the symmetry-TTA, ring-aware post-processing, and q-channel are relatively cheap,
  test-the-hypothesis-fast wins (especially for precision). Full lattice indexing is research-grade.

## Other levers (from the diagnostic)
- **Precision/threshold:** ~12 FP/img with many high-confidence — a higher score threshold trades
  recall for precision; ring-aware FP rejection (Idea B) is the smarter fix.
- **Segment-focused training:** segments dominate and are the weak class; could weight the loss or
  enrich faint/high-q segments in the simulation.
- **Low-visibility ceiling:** vis=1 recall 0.28 may be partly irreducible — worth separating
  "hard-but-possible" from "genuinely ambiguous" before chasing it.

## Suggested sequencing (for discussion — nothing started)
1. **Idea A (SSL backbone)** is the big lever if an unlabeled real corpus exists — gates on data.
2. In parallel / if data-limited: **cheap physics wins** (symmetry TTA, ring-aware FP rejection,
   q-channel) — fast to try, directly target the measured FP + high-q problems.
3. Defer lattice indexing and any further simulation-realism work (Path A showed diminishing returns).

## Key open question
What **unlabeled real GIWAXS data** is available beyond 41 + organic (for Idea A SSL pretraining)?
That determines whether A is the lead or we start with the cheap physics wins.
