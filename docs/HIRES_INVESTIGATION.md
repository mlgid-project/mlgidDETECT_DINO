# Higher input resolution (512×2048) — investigation

Status: **implemented, verified, launched — training verdict PENDING.**
Branch: `development`. Design record for the first Step-2 "representation sensitivity" lever after
the seven single-variable levers of the 2026-07 lever log (MODIFICATIONS.md phases I–P) all
returned null/negative and converged on one conclusion: *the ceiling is the learned
representation's sensitivity to faint and high-q peaks, not the loss / architecture / sim realism.*
The only lever that ever moved the gate was a better backbone (SSL). This lever attacks sensitivity
directly and physically.

## Motivation

The two capped failure modes are **faint** (vis=1 recall ~0.33) and **high-q** (outer-third recall
~0.44) peaks. High-q peaks are small and low-contrast and, at the deployed 512×1024 polar grid, are
near the sampling limit of the q-axis. Giving the detector more q-pixels means more samples across
each peak → more signal for the matched-filter-like response, most directly in the high-q band.

**Raw-data gate (the decisive pre-check).** Higher resolution only helps if the *source* data
carries finer detail than 512×1024 exposes. It does:
- organic (pygid): native `data/img_gid_q` is **1641×1641** (q_xy, q_z each 1641 samples).
- 41 (roi_data): native `image` is **1350×1350**.

Both are well finer than the 1024-wide radial grid, so a 2048-wide polar resample exposes **real**
additional detail, not interpolation. Verified: real frames resampled to 512×2048 are visibly
crisper and GT boxes land on peaks across the full q-range (tmp_diag/hires_real_montage.png).

## Design: 512×2048 (widen q only)

- **HEIGHT stays 512 (χ), WIDTH 1024 → 2048 (q).** Doubling only the q-axis (a) targets the high-q
  deficit where it lives, (b) costs ~2× activation memory (vs 2.25× isotropic 768×1536), and (c)
  leaves *every* HEIGHT-based formula in `simulation.py` untouched, shrinking the change surface.
- **From-scratch** (the trusted regime — see the from-scratch-preference note): SSL-pretrained
  backbone (`backbone_dir`, inherited from `DINO_4scale_swin_ssl.py`) + random detector head +
  ssl1's exact recipe (uniform 1e-5, no amp, 500 ep, lr-drop 280). The single variable vs ssl1 is
  the q-resolution. Matched control = **ssl1** (organic 0.586 / 41 0.762); the deployed ensemble
  (0.605 / 0.780) is the "did we beat production" reference.
- **The Swin-L 48×6 backbone is resolution-agnostic**: windowed attention + a window-sized
  relative-position-bias table (48×6, independent of input H/W) + sine positional encoding + no
  absolute PE. So the 512×1024-pretrained SSL backbone transfers to 512×2048 unchanged — only the
  input tensor is wider. Smoke-confirmed: `<All keys matched successfully>` at 512×2048.
- **This is a NEW model line, not a drop-in.** The exported ONNX input becomes `(1,1,512,2048)`.
  The 512×1024 deployment (ssl1+baseline ensemble) is untouched; a hires winner would deploy as its
  own model.

## Implementation (config-gated; byte-identical to every prior run at 1024)

Resolution is driven by one new config key `polar_shape = [H, W]`, absent in every other config
(→ defaults to [512,1024]), so the deployed inference path and all prior runs are byte-identical.

- **`config/DINO/DINO_4scale_swin_hires.py`** (new) — `_base_ = DINO_4scale_swin_ssl.py`;
  `polar_shape = [512, 2048]`.
- **`main.py`**
  - `SimulationDataset.__init__`: if `args.polar_shape` set, `simulation.HEIGHT/WIDTH` are set to it
    BEFORE constructing `FastSimulation` (which reads the module globals at build).
  - `evaluate_giwaxs_ap` (was a hardcoded `PREPROCESSING_POLAR_SHAPE=[512,1024]`): now
    `list(getattr(args,'polar_shape',[512,1024]))`, so real-image resampling + GT box conversion
    follow the training resolution.
- **`simulation.py`** — four edits, each **byte-identical at WIDTH=1024**:
  1. Removed the dead `global WIDTH` block (`simulate_img`) that force-reset WIDTH to 1024 on 50% of
     calls — it would silently clobber a configured 2048. Kept the 50% re-init quirk ssl1 trained
     with.
  2. Hardcoded background ring box `[116,0,128,512]` → `[116*WIDTH/1024, 0, 128*WIDTH/1024, HEIGHT]`
     (q-coords scale with WIDTH; χ = HEIGHT).
  3. Quazipolar image-mask boundary factor `(1-(WIDTH-512)/1024)` → `(512/WIDTH)`.
  4. Quazipolar **box-clamp** condition + assignment `(1-(WIDTH-512)/1024)` → `(512/WIDTH)`
     (`filter_dark_area`). **This was the label-corruption hazard**: at WIDTH=2048 the old factor is
     `1-1.5 = -0.5` (negative) → it would set box y-top negative → inverted boxes → the matcher
     `AssertionError` crash from the physics track (MODIFICATIONS.md phase P, fourth bug). `512/WIDTH`
     is the resolution-invariant form (= 0.5 at 1024, 0.25 at 2048), keeping `factor·q` constant in
     normalized-q so the physical quazipolar boundary is preserved. The box-clamp queries
     `angle_limits` at `pos = box_q·512/WIDTH ∈ [0,512)` at *both* resolutions, returning χ-limits in
     unchanged HEIGHT units → box clamping is resolution-invariant by construction.

Model, deformable transformer, positional encoding, postprocessing, NMS: **no change** (all read
shapes dynamically or scale with the config value — confirmed by the resolution-dependency audit).

## Verification (pre-launch gate — tmp_diag/hires_smoke.py, hires_compare.py)

- **[A] box-ordering / frame stress** — 8,436 boxes over 200 draws at 512×2048: **0 inverted, 0
  out-of-frame**. The geometry rewrite does not corrupt GT boxes.
- **[B] synthetic pipeline** — `__getitem__` → image (C,512,2048) in [0,1], no-data(0) preserved,
  normalized boxes in range. Montage: tmp_diag/hires_synth_montage.png.
- **[C] real eval preprocessing** — organic (pygid) + 41 (roi_data) both resample to
  `(1,1,512,2048)`, GT boxes 0 inverted / 0 oob, boxes on peaks across the full q-range.
  Montage: tmp_diag/hires_real_montage.png.
- **[D] memory** — real train step (DN + criterion + backward), batch_size=2, (2,1,512,2048):
  peak **8.4 GB / 40 GB** → batch_size=2 fits with headroom, so **no batch-size confound**.
- **[E] distribution match 1024 vs 2048** (hires_compare.py, 150 draws) — PASS: zero_frac diff
  0.030, box-interior contrast rel-diff 0.15, normalized-q-center histogram L1 0.16, box count
  45.5 vs 48.9. NOTE: the first pass showed a box-count/ring-mix divergence (54→41, ring_frac
  0.48→0.66) traced to the detector-gap radius `self.rs` being absolute (unscaled) — it clipped a
  different set of *segment* peaks at 2048; fixed by scaling `self.rs/self.ws` by WIDTH/1024 (edit 5
  above), after which the box count matched. Residual ring_frac wobble (0.63 vs 0.51) is
  50%-re-init sampling noise (the systematic signal, box count, matched).

## Gates (as every lever)

Organic + 41 AP per epoch (`exp_ap_*.txt`), decisive = the faint(vis=1)/high-q recall probe
(`diag_compare.py` / `diag_sweep.py`) vs ssl1 AND vs the deployed ensemble.

**Interpretation caveat (from the Step-1 label-completeness diagnostic, MODIFICATIONS.md phase Q):**
the organic eval is label-LIMITED — ~60% of the ensemble's FPs (74% of confident ones) sit on
labeled rings at un-annotated angles (median 1.8 px), and treating those as ignore lifts precision
0.764 → 0.890. So if the hires model detects *more* faint/high-q real peaks, some are unlabeled and
score as FPs → raw AP can stay flat on a genuine improvement. Judge on the recall probe (matched
operating point, per the phase-P calibration lesson) and the label-adjusted view, not AP alone.

## Runs
- `dino_hires1` — job **2721965**, from-scratch, 512×2048 (launched 2026-08-05). Out:
  `detector_runs/dino_hires1`. Launcher: `backbone_curation/ssl/run_detector_hires.sbatch`.
  RESUMABLE (72h cap, auto-resume checkpoint.pth). Verdict PENDING — decide at the post-lr-drop
  plateau (ep300+) on organic/41 AP + the faint/high-q recall probe vs ssl1 and the ensemble.
