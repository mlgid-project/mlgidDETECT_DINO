# Modifications log — mlgidDETECT_DINO (+ sibling mlgidDETECT)

Running record of changes made during the pygid-eval + ring/segment work. Newest phase last.
Two repos are touched: **DINO** = `mlgidDETECT_DINO`, **PKG** = `mlgidDETECT` (deployment package).

---

## A. Labeled pyGID evaluation port (DINO)
Goal: evaluate pyGID/NeXus `.h5` files (with `data/img_gid_q` + `fitted_peaks` GT) in this repo.
- **NEW `util/pygidloader.py`** — `detect_dataset_type`, `_load_fittedpeaks` (visibility 3/2/1 → confidence
  1.0/0.5/0.1, q-space → polar-pixel xyxy boxes), read-only `PyGIDDataset` (daemon worker + queue,
  `load_labels`). Mirrors the PKG loader minus the ONNX write-back.
- **`util/imgcontainer.py`** — added `visibility` field to `Labels`.
- Auto-detect routing: pygid → `PyGIDDataset(load_labels=True)`, else legacy `H5GIWAXSDataset`.

## B. Fixes to make `--eval` actually run (DINO `main.py`)
- Registered `--eval_file` (dest `eval_file_cli`, distinct so it doesn't collide with the config's
  `eval_file` in the cfg→args merge; CLI overrides config).
- Rewrote the resume/output_dir resolution: cross-platform (`os.path.dirname`, was a Windows `\\`
  split that silently forced the config `root_dir`), honors an explicit `--output_dir`, accepts
  `--resume` as a run directory (appends `checkpoint.pth`).
- Replaced the broken COCO `--eval` path (`data_loader_val`/`base_ds` never built here) with a call
  to the GIWAXS labeled eval. Extracted that into module-level **`evaluate_giwaxs_ap`** (auto-detects
  dataset type, runs the live model, computes intensity-stratified recall/precision/AP).
- **`util/nms.py`** — fixed `perform_nms` image-height (`img_container.boxes` is unset → was
  `AttributeError`); now uses `converted_polar_image.shape[-2]`. (Function later superseded in eval.)

## C. Pre/post-processing parity with the deployed PKG
Goal: DINO-side metrics reflect what the exported ONNX model does in mlgidDETECT.
- **NEW `util/postprocessing.py`** (DINO) — `box_cxcywh_to_xyxy`, `onnx_to_xyxy` (top-225),
  `filter_boxes`, ported verbatim from PKG `postprocessing/utils.py`. `evaluate_giwaxs_ap` now feeds
  the live model's raw `pred_logits`/`pred_boxes` (as numpy) through these — replacing
  `PostProcess(150)` + ring/segment `perform_nms`. Verified byte-identical to PKG.
- **`util/exp_preprocess.py`** — `_contrast_correction` now reads `PREPROCESSING_LOG /
  HISTOGRAMEQUALIZATION / (PERE)PROCESSING_PERFORMCLIPPING` from config (was hardcoded), matching PKG.
- **`util/configuration.py`** — added `MODEL_TYPE='dino'`, `PREPROCESSING_POLAR_SHAPE`,
  `PREPROCESSING_LOG`, `PREPROCESSING_HISTOGRAMEQUALIZATION`, `POSTPROCESSING_SCORE/NMSIOU/TTA`
  (mirrors PKG defaults).

## D. Checkpoint-loading fix (DINO)
- **`models/dino/swin_transformer.py`** — uncommented `window_size_h=48, window_size_w=6` for
  `swin_L_384_22k`. The `dinodetr20260304` checkpoints were trained with this elongated window (an
  uncommitted edit); without it `load_state_dict` fails (bias-table 1045 vs 1081). Window recovered
  from the checkpoint's saved `relative_position_index` (288×288 → (48,6)).

## E. Class head + ring/segment as 2 learned classes (DINO)
Goal: drop the 91-class COCO head; learn ring vs segment (segment=0, ring=1).
- **`config/DINO/DINO_4scale_swin.py`** — `num_classes=2`, `dn_labelbook_size=2`.
- **`simulation.py`** — `simulate_img` now returns `is_ring` as a 4th value, aligned with `boxes`.
- **`main.py` `SimulationDataset`** — `target["labels"] = is_ring.long()` (was all class id 1).
- Verified: model builds with a 2-logit head; full forward + DN + focal loss + backward run.
- ⚠️ The old 91-class checkpoint will NOT load under `num_classes=2`. New runs are from scratch.

## F. Class-aware NMS (DINO + PKG, kept in lockstep)
Goal: use the learned ring/segment class to pick the NMS IoU threshold (principled version of the old
y-extent heuristic). Gated by a flag so the legacy 91-class model is unaffected.
- **DINO `util/postprocessing.py`** + **PKG `postprocessing/utils.py`** — `onnx_to_xyxy` records
  `pred_labels`; `filter_boxes` does per-class NMS (ring=1 → `NMSIOU_RING` 0.1, segment=0 →
  `NMSIOU_SEG` 0.4) when `POSTPROCESSING_CLASSAWARE_NMS` is set, else single-class NMS (unchanged).
  Verified byte-identical between the two repos.
- **DINO `util/configuration.py`** + **PKG `configuration/configuration.py`** — added
  `POSTPROCESSING_CLASSAWARE_NMS` (default False), `POSTPROCESSING_NMSIOU_RING=0.1`,
  `POSTPROCESSING_NMSIOU_SEG=0.4`. `evaluate_giwaxs_ap` sets the flag True.

## G. Per-epoch dual eval, every N epochs (DINO)
- **`config/DINO/DINO_4scale_swin.py`** — `eval_files = {'41':…, 'organic':…}` (real paths; the old
  `eval_file='/datasets/41.h5'` placeholder fixed), `eval_interval = 2`.
- **`main.py`** training loop — every `eval_interval` epochs, evaluates each dataset in `eval_files`,
  writes `exp_ap_<name>.txt` (`epoch<TAB>ap_total`) and prints each; each wrapped so one failure
  never aborts training or skips the other.

## H. Improvement #3 — close the synthetic→real domain gap (Path A) — TRIED & REVERTED (no AP gain)
**Status: reverted.** The `simulation.py` masking/digitalize/quazipolar edits below were implemented,
retrained, found to give no AP improvement (see "Retrain outcome"), and reverted via
`git checkout -- simulation.py`. Only the 2-class `is_ring` work (Phase E) remains in the code. This
section is kept as a record of the audit + negative result. Audited the
synthetic training images (`simulation.py`) vs real preprocessed images (`standard_preprocessing`
on organic + 41) by comparing pixel distributions and spatial masks.

**Audit findings** (figures under `train_output/`: `synth_vs_real_corrected.png`, `mask_geometry.png`,
`synth_vs_real_final.png`):
- Contrast/intensity center already well-matched (means ~0.51–0.56) — the parity work (C) holds.
- **Dominant gap = masking**: synthetic masked only ~3.5% of pixels vs real ~30% (organic 0.35 / 41
  0.27). Real polar images have a large zero region in the high-q / high-angle corner (detector
  geometry) that the simulation barely reproduced.
- **Root cause = a WIDTH-regression bug** in `simulate_img.add_dark_area.calculate_angle_limits_mask`:
  it computed the correct corner region from `AngleLimits` but then scattered it via
  `y_shifted = col*(1+(WIDTH-512)/512)` (works at WIDTH=512; at WIDTH=1024 it doubled the column
  index → only a thin sliver was masked).
- Secondary gap: `digitalize_img` quantised to 16–64 levels/image (real ~250).

**Fixes (`simulation.py`):**
- `calculate_angle_limits_mask` rewritten to use the corner region directly at the same `q/(1+...)`
  radius the box labels use (`filter_dark_area`, line ~643), so the image mask lines up with where
  labels are clipped. Masked region set to **0** (was a gray `level`) in both the polar and
  quazipolar branches, matching real.
- `simulate_img`: re-apply the detector mask (`clahe_img * mask`) at the very end so the masked
  region is **exactly 0** in the final image (the contrast steps otherwise lift it off zero, as real
  zeroes invalid pixels AFTER contrast). Added a guard to regenerate any image masked >70% (rare
  degenerate angle-limit/quazipolar combos).
- `digitalize_img`: 16–64 → **128–256** levels.
- `AngleLimits.size_ratio_range`/`r_size` left unchanged (they also drive box clipping; masking
  magnitude is governed by the corner geometry, not these knobs).
- **Tuned the masked fraction to ~0.30**: the masking is bimodal — the **polar** branch masks ~0.24
  (matches the standard-polar eval; 41 is 0.265) while the **quazipolar** branch (a skewed geometry)
  masks ~0.55. The original 50/50 split gave ~0.40 overall. Reduced the quazipolar branch frequency
  from 50% to ~20% (`filter_dark_area`: `random_nr > .5` → `> .8`), which both centers the overall
  masked fraction at ~0.30 and skews training toward the polar geometry the model is evaluated on.

**Result (verified):** synthetic masked fraction 3.5% → **~0.32 mean (0.21–0.63)**, real ~0.30
(organic 0.35 / 41 0.27); per-image levels 16–64 → ~213 (real ~250); box centers landing in masked
pixels 0.5%; 2-class forward+loss+backward still runs.

**Retrain outcome (run `ringseg_2class_pathA_20260605-214922`, matched pre-LR-drop comparison vs the
old-sim run `ringseg_2class_20260603-142434`): NO improvement.** organic AP tied (~0.52 at ep200-258,
old 0.522 / Path A 0.521); 41 AP slightly WORSE (~0.72 vs 0.74). Plot:
`compare_pathA_vs_old.png`. Conclusion: the masking distribution gap was real but **not
performance-limiting** — DETR already ignores zero regions; the heavier masking removes some high-q
peaks from training labels, marginally hurting 41. **Decision: discarded — `simulation.py` reverted.**
The audit + negative result are themselves a useful finding (synthetic-side tweaks won't move AP;
the real lever for the organic set is #3 Path B, fine-tuning on real labeled data).

---

## Box label convention — `a_coef` 3.5 / `w_coef` 1.0 → **2.80 / 1.30** (2026-09-01)
Goal: fix what a ground-truth box MEANS — how many sigma out its edge sits. `simulation.py` builds a
box as `pos ± widths*w_coef`, `a_pos ± a_widths*a_coef` (`_boxes_from_positions`) and recovers sigma
by dividing by the same coefficients (`img_from_labels`), so the pair **cancels**: changing it
relabels the same image rather than changing the physics. At the old 3.5 / 1.0 a box was ±1.75 sigma
in chi and ±0.5 sigma in q.

**Why these values.** A 7×8 grid of chi × q rescales was applied to a trained model's predicted boxes
about their own centres, with the deployed evaluation run at every node. The sum of `ap_total` over
both gates ridges at chi 0.80–0.85 × q 1.20–1.50. Raw argmax is chi 0.80 / q 1.50 (+0.0195), but
chi 0.80 / q **1.30** (+0.0190) is statistically tied and sits at or beside the maximum on *both*
gates read separately — organic 0.5812 (grid max 0.5831), 41 0.7502 (grid max 0.7503):

    a_coef = 3.5 × 0.80 = 2.80        w_coef = 1.0 × 1.30 = 1.30

The q direction is the robust half: q 0.85 is negative in 6 of 7 chi rows while every row improves
from 0.85 toward ~1.3, and this agrees with an INDEPENDENT measurement — `box_w / FWHM_q` is 0.65 on
organic and 0.63 on 41 (the two real gates AGREE) against 0.39 in the simulator, a 1.6× deficit. In
chi the two real gates DISAGREE (`box_h / FWHM_chi` 0.73 organic vs 1.16 on 41, simulator 1.10), so
no single coefficient satisfies both and 0.80 is the compromise the grid prefers.

**Pre-flight** (relabelling residual, measured not assumed): three filters read the BOX rather than
the widths — detector-gap rejection, the 1.6 px minimum extent in `filter_dark_area`, and
`clamp_boxes` feeding sigma back into `img_from_labels` — so the surviving population shifts
slightly. 84% of frames keep an IDENTICAL object count, segments/frame 29.72 → 29.53 (−0.65%),
ring:segment 0.5232 → 0.5253 (+0.4%), rendered image mean |ΔI|/std(I) = 0.00031 (median 0.00002).
This is a relabelling plus a 0.65% segment loss, reported rather than hidden.

**Result** (from-scratch run: SSL backbone + random detector head, matched control at matched epochs).
PRIMARY gate — neither eval set regresses:

| gate | window | control | 2.80 / 1.30 | delta |
|---|---|---|---|---|
| organic | ep 200–402 (n=102) | 0.5551 | 0.5759 | **+0.0208 ± 0.0021** |
| organic | ep 300–402 (n=52)  | 0.5622 | 0.5849 | **+0.0227 ± 0.0009** |
| 41      | ep 300–402         | 0.7456 | 0.7474 | +0.0018 ± 0.0007 |
| 41      | ep 340–402         | 0.7467 | 0.7468 | +0.0001 ± 0.0009 |

SECONDARY gate — box fidelity improves on both sets (`diagnostics/box_size_probe.py` block 3),
`pred/gt` ratios moving toward 1.0 and matched IoU rising:

| model | gate | pred_h/gt_h p50 | pred_w/gt_w p50 | matched IoU p50 |
|---|---|---|---|---|
| control   | organic | 2.55 | 0.58 | 0.27 |
| 2.80/1.30 | organic | 2.14 | 0.76 | 0.33 |
| control   | 41      | 1.08 | 0.80 | 0.49 |
| 2.80/1.30 | 41      | 0.87 | 1.06 | 0.56 |

**Honest caveats.** The organic gain is the real one; **41 is a wash** (+0.0018 over ep 300–402,
+0.0001 over ep 340–402 — inside its own error bar), which is exactly what the disagreeing chi
measurement above predicts. Close-pair recall is unmoved (organic <5 px chi-gap 0.352 → 0.345, 41
0.449 → 0.472 — signs flip between pre- and post-drop readings, i.e. noise), so this does **not**
address the small-peak resolution problem. Separately, matching organic's measured convention
EXACTLY (`a_coef` 1.85) COSTS −0.0118 organic / −0.0100 on 41: the evaluation's IoU floor of 0.1 does
not reward tightness, so these values target the AP optimum rather than convention-matching, and land
only partway toward the real labels on both axes.

⚠️ Old checkpoints still LOAD (no architecture change), but a model trained at 3.5 / 1.0 predicts
boxes in the old convention. Mixing the two is a silent metric shift, not an error.

## Simulator fixes carried with the convention change
- **`simulate_img` re-init discarded the config (REQUIRED for the above).** On ~50% of calls the
  method ran a bare `self.__init__()`, which reset `self.sim_config` to a DEFAULT `SimulationConfig`
  (and `device` to `'cuda'`) — silently throwing away any configured simulation, `box_coef_override`
  included, on half of all frames. Now re-inits as `self.__init__(sim_config=self.sim_config,
  device=self.device)`. Byte-identical on the default path: `SimulationConfig` has constant defaults
  and its construction consumes no RNG. The surrounding `global WIDTH` block was dead — every arm of
  its `if` assigned 1024 — so it only ever clobbered a configured WIDTH; removed.
- **`min_nms` was a dead config knob.** `simulate_labels` called
  `filter_nms(pos, widths, a_pos, a_widths, sc.min_nms)` — five positional args, which put the
  threshold into the (unused) `is_ring` slot and left `min_nms` at its default. Now passed by
  keyword; `filter_nms` gained a docstring saying `is_ring` is accepted for signature compatibility
  only. Byte-identical today because the config value equals the default (0.001).
- **Three hardcoded 1024/512 constants now scale with `WIDTH`**: the background ring box
  (`Tensor([[116,0,128,512]])`), the quazipolar dark-area coefficient
  (`(1 - (WIDTH-512)/1024)` → `512/WIDTH`, which only agreed with itself at 1024), and
  `create_detector_mask`'s `rs`/`ws` sampling (absolute 80..380 / 1..7 px, so the detector gap sat at
  a different physical q at any other resolution and clipped a different set of segment peaks).
  All byte-identical at the shipped `WIDTH=1024`; correctness only.

---

## Results so far (run `ringseg_2class_20260603-142434`, ep360 of 500; baseline also ~ep350)
| set | new 2-class @ep360 | old 91-class baseline | notes |
|---|---|---|---|
| organic (pygid) | **0.554** (still rising) | 0.552 | even / slight edge new |
| 41 (roi_data)   | **0.758** (peak 0.768) | ~0.751 | slight edge new |
Plot: `train_output/ringseg_2class_20260603-142434/ap_curves.png`.

## Diagnostics & roadmap
- **Diagnostic C** (where AP is lost) run on the best checkpoint: recall 0.49 / precision 0.81 on
  organic; misses dominated by faint (vis=1 recall 0.28), high-q (recall 0.34 for q>682), and segment
  peaks; ~12 FP/img, half high-confidence. Script `diagnostics/diagnose_C.py`, fig `diagnostics/diagnose_C.png`.
  → it's a representation/sensitivity ceiling, not preprocessing. Full analysis + forward ideas
  (self-supervised backbone on real+sim; physics-informed) in **`ROADMAP.md`**.
- **Physics wins validated & shelved** (`diagnostics/{diagnose_rings,sweep_nms,viz_fp}.py` + PNGs):
  symmetry is out-of-frame (single 0–90° quadrant); ring-aware FP rejection fails (FPs are ON rings,
  ~93% within 8px-q of a real peak); NMS tuning doesn't help (FPs aren't duplicates). **The FPs are
  confident, on-ring, at unlabeled angles → likely real peaks the GT missed (incomplete labels).**
  So precision 0.81 is likely pessimistic and the eval may be label-limited, not model-limited. See
  ROADMAP.md "KEY FINDING". Next: expert review of `viz_fp.png` to confirm.

## I. pygidSIM physics peak configuration, second attempt — SUBMITTED AT 100% (2026-09-09)

Replace the simulator's invented peak properties with real crystallography via pygidsim
(CIF -> structure-factor peak list). **The intensities are the point of this track.** The standard
sim draws peak intensities UNIFORMLY in a bounded range (`gen_intensities`, `simulation.py:1161`:
`rand()*(hi-lo)+lo` over ring `(2,50)` / segment `(10,50)`), applies a 2x boost to low-q/narrow
peaks and rescales linearly — no skew, essentially no q-correlation. Real diffraction has a few
strong reflections and a long weak tail spanning orders of magnitude, set by structure and form
factors. Peak POSITIONS come along for the ride but are secondary: the current detector is not
physics-based, so positions matching only roughly is acceptable.

### Relation to the declined phase P
Same lever as phase P on branch `development` (`docs/PHYSICS_SIM_INVESTIGATION.md`), DECLINED
2026-08-03: from-scratch `dino_physics_scratch1` organic 0.5395 vs ssl1's 0.5634, 41 0.6255 vs
0.7454, and at a matched operating point it lost on every stratum including the high-q one it was
built to fix (it only looked good at a fixed score because it fired 5.5x more boxes). Four things
differ now, plus lr 4e-5 where phase P ran at 1e-5:

1. **Library.** Phase P's bank was 98.5 % perovskite — 26,341 of 26,734 entries from a COD
   perovskite selection, only 393 from the 51 organic CIFs. This one is 60,474 COD organics.
2. **q coverage.** The old bank stopped at |q| = 4.24 (`q_xy_max = q_z_max = 3.0`), leaving the
   outer 14 % of an organic frame (q_max 4.95) with no physics peaks. Now 3.5 -> 4.95.
3. **Box convention.** Phase P predates `box_coef_override` and sampled its own half-widths.
   `physics_simulation.py` now builds boxes as centre +/- sigma*coef using the run's own
   `(a_coef, w_coef) = (2.80, 1.30)` and the config's width ranges.
4. **Dilution 25 %, not 50 %.**

### What was built
- **`physics_sim/fetch_cod_organics.py`** (NEW) — COD `result.php` runs the selection server-side
  and returns metadata for 97,992 candidates in one request, so none of the 26.6 GB
  `cod-cifs-mysql.tgz` is needed. Filtered to 60,474 organic structures (must contain C+H; only
  non-metals plus at most one OSC metal centre, since the eval family includes CuPc/ZnPc; cell
  volume 500-8000 A^3; deduped on rounded cell). Median cell 2374 A^3. Fetched by rsync
  `--files-from` against the sharded `cif/<d1>/<d2d3>/<d4d5>/` tree.
- **`physics_sim/generate_bank.py`** (ported from `git show e14f8e9:...`) — parameterised CIF dir
  and output, multiprocessing, random sampling under `--limit`, `Q_XY_MAX = Q_Z_MAX = 3.5`.
  **`physics_sim/run_bank_organic.sbatch`** builds the full bank on cpu-galvani.
- **`physics_simulation.py`** (ported from `git show b8f220b:...`) — the four changes above, plus
  `sim_config` threading: the original built `FastSimulation()` with DEFAULT coefficients, so a
  `box_coef_override` never reached `img_from_labels`' sigma recovery.
- **`simulation.py`** — one additive function `contrast_like_real` (49 insertions, 0 deletions;
  no existing path changes).
- **`main.py`** — `SimulationDataset` gains `use_physics_sim` / `physics_sim_fraction` /
  `physics_bank_path` / `unify_contrast`, default off; `__getitem__` picks per sample.
- **`config/DINO/DINO_4scale_swin_physics2.py`** + **`run_detector_physics2.sbatch`**.
- **`diagnostics/bank_stats.py`**, **`peak_position_gate.py`**, **`peak_intensity_gate.py`**.

### Bundled second change: unify_contrast
The sim and the real preprocessing are two separate implementations differing in ORDER (sim
`apply_log -> apply_he -> apply_clip_img`, where the clip is a rare p=0.05 mean+/-k*std clamp; real
`percentile clip (5, 99.5) -> log -> HE`, always) and in the log ARGUMENT (sim maps onto a
synthetic decade range `normalize(x)*U(50,5000)+1` because its intensity units are arbitrary; real
takes `log10(|x|+1e-7)`). Physics intensities are what make the real form applicable, so the two
levers are tested together — deliberately, at the cost of attribution if the run moves.

### The evidence for the run: intensity SHAPE (`diagnostics/peak_intensity_gate.py`)
Intensity normalized by the brightest peak of its own pattern, so the three sources' different
units cancel. Real organic amplitudes had to be MEASURED from `data/img_gid_q` (patch max minus a
local background ring) because the label file's `amplitude` column is all zeros; 41 uses its
`peak height`.

| source | med I/Imax | frac < 0.1 | log10 dynamic range |
|---|---|---|---|
| REAL organic (measured) | 0.007 | 0.911 | 1.61 |
| REAL 41 (`peak height`) | 0.009 | 0.859 | 2.20 |
| **SIM current (uniform)** | **0.363** | **0.057** | **0.62** |
| bank perovskite | 0.033 | 0.783 | 2.16 |
| bank organic | 0.043 | 0.791 | 1.09 |

The standard sim is off by ~50x on median relative intensity, produces 5.7 % weak peaks where
reality has 86-91 %, and spans 0.6 decades where reality spans 1.6-2.2. Both physics banks are far
closer on all three. **This is the axis the track is for, and the mismatch is large.** Note the
organic bank's dynamic range (1.09) is NARROWER than the perovskite bank's (2.16) and than either
real set -- worth revisiting if the run underperforms.

### Wiring check (`diagnostics/physics_smoke.py`, job 2859173, gate bank)
All contract assertions pass in both contrast modes -- shape, dtype, finiteness, [0,1] range, mask
shape, non-empty boxes, and the `x2>x1 & y2>y1` guard the DINO matcher asserts at
`util/box_ops.py:53`.

| generator | boxes/img | rings | mean px | max abs px outside mask |
|---|---|---|---|---|
| physics, unify_contrast=False | 37.6 | 9.8 % | 0.537 | 1.000 |
| physics, unify_contrast=True | 41.8 | 12.4 % | 0.310 | 0.562 |
| standard sim (control) | 44.2 | 52.8 % | 0.560 | 1.000 |

Two things to watch, neither blocking:
- **Ring fraction.** Physics produces 10-12 % rings against the standard sim's 52.8 %, because the
  composition is 0-1 powder + 1-2 oriented entries with 3-15 rings vs 8-60 spots. At 25 % dilution
  the overall ring share moves 52.8 % -> ~42 %. Tunable via `N_POWDER` / `N_ORIENTED` /
  `RINGS_PER_POWDER` in `physics_simulation.py` if the class balance turns out to matter.
- The standard sim does NOT zero pixels outside the mask either (max abs 1.000), so that is not a
  physics regression; `unify_contrast=True` is actually cleaner (0.562) because
  `contrast_like_real` zeroes the invalid region as the real pipeline does.

### Measured on the way (positions only; kept because it is cheap and reusable)
Peak POSITIONS in the normalized radial coordinate the detector sees, x = q/q_max, against the
real labeled peaks, using each eval set's own q_max (organic 4.95 for every entry; 41 3.82).
Two-sample KS, effect size not p-value; noise floor 1.36/sqrt(817) = 0.048:

| source | KS vs organic | KS vs 41 | KS sum |
|---|---|---|---|
| SIM current (uniform, `simulation.py:933`) | 0.160 | **0.121** | **0.280** |
| bank perovskite (phase P's) | **0.140** | 0.333 | 0.473 |
| bank organic (2,000-CIF gate sample) | 0.215 | 0.207 | 0.422 |

So on POSITIONS alone the uniform sim is already competitive and no bank dominates — expected,
and not a reason to stop, since positions are not what this track is for. Two side findings worth
keeping:
- **Library chemistry does not predict which eval set a bank matches**: the perovskite bank fits
  the ORGANIC eval better and the organic bank fits the PEROVSKITE eval better. Note `41.h5` is
  itself a perovskite set (MAPbI3, FAPbI3, FAPbBr3, 2D BA/PEA, SiOx|InOx), so phase P's bank was
  chemistry-matched to the very set it damaged most.
- **Physics banks skew outward**: a bank holds every symmetry-allowed reflection to q_max and
  reflection count grows ~q^3, while real LABELED peaks are those a human could see and fit.
  Top-200-by-intensity does not undo it.

Label-format note, worth recording because it cost time: `organic_labeled.h5` stores `amplitude`,
`q_xy`, `q_z` and `is_ring` as ALL ZERO — only `radius` (A^-1), `angle` (deg) and `visibility` are
populated. 41's `roi_data` does carry `peak height` and `confidence_level`, with `radius` in
reciprocal-image pixels.

### Dilution: 25% was built, 100% is what runs (user decision, 2026-09-09)
`DINO_4scale_swin_physics2.py` (fraction 0.25) and its sbatch are kept for the record but the run
was CANCELLED before starting. The run that is queued is `DINO_4scale_swin_physics3.py` at
**fraction 1.0** — every training image's peak configuration comes from a CIF. Rationale: the next
model iteration is intended to use physical peak POSITIONS as well as intensities, so the training
distribution should be physical end to end rather than a dilution of a random one. Job 2862076,
`afterok` on the bank job, output `detector_runs/dino_physics3_1`.

Three risks that 25% contained and 100% does not, recorded so the run is read honestly:
- **Ring fraction.** Physics images are 10-12 % rings against the standard sim's 52.8 %. At 100 %
  the detector barely sees rings, and `41.h5` is ring-heavy. If 41 collapses while organic holds,
  check composition before concluding anything about intensities.
- **Positions.** The KS table above shows the organic bank (0.422) is a WORSE match to the real q
  distribution than the uniform sim (0.280). At 25 % that is a perturbation; at 100 % those are
  the only positions the model ever sees.
- **Eval-cleanliness.** See below — unchanged, but it now applies to every image rather than one
  in four.

### On-the-fly generation: measured, deferred
Measured on 145 random CIFs from the library (`mlgid_physics` env): CIF parse median 661 ms,
`giwaxs_sim` per orientation p50 87.5 ms / p90 500 ms / p100 1096 ms, mean 184 ms, 3 % parse
failures. At ~2 sims per image that is ~368 ms/image unscreened, ~6 min per 1000-image epoch,
against ~0.545 s/image of GPU time — so with 12 dataloader workers, on-the-fly generation would
NOT be the bottleneck, and pre-parsing structures into a per-worker pool removes the 661 ms parse
entirely. It would also give a fresh random orientation per image, where the bank freezes 8 per
CIF. Deferred anyway: `pygidsim` + `xrayutilities==1.7.10` live only in `mlgid_physics`, not in
the training env `DINO_GIWAXS`, and the bank already supplies ~525k entries against ~500k training
images — each structure/orientation is drawn roughly once per run. Revisit if the next iteration
needs structure identity carried into the label. (An earlier n=20 sample showed a 41 s outlier and
led to a wrong "catastrophic tail" conclusion; it did not recur in 145 structures.)

### Status and caveat
**CRASHED AND RESTARTED.** Jobs 2862172 / 2865413 both died on a NaN training image; see section
K for the two defects that came out of it. The run is now `dino_physics3_2` (job 2867359) from
epoch 0 -- `dino_physics3_1`'s weights are not a valid warm start, since its 82 epochs covered only
~41,000 distinct images. `_1`'s logs and AP curves are kept for reference.

**NOT EVAL-CLEAN YET.** The bank is built with `--no-exclusions`: the mlgidMATCH-based exclusion
pass (`physics_sim/build_exclusion_list.py` on `development`) is not ported, so a COD structure
matching an eval material can still contribute peaks. Any AP from this run is provisional until
the bank is rebuilt with exclusions. Bank job 2859156 on cpu-galvani; training job 2862076 chains
off it with `afterok`.

## J. Gradient accumulation — effective batch 24 on one GPU (2026-09-09)

Every run in `detector_runs/` is batch 2. The one real large-batch attempt, `dino_truebatch8_1`
(batch 8, lr left at 4e-5, otherwise identical), LOST: organic **0.5808 vs `dino_lr4e5_1`'s
0.6081**, 41 0.7622 vs 0.7613. But it is confounded by OPTIMIZER STEPS — `__len__` is a fixed 1000
images/epoch (`main.py:163`), so batch 8 took 4x fewer updates for the same epoch count and may
simply have been undertrained. This run separates the two by raising lr WITH the batch.

- **`engine.py`** — `grad_accum_steps` (config-only, read via `getattr`, default 1). The loss is
  divided by `accum` so the accumulated gradient is the MEAN over the effective batch, not the sum
  (otherwise lr is silently scaled by `accum`); `clip_grad_norm_` (max_norm 0.1) applies to the
  ACCUMULATED gradient on update steps only; `zero_grad()` moves from before the backward to after
  the step, plus once before the loop. **Warmup counts OPTIMIZER steps, not iterations** — else
  `warmup_steps=300` becomes 300*12 iterations, a 7-epoch ramp instead of 0.6. `onecyclelr` and
  the EMA update move inside the update guard, both being per-optimizer-step semantics.
  At `accum == 1` the path is provably identical to before, so no existing config changes.
  Verified numerically: accum-12 reproduces a true batch-24 gradient to 2.7e-07.
- **`config/DINO/DINO_4scale_swin_accum24.py`** — `grad_accum_steps=12` (effective batch 24 at the
  memory cost of 2, so swin-L at 512x1024 still fits one a100), `lr = lr_backbone = 1.4e-4`
  (sqrt(12) x 4e-5, the Adam scaling rule), `warmup_steps=300` (~7.2 epochs).
- **Sizing is compute- AND axis-matched to `dino_lr4e5_1`**: 1000 images/epoch x 500 epochs =
  500,000 images, so the curves overlay and the post-280 rule applies unchanged. 500 iterations
  and **42 optimizer steps** per epoch; 20,833 total against batch 2's 250,000.
  NOTE raising images/epoch does NOT buy optimizer steps — total steps = total images / effective
  batch however epochs are sliced. Only more compute, a higher lr, or a smaller effective batch
  closes that gap. The 12x step deficit is intrinsic and is what the lr must cover.
- **The lr deliberately enters a band that FAILED at batch 2.** The base config records 1e-4 and
  1.6e-4 classifying fine (class_error 37.7% -> 1.5%) but never localizing (`loss_giou` stuck at
  1.59 / 1.71 at epoch 85 against 0.35). The hypothesis is that this was a GRADIENT-NOISE ceiling,
  not an lr ceiling: 12x the batch cuts noise ~3.5x. **Decisive early gate, epoch ~85:** if
  `loss_giou` is still above ~1.5, the hypothesis is dead, 4e-5 is an lr ceiling independent of
  batch size, and the run should be killed rather than burning 76 h.
- **CAVEAT — not bit-exact to a true batch 24.** DINO normalizes its losses by `num_boxes` over the
  batch (`models/dino/dino.py:408,413`). Accumulating 12 micro-batches each normalized by its OWN
  box count gives the mean of per-micro-batch means, not the true batch-24 mean, so images in
  box-sparse micro-batches are weighted up. Standard accumulation practice, but a real difference
  from `dino_truebatch8_1`'s genuine batch 8.
- **Single GPU on purpose.** DDP is not wired here: `main.py:323` has `init_distributed_mode`
  commented out, `model_without_ddp = model` with no DDP wrap, `DistributedSampler` imported but
  unused, and no sbatch in the repo has ever used `--ntasks=2`. A 2-GPU allocation would idle the
  second card. Accumulation first so a negative result is attributable to batch size rather than
  to an untested distribution path. (Per-rank seeding is already correct at `main.py:374`, so that
  part would not be the hard bit.)
- Job 2862095, output `detector_runs/dino_accum24_1`. Needs ~76 h against the 72 h limit, so
  expect exactly one resubmit. Verdict post-280 against `dino_lr4e5_1` (0.6081 / 0.7613).

## K. Two defects the `dino_phys3` crash exposed (2026-09-10)

`dino_physics3_1` died twice with `AssertionError` at `util/box_ops.py:52`, once at epoch 41 (job
2862172, fresh start) and once at epoch 82 (job 2865413, resumed at 41). Chasing it turned up two
independent bugs, one fatal and one silent. Both are fixed; the run restarts from epoch 0 as
`dino_physics3_2` (job 2867359, `detector_runs/dino_physics3_2`).

### K1. A NaN training image kills the run (fatal)

**Reading the traceback.** The assert is on `boxes1`, which at `models/dino/matcher.py:87` is
`out_bbox` -- the PREDICTIONS, not the targets (an inverted TARGET box trips line 53 instead, which
is the older `dino_physics1` failure, already fixed by the `x1<x2 & y1<y2` filter in
`physics_simulation._attempt`). Predicted boxes are sigmoid `cxcywh`, so `x2 >= x1` can only fail
on NaN, and this run has `amp False`, so there is no fp16-overflow route: the NaN was in the input.

**It is the data, not the model.** `num_workers=0` (`main.py:489`) and the seeds are set once per
process, so image generation is one deterministic stream. Both processes crashed after 41 epochs x
1000 images + ~510 -- the same stream index -- from completely different weight states, and no
`Loss is nan, stopping training` line appeared, so the weights were healthy going in.

**Reproduced and traced.** Replaying the seed-42 stream (`tmp_diag/phys_nan_probe2.py`), draw
15,850 of 25,000 returns `nan_frac = 1.0000` with finite boxes -- about **1 image in 20,000**,
which is why 41 clean epochs ran first. Per-stage instrumentation on that draw
(`tmp_diag/phys_nan_replay.py`):

```
apply_salt_pepper_noise   min +0  max +1   const=False
contrast_like_real        min +0  max +0   const=True    <- every VALID pixel identical
apply_kernel              min +0  max +0   const=True
digitalize_img            min +0  max +0   const=True
normalize                 nan = 524288  (512x1024, i.e. all of them)
```

`mask_valid` was 0.459 and both peaks of that frame fell in the masked-out region, so the valid
area carried no signal: the 5/99.5 clip quantiles coincide, `log10` maps them to one constant, and
the closing `where(m, img, 0)` zeroes the frame. `normalize()` is `(img - min) / (max - min)` --
0/0 -- so every pixel becomes NaN.

**Why the old guards missed it.** `_attempt` checked `img.min() == img.max()`, which is **False for
an all-NaN image** because `NaN != NaN`; a min/max test alone waves NaN through. And nothing
checked the image AFTER the contrast chain at all, which is exactly where it is produced.

**Fix.** `_usable(img)` in `physics_simulation.py` tests `isfinite` AND contrast, replaces both old
guards, and runs after the contrast chain; a failing draw is retried (`simulate_img` already
retries 20x before raising). `main.SimulationDataset.__getitem__` gets the same check as a belt for
BOTH simulators -- nothing here is physics-specific, `FastSimulation` ends in the same `normalize()`
and `apply_poisson_noise` calls it internally too (innocent on this draw, still a possible route).
Verified: 20,000 draws through the guarded simulator, 0 rejected images reaching the caller.

### K2. The RNG is seeded per PROCESS, so a resumed run repeats its first epochs (silent)

`main.py` sets `seed = args.seed + get_rank()` once, at startup, and the resume block restores
model, optimizer, lr_scheduler and `start_epoch` but **no RNG state**. With `num_workers=0`,
`SimulationDataset.__getitem__` ignoring the index it is handed, and `dropout = 0.0` (so the model
consumes no random numbers), every training image comes off that one stream in order, from image 0
of the process.

So each wall-clock resubmit restarted the simulator from the beginning. `dino_physics3_1` resumed
at epoch 41 and its epochs 41-81 were run 1's epochs 0-40, image for image: 82 epochs of training
over ~41,000 distinct images, each seen twice. The two crashes landing at the identical stream
index is the proof.

**This is not specific to the physics track.** Every run that crossed the 72 h limit did it, at its
own restart epoch: `dino_rawcounts1` @218 (twice), `dino_truebatch8_1` @175 and @183, `dino_mc` @395,
`dino_lrsweep` @88 and @221, `dino_hires` @264, `dino_mcc` @4, `dino_physics3_1` @41. `dino_accum24_1`
(section J) expects one resubmit and will now pick up the fix.

**Fix.** `main.py` reseeds at the top of each epoch from `seed + 1000*(epoch + 1)`, so the stream is
a property of the EPOCH NUMBER rather than of how long the process has been alive: a resumed epoch
41 draws what an uninterrupted epoch 41 would, and no epoch repeats another. Restoring saved RNG
state would also have worked; per-epoch seeding gets the same determinism without adding anything
to the checkpoint format.

**Does it explain `dino_physics3_1`'s flat AP after epoch 41?** Consistent, not proven. Slope in
ap_total per 100 epochs, before vs after that boundary:

| run | organic ep1-40 | organic ep41-80 | 41 ep1-40 | 41 ep41-80 |
|---|---|---|---|---|
| `dino_physics3_1` (restarts @41) | +0.295 | **-0.037** | +0.310 | **-0.037** |
| `dino_lr4e5_1` | +0.456 | +0.041 | +0.600 | +0.065 |
| `dino_ssl1` | +0.674 | +0.118 | +1.004 | +0.091 |
| `dino_boxconv1` | +0.513 | +0.129 | +0.405 | +0.295 |
| `dino_rawcounts1` | +0.585 | +0.081 | +0.486 | +0.067 |

Every run flattens hard after epoch 40 -- that is the normal shape of the curve, not evidence by
itself -- but `dino_physics3_1` is the only one that goes NEGATIVE on both eval sets, and it does so
exactly at its restart. Confounds remain: it entered the window higher than the others (0.5388
organic at ep20-40 against `dino_lr4e5_1`'s 0.4811, so less headroom) and it is a different data
recipe. The clean test is `dino_physics3_2`, which now trains on 82 distinct epochs.
The comparison runs are NOT clean controls for this: they carry the same defect at their own
restart epochs, all of which fall outside this window.

## Open / not yet done
- Path A (#3 simulation fix) tried and reverted — no AP gain (see Phase H). The 2-class model from
  `ringseg_2class_20260603-142434` (organic ~0.55 / 41 ~0.76 by ep360) stands as the current best.
- Improvement #3 Path B (fine-tune on real labeled data) — the remaining real lever for organic AP,
  but needs labeled real data held out from eval (currently all of 41 + organic is used for eval).
- Improvement #4 (backbone/schedule) — not started.
- Optional: a clean 500-epoch run of the (un-Path-A) 2-class model for the final number.
- `PREPROCESSING_FLIPHORIZONTAL` and TTA not ported (default off).
- Git: phases A–G committed on branch `pygid-eval-ringseg` (pushed). Phase H reverted (not
  committed). Only `MODIFICATIONS.md` is currently modified (this negative-result record) — commit
  when convenient.
