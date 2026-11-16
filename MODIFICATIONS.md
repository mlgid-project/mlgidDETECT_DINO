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

## L. The simulator's ring rate is the organic/41 lever — `physics_n_powder` (2026-09-11)

`dino_physics3_2` (100% physics-CIF, section I) splits the two eval gates harder than any run so
far. Against `dino_lr4e5_1` at matched epochs it **wins organic in 33 of 35 shared epochs** and
**loses 41 in 34 of 35**:

| | organic | 41 |
|---|---|---|
| `dino_physics3_2`, mean ep>=50 | **0.5528** | **0.5275** |
| `dino_lr4e5_1`, same epochs | 0.5224 | 0.7167 |
| delta | **+0.030** | **-0.189** |

That asymmetry is far too large and too consistent to be noise, and risk (a) in the physics3
config header named the suspect in advance: *"RING FRACTION. Physics images are 10-12% rings; the
standard sim is 52.8%. At 100% the detector sees almost no rings, and 41.h5 is ring-heavy. If 41
AP collapses while organic holds, this is the first thing to check — it is a composition artefact,
not evidence about intensities."*

### L1. The measurement

120 frames per simulator, geometric ring criterion (box spans >= 70% of the valid chi rows at its
own radius — `is_ring_geom`, `diagnostics/postproc_diag.py:115`; `41.h5` does not populate
`is_ring` at all, so geometry is the only trustworthy label, see section AA.4). Current box
convention `a_coef=2.8 w_coef=1.3`. Script: `diagnostics/ring_rate_sims.py`.

| source | rings/frame | segs/frame | ring:segment | frames with 0 rings |
|---|---|---|---|---|
| real 41 | 16.90 | 24.0 | **0.704** | — |
| real organic | 3.50 | 98.6 | **0.035** | — |
| sim legacy | 17.05 | 30.7 | 0.555 | 34/120 |
| sim physics-CIF | **4.38** | 37.2 | **0.118** | 62/120 |

**The legacy sim sits on 41's composition; the physics-CIF sim sits on organic's.** Rings are the
EASY class on both gates (41 ring recall 0.856 against 0.713 for segments) and 41% of 41's
objects, so cutting the ring supply to a quarter costs 41 far more than it costs organic. This
also explains the standing pattern that every legacy-sim run scores 0.74-0.77 on 41 but only
0.55-0.62 on organic: the legacy sim is **5x too ring-heavy for organic**.

CAUTION comparing to the older figure. The eval-dataset table records the legacy sim at 7.9
geometric rings/frame, ratio 0.206. That was measured under the OLD box convention; `a_coef=2.8`
widens boxes in chi, so more arcs now clear the 70% span test. Under the current convention the
sim's own flag and the geometry agree (17.14 vs 17.05) where before they did not (14.8 vs 7.9).
The two numbers are not comparable — do not read a change in the simulator into the difference.

### L2. The mechanism, one line

`physics_simulation.py:49`, `N_POWDER = (0, 1)`: `random.randint(0, 1)` gives **half of all
physics frames zero powder entries** and the other half exactly one (3-15 rings each). The legacy
analogue is `simulation.py:486`, `rings_or_seg_or_both = random.random()` split into thirds (rings
only / segments only / both), which is how it reaches 17 rings/frame.

### L3. The fix — a WIDE range, not a higher fixed one

`PhysicsSimulation.__init__` takes `n_powder=None`; `_attempt` reads `self.n_powder`. `None` keeps
`N_POWDER`, so **every run up to and including `dino_physics3_2` is bit-identical** —
`random.randint(*self.n_powder)` consumes the RNG the same way for the same range. Verified: frame
0 of the default path reproduces `img.sum() = 205665.921875` and 50 boxes either way, and the
measured 4.38 rings/frame is unchanged. A malformed range raises `ValueError`.

`main.py` passes `n_powder=getattr(args, 'physics_n_powder', None)` and prints the value in the
epoch-0 banner, so the wiring is visible in the log rather than inferred.

The two gates want OPPOSITE ring rates (0.704 vs 0.035), so **no single rate matches both** and a
compromise value matches neither. Widening the range instead makes the regime a per-image random
variable, the way the legacy sim already does it coarsely:

| variant | rings/frame | segs/frame | ring:segment | frames with 0 rings |
|---|---|---|---|---|
| default / explicit `(0, 1)` | 4.38 | 37.2 | 0.118 | 62/120 |
| **`(0, 3)` — physics4** | **11.98** | 36.4 | **0.329** | 34/120 |

`(0, 3)` lands between the two gates and its ring-free fraction now matches the legacy sim's
exactly (34/120).

### L4. The run

`dino_physics4_1`, job 2869152, `config/DINO/DINO_4scale_swin_physics4.py`,
`backbone_curation/ssl/run_detector_physics4.sbatch`. Diffed through the config loader against
physics3: `physics_n_powder` is the ONLY key that differs, and `dino_physics3_2` is running
concurrently as its control, so any AP difference is the ring rate alone.

Wiring check at epoch 0: the banner must end `n_powder=(0, 3)`. If it reads `(0, 1)` the config
key did not reach `args` and the run is a duplicate of physics3_2 — kill it rather than let it
burn 72h.

**Three outcomes, only one of which supports the idea.** Recorded before the numbers arrive:
1. 41 climbs toward the 0.72-0.76 band AND organic keeps physics3_2's +0.030 — the ring rate was
   the whole story, and the same lever should then be widened on the legacy sim for organic.
2. 41 climbs but organic falls back to the legacy 0.55-0.62 band — ring rate merely TRADES the two
   gates and the physics bank buys nothing beyond a different point on the same curve.
3. 41 does not move — the cause is elsewhere, and peak POSITIONS become the prime suspect (risk (b)
   of section I: the bank's q distribution is a WORSE match to the eval sets than uniform draws,
   KS sum 0.422 against 0.280, noise floor 0.048).

Judge post-280 for the verdict — pre-drop ranking correlates with the plateau at only rho = 0.49.
The matched-epoch comparison is for spotting a dead run early, not for calling the result.

Inherited risks are unchanged from section I: peak positions (b) and NOT EVAL-CLEAN (c) — the bank
is built with `--no-exclusions`, so a COD structure matching an eval material can contribute peaks.
Any AP from this run is PROVISIONAL until the bank is rebuilt with exclusions.

### L5. A related lever that is now CLOSED

`dino_edge1` (`SimulationConfig.edge_peaks`, keep the full box at the mask edge instead of clamping
it) **ran and lost.** lr 1e-5, `box_coef_override=[2.8, 1.3]`, stopped at epoch 210 on 2026-09-08,
never reached the lr drop. Against its correct control `dino_boxconv1` (same lr, same box
convention), mean over epochs 150-210: organic 0.5310 vs 0.5424 (**-0.011**), 41 0.7233 vs 0.7387
(**-0.015**) — i.e. back to plain `dino_ssl1` level (0.5363 / 0.7235). Negative on both gates.
Judged pre-drop only, but there is no positive signal anywhere in the curve to wait for. Do not
re-propose the edge-peak sim change; `edge_peaks` stays default False.

## M. The physics sim is recall-strong and precision-poor — `dn_number` (2026-09-15)

`dino_physics4_1` finished at epoch 434 (72h wall, not a crash): organic **0.5804**, 41 **0.6412**,
against `dino_lr4e5_1`'s 0.6222 / 0.7631. The question was why. Decomposing AP into recall and
precision says the physics simulator is **not** the problem people assumed it was.

### M1. The measurement

All numbers at the DEPLOYED operating point — top-225 in `onnx_to_xyxy`, class-aware NMS (ring 0.1
/ segment 0.4), score > 0.1 — from `diagnostics/{numselect_sweep,capsplit,fploc,tpiou,nmssweep}.py`.
Each caches the raw `(pred_logits, pred_boxes)` once per gate and replays postprocessing off the
cache, so a whole sweep costs one forward pass. Replay validated exact: physics4 at 225 reproduces
41 = 0.6412 and organic = 0.5804, the tails of its own `exp_ap_*.txt`.

| gate | model | recall | precision | ap_total | med TP score | med FP score |
|---|---|---|---|---|---|---|
| 41 | lr4e5 (ep367) | **0.888** | **0.445** | **0.7631** | 0.848 | 0.198 |
| 41 | ssl1 (ep437) | 0.869 | 0.453 | 0.7441 | 0.810 | 0.199 |
| 41 | physics4 (ep434) | 0.856 | **0.256** | 0.6412 | 0.657 | 0.258 |
| 41 | physics5 (ep240) | 0.834 | 0.274 | 0.5496 | 0.573 | 0.195 |
| organic | lr4e5 | 0.681 | **0.643** | 0.6222 | 0.879 | 0.212 |
| organic | ssl1 | 0.643 | 0.630 | 0.5683 | 0.703 | 0.183 |
| organic | physics4 | **0.752** | 0.434 | 0.5804 | 0.612 | 0.294 |
| organic | physics5 | 0.704 | 0.437 | 0.5685 | 0.436 | 0.219 |

**On organic the physics sim already beats the baseline at finding peaks.** physics4 recovers 76.8
of the 102.1 GT peaks per frame; lr4e5 recovers 69.5. Uncapped (`num_select` 900) physics4 reaches
recall **0.886** while lr4e5 **saturates at 0.687** — the legacy-sim model has nothing above score
0.1 left to give, the physics one has a fifth of the organic peak population sitting in its output,
correctly localised, that the ranking cannot surface. **On 41 there is no recall advantage at all**
(0.856 vs 0.888) and the same precision collapse. The two gates are different problems.

Score separation (median TP − median FP) is the axis: lr4e5 0.650 / 0.667, ssl1 0.611 / 0.520,
physics4 0.399 / 0.318, physics5 0.378 / 0.217.

### M2. The excess is a duplicate spray, not noise firings

Every FP split by IoU with the nearest GT (`diagnostics/fploc.py`):

| model | gate | dup (IoU>0.3) | near (0<IoU≤0.3) | bg (IoU=0) |
|---|---|---|---|---|
| physics4 | 41 | 3.9/fr | **55.3/fr** | 42.9/fr |
| lr4e5 | 41 | 1.5/fr | 13.0/fr | 30.9/fr |
| physics4 | organic | 7.5/fr | **53.2/fr** | 39.2/fr |
| lr4e5 | organic | 0.8/fr | 10.2/fr | 27.6/fr |

Background firings are comparable (1.4×); the near-miss bucket is **4.3×**. And the matched boxes
are fine — median matched IoU on organic is **0.394 for physics4 vs 0.343 for lr4e5**, i.e. it
localises *better* (`diagnostics/tpiou.py`). So it is not blind and not sloppy: it hedges, emitting
several boxes per peak at IoU 0.1–0.4 with the true one — just under the 0.4 segment NMS threshold,
so they survive.

Confirmed by `diagnostics/nmssweep.py`: segment NMS IoU 0.4 → 0.10 is worth physics4 **+0.042 on 41
and +0.032 on organic**, while lr4e5 moves ≤0.002 and ssl1 ≤0.007. **Not shipped** — even at its
best it is still below baseline on both gates (0.6827 vs 0.7636; 0.6124 vs 0.6222), and it would
change the deployed mlgidDETECT path and break comparability with every existing run. Revisit only
if a physics run clears baseline.

### M3. `num_select` — the "free +0.019" is REJECTED

`main.evaluate_giwaxs_ap` calls `onnx_to_xyxy` with **no** `num_select`, so every per-epoch AP curve
is at the function default **225** (`util/postprocessing.py:32`). The config's `num_select = 150`
(`DINO_4scale_swin.py:102`) feeds `PostProcess` in `dino.py` and is **never used on the eval path**.

Δ vs 225 at each model's best alternative: ssl1 −0.006 @450 (41) / **+0.019** @900 (organic);
lr4e5 −0.001 / +0.004; physics4 **−0.038** / +0.005; physics5 **−0.034** / +0.002. Section E's ssl1
gain reproduces exactly but is ssl1-only, costs 0.006 on 41 even there, and is negative on both
gates for both physics models. Below 225 is worse everywhere. **225 is a joint optimum; leave it.**

Why it splits the models: kept boxes/frame at 225→450→900 against 41.0 GT on 41 — ssl1 79→86→86 and
lr4e5 82→83→83 *saturate* (the score cut binds, the cap is slack); physics4 137→274→524 and
physics5 149→272→520 never do. Raising the cap on them adds +135.8 FP/frame of which only +1.0 is
above score 0.5, so ~134 of 136 land in the 0.1–0.3 band — and score thresholds are not recall
levels, so those depress precision across the MID-recall region where most of the AP area is.
`_interpolate_precisions` cannot rescue it because everything to the right is worse.

A second, smaller mechanism is real (`diagnostics/capsplit.py`): `QMatcher` +
`linear_sum_assignment` over |Δq| (`util/matchers.py:31`) **never looks at the scores**, so extra
candidates can hand a GT to a geometrically closer, lower-scored box and demote the previous holder
to a high-score FP. physics4 on 41, 225→450: 98 of 1437 GT (6.8%) move down, median drop 0.283,
≈2.4 displaced boxes/frame — about the whole +1.0 in FP>0.5. Controls: ssl1 17/1460, lr4e5 3/1492.

### M4. What was queued

`dino_physics4_dn400_1` (job 2882738, config `DINO_4scale_swin_physics4_dn400.py`, sbatch
`backbone_curation/ssl/run_detector_dn400.sbatch`). ONE VARIABLE vs `dino_physics4_1`: `dn_number`
100 → 400, verified by diffing the resolved configs.

`prepare_for_cdn` (`models/dino/dn_components.py:43`) turns `dn_number` into GROUPS as
`dn_number // max_gt_in_batch`. Contrastive denoising — reconstruct the box from a noised copy,
reject the more-noised negative — is precisely the mechanism that teaches one box per object, and
our ~50–70-box frames train at **1–2 groups** where COCO's ~7-object images give ~14.
`dn_number = 400` restores ~6.

**CAVEAT, written into the config header too:** the legacy sim is starved identically (~46
boxes/frame), so this may be a general DINO gain rather than a physics-specific fix. The legacy-sim
control arm (lr4e5 recipe, `dn_number=400`) is deliberately **not queued** — no free a100-galvani —
and a positive result here does NOT establish that the hedging is what got fixed.

### M5. Caveats on all of the above

- organic is **8 frames / 817 GT boxes**. A few percent there is noise; the +0.20 recall is not.
- physics5 was at epoch 240, inside the pre-280 zone where run ranking correlates only ρ = 0.49.
  Do not rank it against physics4 at ep434.
- recall/precision here are the matcher's assignment counts at score > 0.1 — a different quantity
  from `ap_total`.
- the bank is still built `--no-exclusions` (section I), so every physics AP stays PROVISIONAL.

### M6. Three levers killed the same day

- **`num_select`** — M3 above. 225 is a joint optimum.
- **polar / quazipolar composition** (`diagnostics/geomcost.py`, `diagnostics/geomreal.py`). Both
  simulators draw the dark area 50/50 between a polar and a quazipolar wedge
  (`simulation.py:799`), and DINO evaluation of real data is always standard polar, so half of
  every training stream is a geometry the eval never contains. It costs nothing. On its own
  simulator lr4e5 scores **0.9990 polar / 0.9976 quazipolar** and physics4 **0.8175 / 0.8560** (the
  latter confounded: quazipolar frames carry 39.9 GT vs 57.6, because the wedge eats peaks). On
  real 41, switching eval to `PREPROCESSING_QUAZIPOLAR` (`exp_preprocess.py:303`, a genuine
  coordinate remap, NOT the simulator's dark-area wedge) costs lr4e5 −0.015, ssl1 −0.006, physics4
  −0.032, physics5 −0.031 and leaves the ranking intact. **organic could not be tested**:
  `PyGIDDataset` has no quazipolar GT transform — only `H5GIWAXSDataset` does
  (`labeleddataset.py:131`) — so it would remap the image and leave the labels in polar
  coordinates. Port that transform before anyone retries this on organic.
- **physics-sim under-labeling** (`diagnostics/underlabel.py`). Hypothesis: physics4 keeps 96.3
  boxes for 57.6 GT on its OWN simulator (precision 0.507) where lr4e5 keeps 46.8 for 44.7 (0.954),
  so perhaps the simulator renders signal it does not label, teaching the model to fire on
  unlabeled structure. **Rejected.** Unmatched FPs touching no GT have median intensity **0.633**,
  *below* a mask-aware radius-matched control at **0.699** (matched TPs 0.755). No
  rendered-but-unlabeled signal.

  A methodological note worth keeping: the first version of that probe sampled the control angle
  uniformly over the frame height. That lands in the dark-area wedge ~45% of the time, deflated the
  control to 0.542 and produced a confident FALSE POSITIVE (FP 0.834 > TP 0.772 > control 0.542)
  which the rendered frames then contradicted. **Mask-restrict every background control in polar
  frames.**

### M7. What the negative results leave

physics4's own-sim FP split is **dup 4.0 / near-miss 33.4 / background 8.0 per frame** — 74%
near-misses, the same duplicate spray measured on real data. The hedging is therefore present on
the distribution the model trains on: not a domain-transfer artifact, not a data bug, not a
geometry mismatch. That is what `dino_physics4_dn400_1` (M4) targets.

The one lever held back deliberately is **mixing the simulators** (`physics_sim_fraction = 0.5`).
The two fail in opposite directions — physics wins organic recall (0.752 vs 0.681; 0.886 vs 0.687
uncapped), legacy wins 41 and precision decisively (own-sim 0.954 vs 0.507) — so a mix tests
whether the failure modes cancel rather than compound. One config line, but **not to be queued
until physics5 and physics6 report**: the right physics arm to mix may be physics6's frame-type
mixture rather than physics4's composition.


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


## Heatmap-first detector (branch `heatmap`, started 2026-10-08)

Standalone alternative to the DINO decoder/matcher/loss, reusing the SSL backbone, the simulator and the labeled eval
files. Code: `models/heatmap_head.py`, `heatmap/` (targets_loss, train, evaluation, evaluate, visualize, plot_runs,
time_step, local_smoke, sbatch/). Nothing in the DINO code paths is modified. Question: does it beat ssl1 on close pairs
(< 5 px apart), where nine box-side mechanisms failed?

**Design.** Frozen SimMIM swin-L (48x6) -> FPN to a stride-2 map (256x512) fused with a small image stem. Outputs: 2 class
heatmaps (segment, ring) + 4 regression channels (dx, dy sub-cell offset; log w, log h). No angle: boxes are axis-aligned.
Targets: small Gaussian at each box centre cell (sigma = size/6 clipped 1..4 px), regression on the 3x3 cells around each
centre. Loss: CenterNet focal on the heatmaps + L1 on the regression (`heat + 4 x reg`). Decode: 3x3 max-pool peak
picking, top 225, score = heat (replaces NMS); also scored through the shared class-aware `filter_boxes` (`+nms`).
Training: live simulator (box convention 2.80/1.30), batch 4, 250 steps/epoch (1000 images), AdamW lr 3e-4 on the head,
60 epochs, lr drop x0.1 at 45, fp32, 1,228,582 trainable parameters. It runs on an RTX 5070 (colorbox1): network step
0.137 s/img, simulator 0.007 s/img (5% of a step), so a bigger batch buys no speed.

**Run 1: `hm_simmim_frozen` (60 epochs, final epoch 59).** Single model, same Evaluator as `--eval`.

| set | native AP | +nms AP | recall >0.3 (seg / ring) | precision >0.3 | recall NN<5px |
|---|---|---|---|---|---|
| organic | 0.599 | 0.612 | 0.580 (0.586 / 0.433) | 0.773 | 0.438 (n=121) |
| 41 | 0.578 | 0.667 | 0.651 (0.697 / 0.586) | 0.710 | 0.370 (n=73) |

**Same-code comparison (job 2949416, `heatmap/evaluate.py`, final checkpoints, 2026-10-08).** ssl1 reproduces its recorded
numbers (organic AP 0.5683, recall 0.537, precision 0.841), so the pipeline is validated. Heatmap = `hm+nms`.

| | organic AP | 41 AP | organic recall/prec >0.3 | 41 recall/prec >0.3 | ring recall >0.3 (org / 41) | ring recall >0.1 (org / 41) |
|---|---|---|---|---|---|---|
| heatmap (frozen SimMIM) | 0.612 | 0.667 | 0.580 / 0.773 (FP 139) | 0.651 / 0.710 (FP 447) | 0.433 / 0.586 | 0.800 / 0.941 |
| ssl1 | 0.568 | 0.744 | 0.537 / 0.841 (FP 83) | 0.772 / 0.705 (FP 543) | 0.733 / 0.853 | 0.800 / 0.938 |
| dino_lr4e5_1 | 0.622 | 0.763 | 0.592 / 0.827 (FP 101) | 0.798 / 0.707 (FP 555) | 0.700 / 0.829 | 0.767 / 0.932 |

Recall at score > 0.3 by nearest-neighbour (Euclid) distance of the GT peak (n peaks in brackets):

| | organic <5 (121) | organic 5-10 (85) | organic >10 (581) | 41 <5 (73) | 41 5-10 (62) | 41 >10 (846) |
|---|---|---|---|---|---|---|
| heatmap | 0.438 | 0.424 | 0.640 | 0.370 | 0.629 | 0.733 |
| ssl1 | 0.372 | 0.353 | 0.589 | 0.370 | 0.452 | 0.766 |
| dino_lr4e5_1 | 0.388 | 0.447 | 0.651 | 0.397 | 0.677 | 0.818 |

Reading: organic AP is above ssl1 (+0.044) and 0.010 below dino_lr4e5_1; 41 AP is 0.077 / 0.096 below. The `<5 px` bucket
on organic is +0.066 / +0.050 over ssl1 / lr4e5 (121 peaks, about 1 SE: weak evidence); on 41 it is a tie / below. In
absolute terms +0.05..0.07 of 121 peaks is about 8 peaks, ~1% of organic recall, far from the ~+0.06 prize. Rings are found
as often as DINO finds them at score > 0.1 (0.800 / 0.941) but score lower, so they drop out at > 0.3: a ring SCORE
calibration problem, which explains most of the 41 gap (rings are 41% of its objects).

**Caveats.** (1) The in-training AP is the native decode, not comparable to DINO `exp_ap` logs; use `+nms`. (2) ssl1 and
`dino_lr4e5_1` have not been rescored with `heatmap/evaluate.py`, so the `<5 px` bucket (my definition) is not
comparable to the earlier ssl1 figures (different peak sets). (3) The 41 chi-gap `<5 px` bucket holds only 2 peaks; use the
Euclidean bucket (73). (4) The interrupted run was resumed twice (simulator reseeded), so not bit-identical to an
uninterrupted run. (5) head lr 3e-4 is an untuned guess (DINO uses 1e-5..4e-5).

**Control arm `hm_random_frozen_2.80_1.30` (frozen random-init backbone, same recipe), final.** Means over epochs 55-59,
native: organic AP 0.563 (SimMIM 0.601), 41 AP 0.365 (SimMIM 0.578); final train loss 1.78 vs ~1.1. So in this frozen setup
the SimMIM backbone is worth ~0.04 AP on organic and ~0.21 on 41; close-pair recall (NN<5px, organic) is identical (0.438).

**Run 2: `hm_simmim_frozen_ridge_2.80_1.30` (`--ring_target ridge`, 60 epochs; otherwise identical to run 1).** Ring GT gets a
tall ridge target (sigma_y = h/6 capped 40 px); all ridge cells regress the same box; y-offset unsupervised off-centre.
Costs ~28% more time per epoch (Python target loop). Final (epoch 59), `evaluate.py`, same code as the DINO references above:

| | organic AP native / +nms | 41 AP native / +nms | ring recall >0.3 (org / 41) | 41 recall / prec >0.3 (+nms, FP) |
|---|---|---|---|---|
| ridge | 0.583 / 0.606 | 0.507 / 0.726 | 0.567 / 0.813 | 0.776 / 0.724 (496) |
| run 1 (legacy rings) | 0.599 / 0.612 | 0.578 / 0.667 | 0.433 / 0.586 | 0.651 / 0.722 (422) |
| ssl1 | - / 0.568 | - / 0.744 | 0.733 / 0.853 | 0.772 / 0.705 (543) |
| dino_lr4e5_1 | - / 0.622 | - / 0.763 | 0.700 / 0.829 | 0.798 / 0.707 (555) |

Recall at score > 0.3 by NN distance (ridge, native): organic <5px 0.421 (ssl1 0.372, lr4e5 0.388; n=121), 5-10 0.506, >10 0.661;
41 <5px 0.384 (0.370 / 0.397; n=73), 5-10 0.677, >10 0.800. Organic at >0.3 (+nms): recall 0.606, precision 0.739, FP 175 (ssl1 0.537 / 0.841 / 83).

Verdict (single seed): the ridge target is a clear win on 41 (+nms AP 0.667 -> 0.726, ring recall 0.586 -> 0.813, closing most of the
gap to DINO: -0.018 vs ssl1, -0.037 vs lr4e5) and neutral on organic (0.612 -> 0.606, inside noise). The ridge model emits duplicate
peaks along rings: native 41 AP 0.507 vs 0.726 with the class-aware NMS, so the +nms number is the deployed one. KEY QUESTION
(< 5 px close pairs): NOT answered in the heatmap's favour -- organic +0.03..0.05 over the DINO models (4-6 of 121 peaks, under 1 SE),
41 a tie. The heatmap does not close the close-pair gap; the prize (~+0.06 recall) is not moved. NEGATIVE on the headline hypothesis,
POSITIVE on "a frozen-backbone heatmap with a 1.2M-parameter head reaches DINO's neighbourhood".

**Evaluation convention (2026-10-08, user decision): headline = score floor 0.1 + top-225, as in the DINO `--eval` path.**
A no-floor evaluator was tried (`HM_SCORE_FLOOR=0`, files `evaluate_final_nofloor.txt`): it did NOT raise AP (organic +-0.001, 41 -0.014,
control +0.001 / 41 -0.001...), because the repo evaluator's Hungarian IoU matching ignores scores, so extra low-score boxes steal
matches. Score sweeps (`hm_runs/score_sweeps/`): at top-K 225 floors 0-0.02 are identical and 0.1-0.2 is best; at top-K 900 floors < 0.05
collapse AP (ridge 41: 0.453 at floor 0, 0.711 at 0.1). Best floor/K per run (tuned on the eval sets, so optimistic): ridge organic 0.625
(K 900, floor 0.1) / 41 0.736 (floor 0.2); main 0.616 / 0.667; control 0.587 / 0.447. The 0.3 cut applies only to drawn boxes and the
recall/precision lines, never inside the evaluator. All AP numbers above are floor-0.1 / top-225 unless stated.

**Overnight runs (2026-10-08/09, colorbox1, all ridge target, TF32, finished 05:18 CEST, no NaN/abort; single seed each).**
All numbers floor 0.1 + top-225, final epoch, `evaluate_final.txt`. `+nms` AP organic / 41 (native in brackets):

| run | setup | organic +nms | 41 +nms | ring recall >0.3 (org / 41) | recall NN<5px >0.3 (org / 41) |
|---|---|---|---|---|---|
| ridge (run 2) | SimMIM, lr 3e-4, bs 4, 60 ep | 0.6057 (0.5834) | 0.7259 (0.5071) | 0.567 / 0.805 | 0.421 / 0.384 |
| A | SimMIM, lr 1e-4, 60 ep | 0.5848 (0.5583) | 0.6956 (0.4606) | - | 0.446 / 0.370 |
| B | boxconv1 backbone (frozen), lr 3e-4, 60 ep | 0.5732 (0.5488) | 0.7489 (0.4952) | 0.633 / 0.921 | 0.339 / 0.397 |
| C | SimMIM, lr 4.2e-4, bs 8, 120 ep (lr drops 90, 112) | 0.6237 (0.5981) | 0.7178 (0.4807) | 0.533 / 0.819 | 0.463 / 0.342 |
| ssl1 (DINO) | | 0.568 | 0.744 | 0.733 / 0.853 | 0.372 / 0.370 |
| dino_lr4e5_1 | | 0.622 | 0.763 | 0.700 / 0.829 | 0.388 / 0.397 |
| dino_boxconv1 | | 0.588 | 0.752 | - | 0.339 / 0.370 |

lr pick for C: the lr-1e-4 run (A) had mean +nms AP 0.6402 vs ridge 0.6658, so the queue chose 4.2e-4 (1.4x the 3e-4 baseline) for the long run.
Lower lr was worse, so the untuned-lr caveat is partly answered (1e-4 loses); lr above 3e-4 was only tried inside the long run, which also
changed batch and epochs, so lr and schedule are confounded there.

Reading: (1) Organic: C is the best heatmap run and matches the best DINO run (0.6237 vs dino_lr4e5_1 0.622, ssl1 0.568, boxconv1 0.588),
a tie with lr4e5 inside noise. (2) 41: no heatmap run beats DINO; best is B 0.7489 (boxconv1 backbone) vs lr4e5 0.763, ssl1 0.744,
boxconv1 DINO 0.752; C 0.7178 is 0.008 below its 60-epoch ridge predecessor, so longer training did not help on 41. (3) The SimMIM-to-boxconv1
backbone swap helps 41 (+0.023 over ridge, ring recall >0.3 0.805 -> 0.921) and hurts organic (-0.032). (4) Close pairs <5px at >0.3:
C organic 0.463 (n=121) vs lr4e5 0.388 / ssl1 0.372 / boxconv1 DINO 0.339: +0.075..0.124, i.e. 9-15 of 121 peaks (about 1.7 SE vs lr4e5),
the best close-pair figure of any heatmap run, but on 41 C is 0.342 vs 0.370-0.397 (n=73, below DINO). Combined: organic mildly in the heatmap's
favour, 41 against, and both are single seeds. The headline hypothesis is NOT confirmed: the < 5 px bucket still recalls under half of the
peaks and the ~+0.06 recall prize is not clearly moved. C's native decode still trails +nms (organic 0.598 vs 0.624, 41 0.481 vs 0.718).
Organic ring recall at >0.3 fell with the long run (0.533 vs 0.567), so ring calibration on organic is still below DINO (0.70-0.73).

C score sweep (`hm_runs/.../score_sweep/`, AP at top-K 225): organic floors 0-0.05 0.625, 0.1 0.6237, 0.2 0.606, 0.3 0.565; 41 floors 0-0.05 0.7045-0.705,
0.1 0.7178, 0.2 0.7274, 0.3 0.698. At top-K 900 floors < 0.05 collapse AP (organic 0.532, 41 0.517 at floor 0). Best F1 0.670 (organic, thr 0.3), 0.747 (41, thr 0.4).
Images (+nms only) for A, B, C are in `hm_runs/<run>/images/nms/`. Run B's sweep OOM'd (GPU shared with C) and was re-run by hand.

**q-dependent (gradient) score threshold on existing checkpoints (2026-10-09, `heatmap/q_threshold.py`, no training).**
thr(x) = t0 + (t1 - t0) x/1024 on the box-centre column (t0 = low q, t1 = high q; t0 = t1 is the constant control). Two uses:
`filter` (keep score > thr(x), scores unchanged) and `norm` (also rescale s' = (s - thr)/(1 - thr)). Files: `hm_runs/q_threshold/hm_{C,ridge,B}_q_threshold.txt/.json`.
Best AP (organic / 41) vs constant 0.1 and the best constant:

| run | constant 0.1 | best gradient | best constant (41) |
|---|---|---|---|
| C (long) | 0.6237 / 0.7178 | 0.6336 (0.15->0.03) / 0.7279 (0.3->0.15) | 0.7274 (0.2) |
| ridge | 0.6057 / 0.7259 | 0.6145 (0.15->0.03) / 0.7361 (0.2->0.15) | 0.7357 (0.2) |
| B (boxconv1) | 0.5732 / 0.7489 | 0.5876 (0.15->0.03) / 0.7653 (0.4->0.1) | 0.7592 (0.3) |

Reading: (1) gains are +0.009..+0.014 AP on organic and on 41 the gradient is within 0.0005 of the best constant (C, ridge) or +0.006 over constant 0.3 (B).
(2) Cross-set (best on one set, scored on the other) is below the constant 0.1 headline in AP for every run, e.g. ridge 0.7210/0.7221 on 41 vs 0.7259 and
0.5855/0.5937 on organic vs 0.6057: the optima are tuned to the set. (3) Premise only half true: faint GT share rises with q (organic 0.20/0.33/0.48), organic
recall at floor 0.1 falls with q (C 0.835/0.763/0.603; 41 does not), but the TP score does not (C median TP 0.475/0.551/0.487 vs FP ~0.2): the high-q loss is
peaks never found, not found peaks scored low. (4) The two sets prefer different constant floors (organic ~0.1-0.15, 41 0.2-0.3; B 0.3-0.4), a set-level
calibration difference, not a q effect. The simulator flips images left-right, so the net cannot learn q from position and a position-based correction is the only way
to use it. Headline stays floor 0.1 + top-225; the tuned numbers are optimistic (tuned on the eval sets).

**Later options and runs (2026-10-09/10).** New code, all default-off (default path unchanged):
`--chan {he,he_mask,full,contrast}` extra stem channels (swin sees channel 0 only), `--zero_invalid` (masked pixels = 0 in every input, train and eval),
head size `--dim/--tower_ch/--tower_depth/--stem_ch` (wide 256/128/4/64 = 4.76 M trainable vs 1.23 M), `--out_stride 1` (512x1024 map, bs 2 x 500 steps because bs 4 OOMs),
`--ring_head [--ring_head_stride 8|16]` (ring-only dilated head on the merged FPN map, decode modes fine / coarse_rings (pre-declared headline) / union),
`heatmap/predict.py` (boxes for unlabeled h5), `heatmap/summarize_runs.py` (Markdown results table), `heatmap/make_architecture_pdf.py`, eval every 2 epochs via `eval_interval.txt`.

Results so far (+nms, floor 0.1, top-225; organic / 41; single seeds, eval-to-eval noise about +-0.03):
- chanfull (4 channels, TF32): 0.5792 / 0.7048 vs ridge 0.6057 / 0.7259 (-0.027 / -0.021; ridge was fp32, chanfull TF32, so confounded). NN<5px at >0.3: 0.430 / 0.397 (ridge 0.421 / 0.384). chanhemask stopped at epoch 2.
- stride 1 (`--out_stride 1`, bs 2 x 500, 196 s/epoch vs about 121 for stride 2, peak 10.1 GB): 0.6106 / 0.7298 = within noise of the stride-2 ridge run. Score>0.3: organic recall 0.546 prec 0.812 ring recall 0.500;
  41 recall 0.745 prec 0.765 ring recall 0.784. NN<5px at >0.3: 0.438 (n=121) / 0.397 (n=73) vs ridge 0.421 / 0.384 and DINO ssl1 0.372 / 0.370, lr4e5 0.388 / 0.397: +0.017 / +0.013 over ridge, i.e. 2 and 1 peaks, not significant.
  Native decode is the lowest of all ridge-family runs on 41 (0.4381 vs 0.4606-0.5784), so NMS helps stride 1 more (+0.292 vs +0.219 for ridge); plain reading, no explanation. Organic ring recall >0.3 is 0.500, lower than the earlier 0.53-0.57.
  Verdict: output stride 1 does not help AP and does not clearly move the close-pair bucket, at 1.6x the training time.
- contrast channels (`--chan contrast`: log+HE, plain log, log+CLAHE 4@16x16, as stem channels beside the swin's channel 0; bs 4 x 250, 121 s/epoch, TF32): 0.5873 / 0.7236 = no better than plain ridge (organic -0.018, 41 -0.002, both within noise); below stride 1 on both sets.
  Score>0.3: organic recall 0.600 prec 0.697 ring recall 0.533 (ridge 0.606 / 0.739); 41 recall 0.805 prec 0.668 ring recall 0.858 (ridge 0.776 / 0.724): more recall on 41, lower precision. NN<5px at >0.3: 0.446 (n=121) / 0.384 (n=73) vs ridge 0.421 / 0.384.
  Native 41 AP 0.5434 (higher than ridge 0.5071). 41 AP at floor 0.2 is 0.7396 (headline floor 0.1: 0.7236, convention unchanged). Compare with chanfull: both multi-channel variants fail to beat the plain input; the plain-log channel was a possible out-of-distribution risk on organic and organic does end lower.
- TF32 plain control (`--chan he --tf32`, same recipe and the same seed 42 as the fp32 ridge run (both args.json checked); bs 4 x 250, 117 s/epoch): 0.6320 / 0.7260, the highest organic +nms AP of all heatmap runs so far (C 0.6237, ridge 0.6057), 41 equal to ridge (0.7259).
  Versus the fp32 ridge run: organic +0.026, 41 +0.0001. Two single runs of the same recipe therefore differ by 0.026 on organic; Seed was the same (42), so the seed is not the cause; the cause (TF32, non-deterministic GPU ops, code version at run time, ...) was not tested. A different seed (TF32 seed 1) gave 0.5989 / 0.7225: organic spread across seeds 0.033, 41 0.0035.
  Score>0.3: organic recall 0.605 prec 0.758 ring recall 0.500; 41 recall 0.779 prec 0.725 ring recall 0.818. NN<5px at >0.3: 0.430 / 0.384 (ridge 0.421 / 0.384). Native 41 AP 0.4947, native organic 0.6097.
  Consequence: differences below about 0.03 between single runs (stride 1, contrast, chanfull) are inside the run-to-run spread and should not be read as effects.
- zeroinv (`--chan he --zero_invalid`, bs 4 x 250, 117 s/epoch): 0.6097 / 0.7169. Versus the TF32 plain control (0.6320 / 0.7260): organic -0.022, 41 -0.009; versus fp32 ridge (0.6057 / 0.7259): +0.004 / -0.009. Both differences are inside the ~0.03 spread between the two plain runs: no clear +nms gain from zeroing invalid pixels.
  Score>0.3: organic recall 0.583 prec 0.746 ring recall 0.567; 41 recall 0.793 prec 0.690 ring recall 0.851. NN<5px at >0.3: 0.455 / 0.356 (plain TF32 0.430 / 0.384). Native 41 AP 0.5529 (plain TF32 0.4947, ridge 0.5071), native organic 0.5839; the native-decode 41 gain is unexplained.
  41 AP at floor 0.2 is 0.7262 (headline floor 0.1: 0.7169).
- wide head (`--dim 256 --tower_ch 128 --tower_depth 4 --stem_ch 64`, 4.76 M trainable vs 1.23 M, bs 4 x 250, 190 s/epoch, peak 11.2 GB): 0.6109 / 0.7184. Versus the TF32 plain control (0.6320 / 0.7260): -0.021 / -0.008; versus fp32 ridge: +0.005 / -0.008. All inside the ~0.03 single-run spread: a 4x larger head gives no gain.
  Score>0.3: organic recall 0.586 prec 0.769 ring recall 0.467; 41 recall 0.766 prec 0.688 ring recall 0.776. NN<5px at >0.3: 0.446 (n=121) / 0.397 (n=73). Native 41 AP 0.5200, native organic 0.5907. 41 AP at floor 0.2: 0.7267.
  The epoch-58 evaluation (0.621 / 0.722) was above the final one (0.6109 / 0.7184) for the same model; cause not identified.
- ring head (`--ring_head`, stride 8, bs 4 x 250, 120 s/epoch, peak 5.8 GB, TF32). Pre-declared headline decode = coarse_rings: 0.6075 / 0.7155. Versus the TF32 plain control (0.6320 / 0.7260): -0.025 / -0.010; versus fp32 ridge (0.6057 / 0.7259): +0.002 / -0.010. Inside the ~0.03 spread: no gain.
  Other decodes (secondary, not the declared headline): fine 0.6016 / 0.7123; union 0.5943 / 0.7484 (union lifts 41 to near the DINO refs but lowers organic).
  Score>0.3 (coarse_rings): organic recall 0.621 (seg 0.619, ring 0.667) prec 0.686; 41 recall 0.804 (seg 0.759, ring 0.866) prec 0.654. NN<5px: organic 0.455 (n=121), 41 0.342 (n=73, the lowest 41 value of all runs). Native 41 AP 0.6503 (highest native 41 so far; cause untested), native organic 0.5915.
  41 AP at floor 0.2 is 0.7399 (headline floor 0.1: 0.7155). train.log epoch-59 line shows 41 0.7142 vs 0.7155 in evaluate_final; difference not investigated.
- Regression check: evaluate.py on the old fp32 ridge checkpoint reproduces 0.6057 / 0.7259 exactly. Step timing (TF32, plots/time_step_*.txt): simulator is about 4-5% of a step; wide head 0.19 s/img; stride 1 0.195 s/img at bs 2, OOM at bs 4.

Queued on colorbox1 (weekend queue, started 02:09): he_tf32 seed 1, seed 2, ring head stride 16, lr 5e-4, 120 epochs (lr drop 90); expected end about 15:00. On galvani (job 2952793, pending): run C recipe with the swin unfrozen (lr 1e-5, drop path 0), output in datasets/DINO_HEATMAP.

Seeds of the plain TF32 recipe (`--chan he --tf32`, +nms floor 0.1 top-225; organic / 41; native 41): seed 42 0.6320 / 0.7260 / 0.4947; seed 1 0.5989 / 0.7225 / 0.4825; seed 2 0.6027 / 0.7266 / 0.5245. Mean 0.611 / 0.725; range organic 0.033, 41 0.004, native 41 0.042.
Seed 2 score>0.3: organic recall 0.601 prec 0.773 ring recall 0.567, NN<5px 0.463 (n=121); 41 recall 0.772 prec 0.738 ring recall 0.809, NN<5px 0.356 (n=73). Seed 1: organic NN<5px 0.430, 41 0.397.
Reading: all organic values of the variants (wide 0.6109, ring head 0.6075, zeroinv 0.6097, fp32 ridge 0.6057) lie inside the seed range. On 41 the wide head, ring head (coarse_rings) and zeroinv sit 0.007-0.010 below the seed mean against a seed range of 0.004: slightly larger than the seed spread, but three seeds do not establish it. Native 41 of ring head (0.6503) and zeroinv (0.5529) exceed the seed range 0.4825-0.5245; cause untested.

Ring head stride 16 (`--ring_head --ring_head_stride 16`, 119 s/epoch, peak 5.8 GB, TF32). Declared headline decode (coarse_rings), +nms floor 0.1 top-225: organic 0.5980 / 41 0.6573. Versus ring head stride 8 (0.6075 / 0.7155): -0.010 / -0.058; versus the plain TF32 seeds (organic 0.5989-0.6320, 41 0.7225-0.7266): organic at the low end of the range, 41 0.065-0.069 below it (seed range on 41 only 0.004).
Other decodes of the same model: fine 0.5884 / 0.7161, union 0.5843 / 0.7300. The 41 loss is specific to coarse_rings (fine and union are near stride 8's); hypothesis that the stride-16 coarse ring map localises ring peaks worse than stride 8: untested.
Score>0.3 (coarse_rings): organic recall 0.613 (seg 0.609, ring 0.733) prec 0.723; 41 recall 0.721 (seg 0.734, ring 0.703) prec 0.675. NN<5px: organic 0.455 (n=121), 41 0.342 (n=73). Native 41 0.6390, native organic 0.5947. 41 AP at floor 0.2 is 0.6778.

Unfrozen swin on galvani (job 2952793, a100-galvani cn202, A100 40 GB; run C recipe: `--bb simmim1 --ring_target ridge --tf32 --lr 4.2e-4 --lr_drops 90 112 --epochs 120 --bs 8 --steps_per_epoch 125 --unfreeze --lr_backbone 1e-5 --bb_drop_path 0 --grad_ckpt`). Output: datasets/DINO_HEATMAP/hm_simmim_UNFROZEN_ridge_long_tf32_2.80_1.30 (evaluate_final.txt). Smoke at bs 8: plain OOM (39 GB), with `--grad_ckpt` OK (1.42 s/step); training 169 s/epoch, 21.6 GB, total 5 h 58 min.
Final (+nms floor 0.1, top-225): organic 0.6146 / 41 0.7531. Versus the frozen run C (0.6237 / 0.7178): -0.009 / +0.035; versus the plain TF32 seeds (organic 0.5989-0.6320, 41 0.7225-0.7266): organic inside the range, 41 +0.027 above the best seed (seed range on 41 only 0.004). Single run, one seed.
Score>0.3: organic recall 0.592 (seg 0.591, ring 0.633) prec 0.818; 41 recall 0.827 (seg 0.779, ring 0.895) prec 0.697. NN<5px: organic 0.430 (n=121), 41 0.425 (n=73; highest 41 value among the heatmap runs, ssl1 0.370, lr 4e-5 0.397). Native decode: organic 0.5943, 41 0.6195.
Last evaluations in train.log: epoch 116 0.6155 / 0.7537, final 0.6146 / 0.7531 (stable after the second lr drop).

lr 5e-4 (`--chan he --tf32 --lr 5e-4`, otherwise the plain recipe, 117 s/epoch): +nms floor 0.1 top-225 organic 0.6109 / 41 0.7168. Organic equals the three-seed mean (0.611, range 0.5989-0.6320); 41 is 0.006 below the lowest seed (0.7225) and 0.008 below the mean, slightly outside the 41 seed range of 0.004 but a single run: no sign that a higher lr helps. With lr 1e-4 (run A, 0.5848 / 0.6956), lr 3e-4 remains the choice.
Score>0.3: organic recall 0.589 (seg 0.591, ring 0.533) prec 0.794; 41 recall 0.759 (seg 0.731, ring 0.799) prec 0.717. NN<5px: organic 0.413 (n=121), 41 0.397 (n=73). Native 41 0.4606 (seeds 0.4825-0.5245), native organic 0.5873. 41 AP at floor 0.2: 0.7176.

120 epochs on the plain recipe (`--chan he --tf32 --epochs 120 --lr_drop 90`, bs 4 x 250, lr 3e-4, 117 s/epoch, 4 h): +nms floor 0.1 top-225 organic 0.6403 / 41 0.7121. Organic is the highest of all runs (+0.008 over the best seed run, +0.029 over the seed mean 0.611) and 41 is below all three 60-epoch seeds (0.7225-0.7266) and below run C (0.7178); each difference is about the size of the run-to-run noise, so on one run 120 epochs is not shown to be better. After the lr drop the 41 value stays flat at 0.70-0.73 (epoch 100 0.718, 116 0.720, 118 0.697, final 0.712).
Score>0.3: organic recall 0.635 (seg 0.638, ring 0.567) prec 0.737; 41 recall 0.789 (seg 0.754, ring 0.838) prec 0.662. NN<5px: organic 0.455 (n=121), 41 0.370 (n=73). Native 41 0.4819, native organic 0.6205. 41 AP at floor 0.2: 0.7148.
Queue state: the whole colorbox1 queue (final, extra and weekend queues) and the galvani unfrozen run are done; no job is running. The native-decode images of the three early runs have not been deleted (question to the user unanswered).

Noted idea (user, 2026-10-10): once the cluster is working again, try longer unfrozen runs on galvani (more epochs than the 120-epoch unfrozen run). Not queued. Planned repeats: unfrozen swin seeds 1 and 2 on galvani (not yet submitted, Slurm refused submissions on 2026-10-10), run C recipe seeds 1 and 2 on colorbox1 (running).

Run C recipe replicates (bs 8 x 125, 120 epochs, lr 4.2e-4, lr drops 90/112, frozen swin, TF32, 118 s/epoch, colorbox1; seeds 1 and 2 added to run C = seed 42). +nms floor 0.1 top-225, organic / 41:
 seed 42 (run C) 0.6237 / 0.7178; seed 1 0.6169 / 0.7302; seed 2 0.6138 / 0.7286. Mean 0.618 / 0.726; range 0.010 / 0.012.
 Native AP organic 0.5981 / 0.5869 / 0.6006; native 41 0.4807 / 0.4387 / 0.4966 (range 0.058).
 Score>0.3 (organic; 41): seed 1 recall 0.590 prec 0.755 ring 0.500; 0.776 / 0.730 / 0.833. Seed 2 recall 0.614 prec 0.722 ring 0.533; 0.793 / 0.703 / 0.836. NN<5px: seed 1 0.455 / 0.397, seed 2 0.463 / 0.356 (run C 0.463 / 0.342).
 41 AP at floor 0.2: seed 1 0.7336, seed 2 0.7377. GPU memory by nvidia-smi: seed 1 8392 MiB, seed 2 10475 MiB with the same command (cause untested).
Versus the 60-epoch bs 4 lr 3e-4 recipe (mean 0.611 / 0.725, range 0.033 / 0.004): means equal within noise (organic +0.007, 41 +0.001): bs 8 / 120 epochs / lr 4.2e-4 gives no mean gain. The 120-epoch bs 4 lr 3e-4 run (0.6403 / 0.7121) is one run.
Frozen 120-epoch recipe mean 0.618 / 0.726 is the frozen reference for the unfrozen swin (first run 0.6146 / 0.7531; unfrozen seeds 1 and 2 on galvani are NOT yet queued: Slurm refused all submissions on 2026-10-10, user checks the cluster).

Blob splitting (post-processing idea, CLOSED for now by the user, 2026-10-11). Tools: heatmap/dump_dets.py (GPU, caches +nms detections, images, GT of a checkpoint), heatmap/blob_diag.py (why close-pair peaks are missed), heatmap/split_blobs.py (1 vs 2 Gaussians per segment detection by BIC, replaces the box by two). Model: plain TF32 seed 42.
 Diagnostic (score>0.3), NN<5 px bucket: organic (121 GT) hit 0.430, missed with neighbour matched 0.421, missed with a box near but neither matched 0.041, nothing near 0.107; 41 (73 GT) 0.384 / 0.452 / 0.110 / 0.055. Sim frames contain no pair closer than 5 px (110 in 5-10 px), so the simulator cannot test the case; development sets = sim and 41, organic only with --final.
 Splitter sweep (sep 1.5-8 px, amplitude ratio >= 0.3, score >= 0.2), tau 10/20/30/50/80: AP on 41 0.6941/0.6960/0.7021/0.7110/0.7177 (baseline 0.7260), sim 0.8553/0.8754/0.8894/0.9094/0.9309 (baseline 0.9752); precision at >0.3 drops everywhere (41 0.725 -> 0.689-0.716); NN<5 px recall on 41 0.397 at tau 10 (one more peak), 0.384 otherwise. 184 / 151 / 117 / 62 / 29 splits on 41 (73 GT in the bucket). Organic was not run. No tau improves AP.
 Untested reasons: pairs under 5 px may not be resolvable in the pixels; the splitter fires on single peaks; the second box may land too far off. Possible follow-up (not done): log fitted vs true separation and delta BIC for the 84 'neighbour matched' peaks.
Queue started 2026-10-11 11:43 on colorbox1 (heatmap/queue_coord_unfrozen_b.sh): coord channels seed 1, unfrozen swin seeds 1 and 2 (bs 4, 60 epochs), run B seeds 1-3. `--chan coord` = HE + x/W + y/H into the stem.

Coordinate channels, seed 1 (`--chan coord`, frozen swin, plain 60-epoch TF32 recipe, bs 4 x 250, lr 3e-4, 117 s/epoch, colorbox1, peak GPU 5912 MiB): +nms floor 0.1 top-225 organic 0.6073 / 41 0.6922. Native: organic 0.5888, 41 0.5185.
 Plain-recipe seeds 42/1/2: organic 0.6320/0.5989/0.6027 (mean 0.611): coord is inside the range, 0.004 below the mean. 41: 0.7260/0.7225/0.7266 (mean 0.725, range 0.004): coord is 0.033 below the lowest seed. Native 41 seeds 0.4947/0.4825/0.5245: coord inside.
 Score>0.3: organic recall 0.581 prec 0.756 ring 0.500; 41 recall 0.779 prec 0.697 ring 0.826. NN<5 px: organic 0.455 (n=121), 41 0.397 (n=73). Sweep (top-225): organic best 0.6073 at floor 0.1 (top-900 0.6169); 41 best 0.7262 at floor 0.2 (0.6922 at 0.1).
 The 41 gap at the headline floor is largely a floor effect (at floor 0.2 coord equals the plain seeds at floor 0.1); not investigated why. One seed. No sign that the coordinate channels help; at the headline floor they look worse on 41.

Unfrozen swin, seed 1, colorbox1 (`--chan he --unfreeze --lr_backbone 1e-5 --bb_drop_path 0 --grad_ckpt`, plain 60-epoch TF32 recipe, bs 4 x 250, head lr 3e-4, 348 s/epoch, 6.0 h, peak GPU 11502 of 12227 MiB, no OOM/NaN; evaluate.py on the checkpoint loaded the stored backbone without error, its output equals the epoch-59 in-training eval): +nms floor 0.1 top-225 organic 0.6219 / 41 0.7684. Native: organic 0.5934, 41 0.6090.
 Plain-recipe seeds 42/1/2: organic 0.6320/0.5989/0.6027 (mean 0.611): unfrozen is inside the range, 0.011 above the mean. 41: 0.7260/0.7225/0.7266 (mean 0.725, range 0.004): unfrozen is 0.042 above the best seed. Native 41 seeds 0.4947/0.4825/0.5245: unfrozen 0.084 above the best.
 Score>0.3: organic recall 0.632 prec 0.744 ring 0.667; 41 recall 0.821 prec 0.720 ring 0.901. NN<5 px: organic 0.438 (n=121), 41 0.425 (n=73). Sweep (top-225): organic best 0.6219 at floor 0.1 (top-900 0.6279); 41 best 0.7894 at floor 0.2 (0.7684 at 0.1; top-900 0.7653).
 In-training evals: 41 AP rose from 0.74 (epoch 18-28) to 0.7755 at epoch 48 and 0.7669 at epoch 56; organic stayed at 0.57-0.62.
 Consistent with the galvani unfrozen run (bs 8, 120 epochs, 1 run: 0.6146 / 0.7531): both single unfrozen runs are above the frozen seeds on 41 and inside their range on organic. One seed here; unfrozen seed 2 is queued. Not a like-for-like comparison with the frozen runs (196.7 M trainable parameters vs 1.2 M, 3x time per epoch, 11.5 GB GPU). Why 41 gains and organic does not: untested.

Run B replicate, seed 1 (`--bb ssl1 --bb_path boxconv1_backbone.pth`, frozen, plain 60-epoch TF32 recipe, bs 4 x 250, lr 3e-4, 117 s/epoch, colorbox1, peak GPU 5648 MiB): +nms floor 0.1 top-225 organic 0.5801 / 41 0.7493. Native: organic 0.5505, 41 0.4880.
 Run B (seed 42): 0.5732 / 0.7489, native 0.5488 / 0.4952: seed 1 is +0.007 organic, +0.000 on 41, -0.007 native 41 (reproduces run B).
 Plain SimMIM seeds 42/1/2: organic 0.6320/0.5989/0.6027 (mean 0.611): B seed 1 is 0.019 below the lowest. 41: 0.7260/0.7225/0.7266 (range 0.004): B seed 1 is 0.023 above the highest. So the boxconv1 backbone looks better on 41 and worse on organic, in two runs; seeds 2 and 3 are queued.
 Score>0.3: organic recall 0.572 prec 0.792 ring 0.633; 41 recall 0.847 prec 0.647 ring 0.917. NN<5 px: organic 0.355 (n=121), 41 0.411 (n=73). Sweep (top-225): organic best 0.5895 at floor 0.05 (0.5801 at 0.1; top-900 0.5893); 41 best 0.7676 at floor 0.3 (0.7660 at 0.2; top-900 at 0.1 0.7406).
 Unfrozen SimMIM seed 1 (0.6219 / 0.7684) is above B on both sets.

Run B replicate, seed 2 (boxconv1 backbone, frozen, plain 60-epoch TF32 recipe, 117 s/epoch, colorbox1, peak GPU 5923 MiB): +nms floor 0.1 top-225 organic 0.5988 / 41 0.7578. Native: organic 0.5653, 41 0.4707.
 Score>0.3: organic recall 0.576 prec 0.784 ring 0.633; 41 recall 0.843 prec 0.655 ring 0.915. NN<5 px: organic 0.364 (n=121), 41 0.411 (n=73). Sweep (top-225): organic best 0.5988 at floor 0.1 (top-900 0.6103); 41 best 0.7727 at floor 0.2 (0.7578 at 0.1; top-900 0.7492).
 Run B recipe, seeds 42 / 1 / 2: organic 0.5732 / 0.5801 / 0.5988 (mean 0.584, range 0.026); 41 0.7489 / 0.7493 / 0.7578 (mean 0.752, range 0.009); native 41 0.4952 / 0.4880 / 0.4707.
 Plain SimMIM seeds 42 / 1 / 2: organic 0.6320 / 0.5989 / 0.6027 (mean 0.611); 41 0.7260 / 0.7225 / 0.7266 (mean 0.725).
 Means B minus SimMIM: organic -0.027, 41 +0.027. On organic the ranges overlap (best B 0.5988 vs second-lowest SimMIM 0.5989); on 41 they do not (lowest B 0.7489 above highest SimMIM 0.7266). 3 seeds each; B seed 3 is queued. Unfrozen SimMIM seed 1 (0.6219 / 0.7684) is above the B mean on both sets.

Real-background simulator in the heatmap branch (2026-10-12). `realsim/` = the realbkg simulator of `multi-channel-analysis` (+ the unpushed ring-appearance options of DINO run 7), a separate package; `--sim realbkg [--sim_workers N] [--realsim_preset r6|r7|r8] [--realsim_opt key=value ...]` in heatmap/train.py (default `--sim legacy` unchanged). Simulator costs ~190 ms/frame and 13 GB RAM per process, so frames come from a worker process (~215 s/epoch instead of 117 s).
 First run (r6 settings, single structure per image, 25 % rings, 25-75 reflections, plain 60-epoch recipe, stopped by the user at epoch 14): organic +nms 0.6535, 41 0.4386 (41 ring recall 0.25); 41 stalled at ~0.41-0.44 from epoch 2, as the DINO realbkg runs did (~0.45).
 Diagnosis (heatmap/miss_analysis.py, appearance_compare.py, ring_compare.py, calibrate_sim.py, debug_ring_chain.py): the model finds 21 % of real 41 rings (plain 82 %), independent of contrast; on its own simulator frames 62 %. Processed-image appearance, medians (real 41 / sim): ring amplitude 0.205 / 0.402, integrated ring SNR 20 / 62, ring FWHM 12 / 30 px, chi variation 0.18 / 0.000, box/FWHM 0.62 / 0.22; segment amplitude 0.25 / 0.09, FWHM in q 10 / 22 px; ~2 ring boxes per frame against 17. Causes found: (1) with `intensity_decades` set the rendered peak image is rescaled so its brightest pixel is 10^U(3,7.3) counts, which discards the fitted amplitude/noise values and all gains (raw ring contrast 1e1-1e5 sigma); the histogram equalisation then saturates rings to white; (2) within-frame width scatter (0.53 / 0.83) and arc stretching gave rings of 5-36 px; (3) ring exposure.
 Calibrated settings (E1/E3, sheets approved by the user): amplitude_mode fitted, intensity_decades None, ring_gain 0.4 capped at 4 sigma (ring_cap), ring_contrast_min 0.3, ring_sq_scale 1.0 / ring_w_coef 2.2, no ring stretching, width_jitter (0.2, 0.832), ring_follow_bkg (ring amplitude follows the local background along chi) with ring_chi_texture 0.05, seg_sc_scale 0.6. Calibrated medians (real 41): ring amplitude 0.23 (0.205), per-pixel visibility 1.05 (0.965), SNR 21 (20), FWHM 11 (12), box width 8.1 (7.5); segments amplitude 0.29 (0.25), box 4.7 x 29 px (4.5 x 37). Not matched: background neighbouring-pixel correlation 0.05 (real 0.19 along q, 0.34 along chi).
 E1 (rings + segments in the same frame, p_ring 0.85, ~15 ring boxes/frame, 20 % ring-free, narrow 41-style segments; plain recipe, seed 42, run time 3.6 h): final +nms floor 0.1 top-225 organic 0.6111 (plain mean 0.611), 41 0.4976 (plain mean 0.725). Score>0.3: organic recall 0.667 prec 0.653 ring 0.633, NN<5px 0.463; 41 recall 0.698 prec 0.515 ring 0.675, NN<5px 0.425. 41 ring recall rose from 0.25 (first run) to 0.675 (plain 0.82); the remaining 41 gap is false positives: segment-class FP per frame in empty background 15.8 (plain 6.0), on ring columns 7.1 (3.7), ring class elsewhere 2.3 (1.5); objects found 0.698 vs 0.779. 41 AP did not recover after the lr drop (0.49-0.53). Floor 0.2 gives 41 0.5518 (organic 0.6269). Suspect, untested: correlated noise of real backgrounds (clumps that look like faint peaks) which the near-white simulated noise never shows as negatives.
 Unfrozen swin on galvani, bs 8 x 125, 120 epochs, backbone lr 1e-5: seed 2 final organic 0.6155 / 41 0.7534 (NN<5px 0.479 / 0.397); seed 1 and the first run: 41 0.7531-0.7561; all three 41 +0.03 above the frozen recipe, organic level. A third run with backbone lr 4e-5 (the DINO value) is running.
 hm_runs layout: run folders grouped (frozen_simmim_plain, run_C_recipe, head_variants, channel_variants, backbone_variants, unfrozen, realsim, _scratch); heatmap/runs_layout.py (dry run / apply / undo); results_table.md has a "Recipes run more than once" section (mean, SD, SE, range).

Results of 2026-10-12/13 (all +nms, floor 0.1, top-225; single models; plain = frozen SimMIM plain recipe, organic 0.611 / 41 0.725 mean of 3 seeds).
 Unfrozen swin on galvani, bs 8 x 125, 120 epochs, backbone lr 4e-5 (seeds 1/2/3, `train_long_unfrozen.sbatch` with LR_BB=4e-5): organic 0.5955 / 0.6058 / 0.5891 (mean 0.597, SD 0.008, SE 0.005); 41 0.7746 / 0.7616 / 0.7824 (mean 0.773, SD 0.011, SE 0.006). Score>0.3: organic recall 0.570 prec 0.821, 41 recall 0.833 prec 0.703 ring 0.915; NN<5 px organic 0.413, 41 0.406. Same recipe at backbone lr 1e-5 (3 runs): organic 0.612 / 41 0.756. So 4e-5 is +0.017 on 41 (about 1.5 SD, suggestive) and -0.015 on organic (about 2 SD); the 41 mean is above all three DINO refs (0.744 / 0.763 / 0.752 for ssl1 / lr 4e-5 / boxconv1, the lr 4e-5 one by 0.010). Organic stays level with the DINO refs (0.568-0.622). 6 h per run on an A100 40 GB with --grad_ckpt (one seed-3 submission died after 1 s on galvani-cn232 without output; resubmitted on another node).
 Unfrozen swin on colorbox1, plain 60-epoch recipe, seed 2: organic 0.5819 / 41 0.7566 (NN<5 px 0.413 / 0.370). With seed 1 (0.6219 / 0.768): mean organic 0.602 (SE 0.020), 41 0.762 (SE 0.006), n=2 only.
 Run B (boxconv1, frozen) seed 3: organic 0.5898 / 41 0.7613 (NN<5 px 0.355 / 0.397). Four seeds: organic 0.585 (SE 0.006), 41 0.754 (SE 0.003); B minus plain: organic -0.026, 41 +0.029.
 Real-background simulator E1b (E1 + correlated background noise, bkg_noise_blur (0.0,1.2), bkg_noise_power 1.5; plain recipe, seed 42, galvani, 2 h): organic 0.6059 / 41 0.5850 (E1: 0.611 / 0.498; plain simulator 0.611 / 0.725). Score>0.3 on 41: recall 0.674, prec 0.662 (E1 0.515), FP 578 (E1 1104), ring recall 0.661; organic recall 0.643 prec 0.704; NN<5 px organic 0.455, 41 0.384. The noise correlation fixes about half of E1's false-positive excess on 41 and gives +0.087 AP; the 41 gap to the plain simulator is still 0.14. Not yet investigated which false positives remain (miss_analysis on E1b not run), nor the open gaps: detector-gap bands, fewer segments per frame than real 41, E3 (not trained).
 Tools: heatmap/runs_layout.py prints SKIP when the destination exists; summarize_runs.py has labels and a replicate group for the bblr 4e-5 seeds and E1b. Galvani text results are in heatmap/galvani_results/ (git as transport to colorbox1).

Realsim runs E5-E11 (2026-10-13/14, all frozen SimMIM, plain 60-epoch recipe, seed 42, galvani, +nms floor 0.1 top-225; plain simulator 0.611 organic / 0.725 41; full table with recall/precision/ring/FP/NN<5: heatmap/galvani_results/REALSIM_RESULTS.md, on colorbox1 hm_runs/old_convention/REALSIM_RESULTS.md; per-run text results in heatmap/galvani_results/<run>/, analysis in heatmap/galvani_results/realsim_diagnosis_2026-10-13/ with README).
 E5 (E1b + mask_keep [0,3,6,7] = the four detector masks with q_max 3.6-4.0 matching real 41's q range 2.7-3.5, + ring_x_share (0.106,0.179,0.257,0.238,0.220)): organic 0.6006 / 41 0.6213. E6 (E5 + sq_q_slope 0.6, n_oriented (2,3), spots_cap (25,120); ~15 rings and ~21 segments per frame, real 41 17 / 24): 0.6266 / 0.6104. E7 (E6 + bkg_flat_q 1.0 + ring_rel_shuffle 0.5; image statistics now match 41): 0.5581 / 0.5500, worse, 10.4 ring FPs per 41 frame. E8 (E6 appearance, sparse: n_powder (1,1), rings_cap (4,10), n_oriented (1,1), spots_cap (6,25); ~5.4 rings / 5.1 segments): 0.5302 / 0.6346 (best 41 of the realsim runs). E9 (E6 + mask_wedge_only_frac 0.2): 0.6316 / 0.6154 (best organic; ring hit rate at q column >= 781 on 41 0.57 vs E6 0.46, plain 0.82). E10 (E8 structure + rings_cap (6,16), spots_cap (10,40) + bkg_noise_edge_fix): 0.5587 / 0.6019. E11a / E11b (E6 + edge fix + mask_wedge_only_frac 1.0 / 0.5): 0.6083 / 0.5707 and 0.6120 / 0.5901. No realsim run reaches the plain simulator on 41 (best 0.6346 vs 0.725); more wedge-only frames made 41 worse (0 / 20 / 50 / 100 %: 0.6104 / 0.6154 / 0.5901 / 0.5707, but 50 / 100 % also carry the edge fix); single seed each.
 New default-off simulator options (realsim/realbkg_simulation.py, detector_masks.py): ring_x_share, ring_rel_shuffle, bkg_flat_q, mask_no_gaps, mask_wedge_only_frac, bkg_noise_edge_fix. Bright rim along the invalid-pixel mask traced to _correlate_noise (invalid pixels entered the high-pass; +0.05 at 1-3 px) and removed by bkg_noise_edge_fix (profile flat 0.423-0.438 vs 0.52). Diagnosis: the high-q real rings (q column >= 781) are not detected at all (nothing within 10 px); not box shape, ring count, width, brightness, background level or gap bands (E11). Tools: heatmap/ring_stats_sim.py, sim_pickle.py, ring_compare.py, bkg_profile_x.py, ring_q_diag.py, ring_miss_kind.py, highq_ring_panels.py, edge_profile.py, sheet_job.py, summarize_runs.py labels E5-E11.

New-box datasets (2026-10-21/22): the user re-fitted all eval boxes (stored widths = 2 sigma, drawn box = centre +- sigma = label convention a_coef = w_coef = 2); the visibility-fixed files organics_visfix.h5 / perovskites_nan_wedge_visfix.h5 are the evaluation sets; old APs are NOT comparable (details in the DINO repo notes). Old heatmap checkpoints re-scored on the new files lose 0.062 on 41 (0.725 -> ~0.663) and 0.006 on organic. H1 (plain frozen recipe, box_coef 2,2, colorbox1, seed 42): organic 0.6175 / 41 0.6844; 2,2 seeds 1 / 2 / 42: organic 0.6065 +- 0.0097, 41 0.6835 +- 0.0033.
 Evaluator change (heatmap/evaluation.py, commits d9b2d82 and 91946fc): the stored GT is +-1 sigma, but a run trained at box_coef a,w predicts boxes of half-extent coef * sigma / 2, so with a fixed GT the larger-box runs are penalised. HM_GT_BOX_COEF=<a_coef,w_coef> (or 'auto' = the run's own box_coef from hm_args or its args.json) scales the GT about its centre by (w_coef/2 in x, a_coef/2 in y) before the Evaluator and the matcher; ring boxes only in x; the neighbour-distance buckets (<5 px ...) stay on the unscaled GT; default off (old numbers unchanged). Checked on synthetic boxes (heatmap/test_gt_scale.py) and with images (heatmap/gt_scale_panels.py).
 Queue 8 re-scored (fixed GT -> scaled GT, organic / 41): 2,2 seeds unchanged (0.6026/0.6798, 0.5994/0.6863, 0.6175/0.6844); 2.5,2.5 0.6072/0.6808 -> 0.6419/0.6891; 3.0,3.0 0.5750/0.6594 -> 0.6398/0.7030; 2.8,1.3 0.6021/0.6543 -> 0.5951/0.6652. Against 2,2 seed 42 under the scaled GT: 2.5,2.5 +0.024 / +0.005; 3.0,3.0 +0.022 / +0.019; 2.8,1.3 -0.022 / -0.019. Single seed for the larger boxes (2,2 seed SD 0.0097 organic / 0.0033 41); a larger box also makes the IoU match easier, so 'better' is partly easier matching. Queue 9 (3.5,3.5 and 4.0,4.0, colorbox1, started 2026-10-22 11:22) and seed replicates for the larger boxes are pending.
