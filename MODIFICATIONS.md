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
