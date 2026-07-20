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

## I. Semi-supervised pseudo-labeling on the real corpus (Semi-DETR MVP, DINO) — TRAINING-ONLY
Goal: sim→real adaptation via mean-teacher pseudo-labeling of the 13k-frame real unlabeled corpus
(the #1 untried lever per ROADMAP). Full design + rationale: **`docs/SEMI_DETR_INTEGRATION.md`**
(Semi-DETR, Zhang et al. CVPR 2023, arXiv:2307.08095). The exported ONNX graph, pre/post-processing,
and `--eval` path are all unaffected — only training changes; old checkpoints still load.
- **`datasets/real_unlabeled.py`** — NEW. `RealUnlabeledDataset` serves weak/strong view pairs of
  `backbone_ssl_corpus.h5` (CPU/numpy → `num_workers>0` allowed, unlike the cuda `SimulationDataset`).
  Strong aug = `photometric_strong` (gamma/exposure/q-ramp/noise, mirrors `augment_v2` MINUS its
  internal χ-flip — weak/strong must stay pixel-aligned or pseudo-boxes would be wrong); optional
  shared χ-flip is applied to BOTH views before it. No-data stays exactly 0.
- **`engine.py`** — NEW `make_pseudo_targets` (EMA-teacher forward on weak views → per-class score
  thresholds, seg lower than ring for faint recall → class-aware NMS ring 0.1/seg 0.4 → target dicts
  in the exact `SimulationDataset` schema) and NEW `train_one_epoch_semi`
  (`L_sup(synthetic, DN on) + λ(epoch)·L_unsup(real pseudo, DN off)`; λ ramps after
  `semi_start_epoch`; teacher hard-seeded from the student at the semi boundary via `ema_m.set`).
  DN is disabled on pseudo-targets by zeroing `model.dn_number` for that forward (calling
  `model(x)` with no targets in train mode would crash in `prepare_for_cdn`; DN toward noisy
  pseudo-boxes would also just reinforce teacher error). Logs `loss_unsup`/`lam`/`pseudo_seg`/
  `pseudo_ring` per step — if `pseudo_seg` decays to 0 the loop is eating its own tail.
- **`models/dino/matcher.py`** — NEW `TopkMatcher` (one-to-many: each pseudo-box supervises its
  top-M lowest-cost queries), the building block for Semi-DETR stage-wise hybrid matching.
- **`models/dino/dino.py`** — `SetCriterion` gained `matcher_o2m`/`use_o2m` (default off); the four
  matcher call sites (main/aux/interm/enc) use the o2m matcher only while the engine flips
  `use_o2m` around the unsupervised criterion call. `build_dino` attaches the `TopkMatcher`.
- **`main.py`** — builds the real loader once before the epoch loop when `use_semi`; branches to
  `train_one_epoch_semi`; per-epoch eval additionally logs the EMA teacher's AP during the semi
  phase (`exp_ap_<name>_teacher.txt`) — the teacher is the natural deployment candidate.
- **`config/DINO/DINO_4scale_swin_semi.py`** — NEW config with all knobs (`semi_start_epoch=50`,
  `unsup_loss_weight=2.0` + 5-epoch ramp, `pseudo_thr_ring/seg=0.4/0.3`, `use_ema=True`,
  `ema_decay=0.999`; phase-3 `hybrid_matching=False` default).
- **`backbone_curation/ssl/run_detector_semi.sbatch`** (+ `_smoke`) — launchers; `--amp` required
  (two student forwards per step ≈ 2× activations). Auto-resume conventions unchanged.

**Run 1 (`dino_semi1`, job 2650922, from-scratch + 50-ep burn-in): FAILED — class collapse,
2026-07-03.** The epoch-50 teacher was under-confident on RINGS in the real domain (sim→real gap;
the mature ssl1 teacher on the same corpus yields ~24 ring pseudo-boxes/batch, the young one ~3), so
`thr_ring=0.4` starved ring pseudo-labels to extinction within ~8 epochs → unsup loss taught "no
rings in real-like images" → 41 AP collapsed 0.63→~0.10 (UT §3.3 class-bias spiral, on the class we
did NOT protect: the recall-chasing LOW seg threshold was inverted vs the teacher's actual real-domain
confidence profile). Late phase: ~200 hallucinated segments/frame passed `thr_seg=0.3` (avg
pseudo_seg ~400/batch) → garbage-dominated unsup signal at λ=2. Killed at ep159. Diagnosed entirely
from the logged `pseudo_seg`/`pseudo_ring` counters — keep them. **v2 fixes (in config + engine):
warm start from ssl1 (`semi_start_epoch=5`), thresholds swapped (ring 0.30 / seg 0.35), hard
`pseudo_max_per_img=30` cap after NMS, λ 2.0→1.0. Run 2 = `dino_semi2`.**

**Run 2 (`dino_semi2`, job 2651525, ssl1 warm start + EMA 0.999 + λ 1.0): FAILED — slow
confirmation-bias drift, 2026-07-04.** The v2 fixes eliminated the collapse (pseudo counters healthy
and balanced through ep54: ~16 seg / 20–34 ring per batch, cap never bound) but ALL four AP curves
declined monotonically over ~50 semi epochs: teacher organic 0.552→0.406, teacher 41 0.744→0.663
(EMA-smoothed ⇒ genuine degradation, not noise); student similar. Secondary signal: ring pseudo-count
inflation 19→34/batch with flat segments = slow confidence inflation (run-1's spiral in slow motion,
also inflates eval FPs past the 0.1 score cut). Read: at λ=1.0 roughly half the gradient budget is
pseudo-label noise; the sup branch cannot anchor against it and the error compounds through the EMA
loop (worse student → worse teacher → worse labels). Killed at ep54. **v3 = FROZEN-TEACHER
DIAGNOSTIC (`dino_semi3`): `ema_decay=1.0` (teacher = settled ssl1, never updates — severs the
feedback loop by construction) + λ 1.0→0.5. Decision rule: AP still declines ⇒ the pseudo-label
signal itself is harmful at these thresholds (next lever: drop box-regression on pseudo-labels per
Unbiased Teacher, or SSRT-style quality weighting); AP holds/improves ⇒ the EMA loop was the culprit
(next lever: slower teacher / periodic re-anchoring).**

**Run 3 (`dino_semi3`, job 2652578, frozen teacher + λ 0.5): VERDICT = the pseudo-label SIGNAL
itself is a net tax, 2026-07-04.** 100 semi epochs: counters dead flat (18/18 per batch — loop
severed as designed, retro-confirming run-2's ring inflation was teacher drift), NO collapse, 41
held ~0.72 (vs 0.746 start) — but organic sat ~0.51 (ep16-70) then eroded to ~0.48 (ep72-100) vs
the 0.573 warm-start point, never recovering. Both branches of the drift mechanism are now
excluded; what remains is the content of the labels. Prime suspect = BOX COORDINATES (confidence
vouches for what, not where — Jiang 2018 / Unbiased Teacher): L1/GIoU regression toward
slightly-wrong pseudo-boxes steadily degrades localization, which AP punishes directly. Killed at
ep102. **v4 (`dino_semi4`): `unsup_cls_only=True` — unsupervised branch keeps only the
classification loss (`loss_ce*` keys); pseudo-boxes still steer the Hungarian matching but carry no
coordinate gradient (engine.py train_one_epoch_semi). Frozen teacher + λ 0.5 kept (one variable at
a time). Outcomes: organic holds/climbs ⇒ geometry noise was the poison → reintroduce boxes via
quality weighting (SSRT-DETR §3.5); still erodes ⇒ MVP pseudo-labeling recorded as a documented
negative with three clean ablations (loop / λ / loss content).**

**Run 4 (`dino_semi4`, job 2653256, frozen teacher + λ 0.5 + cls-only): STILL a net tax —
SERIES CLOSED as a documented NEGATIVE, 2026-07-05.** 200 semi epochs: counters flat (17-18/18)
throughout, 41 roughly held (0.737 → ~0.715), but organic eroded 0.562 → ~0.50 (ep40-100) →
**plateau ~0.465 (ep100-204)** — slower than run 3 but same direction and endpoint, ~0.10 below the
warm-start point. Killed at ep206 (plateaued ~100 epochs).

**Series conclusion (4 single-variable ablations: teacher maturity / EMA loop / λ / loss content):**
MVP teacher-student pseudo-labeling (Semi-DETR/Unbiased-Teacher style, hard pseudo-labels at fixed
per-class thresholds) does NOT transfer to this synthetic→real GIWAXS setup at any ablated operating
point — every variant ends below its own warm-start AP. The failure is *graded* (each fix slowed the
degradation: collapse → drift → slow erosion → slower erosion) but never crossed into net positive.
- **Unresolved confound (record honestly):** eval label-incompleteness. The teacher plausibly labels
  real-but-unannotated peaks (Diagnostic C: FPs sit ON rings); a student trained to be confident on
  them loses measured precision/AP even if it genuinely improved. Cheap future test:
  recall-at-fixed-FP or FP visualization (`diagnostics/` pattern) of the `dino_semi4` checkpoint
  vs ssl1 — if run-4's "extra" detections are real peaks, the negative verdict softens.
- **If revisited, try first:** SSRT-DETR §3.5 quality weighting (soft, per-image trust instead of
  hard thresholds), §3.4 multi-view consistency filtering, much HIGHER precision-protecting
  thresholds, or pseudo-labels restricted to a curated high-quality corpus subset.
- The semi machinery stays in the repo, default-off (`use_semi` unset in base configs) — the
  training path of every non-semi config is byte-identical to before phase I.
- Run records: `detector_runs/dino_semi{1,2,3,4}` + `backbone_curation/ssl/dino_semi-*.out`.

## J. Cross-DINO Boost Loss + Category-Size soft label (Exp A) — TRAINING-ONLY
Next lever after phase I closed (per `docs/CROSS_DINO_INVESTIGATION.md` §4a/§7): the portable,
ONNX-safe subset of Cross-DINO (arXiv:2505.21868). Equations verified against the paper before
implementing (Eq. 4: `cs = sqrt((h/H)(w/W))·y`; Eq. 5 Boost Loss; α=0.25/β=1.0/γ=2.0; applied at
all classification-loss sites; matching unchanged).
- **`models/dino/utils.py`** — new `boost_loss` beside `sigmoid_focal_loss`. Unit-tested property:
  with `cs == targets` (cs=1 at positives) and β=1 it reduces EXACTLY to `sigmoid_focal_loss` —
  the CS soft label is the only difference.
- **`models/dino/dino.py`** — `SetCriterion.loss_labels` gains a `use_boost_loss` branch: builds the
  CS map from the matched GT boxes (cxcywh already normalized → `sqrt(w·h)` at the class slot) and
  calls `boost_loss`. Fires at every call site (main/aux/interm/enc/DN) = the paper's "encoder and
  all decoder predictions". `build_dino` attaches `use_boost_loss`/`boost_{alpha,beta,gamma}`
  (defaults off — all existing configs unchanged).
- **`config/DINO/DINO_4scale_swin_boost.py`** — Exp-A config (paper defaults). Documents the domain
  caveat: our elongated boxes give cs ~0.17 (ring) / ~0.04 (segment), so all positive targets sit
  far below 1 and rings are weighted ~4× over segments — **β is the calibration knob** (sweep
  {0.5, 0.25} if training degrades or segment recall drops).
- **`backbone_curation/ssl/run_detector_boost.sbatch`** — fine-tune from the ssl1 checkpoint,
  output `detector_runs/dino_boost1`. A/B read-out: organic + 41 AP vs the warm-start band
  (organic ~0.55-0.58 / 41 ~0.73-0.75, established across the dino_semi2-4 burn-ins). Gate on
  organic; stop rule per the investigation doc (if Exp A and CCTM both fail to move organic,
  Cross-DINO is investigated-and-declined).

**Exp A VERDICT (2026-07-06): DECLINED — the β-sweep is monotonically negative.**
- **β=1.0** (`dino_boost1`, job 2655289): organic eroded from the 0.586 ssl1 warm start to a flat
  **~0.42** plateau by ep~40 (−0.17, no recovery over 160 ep); 41 0.737 → ~0.65. Killed ep199. This
  is exactly the pre-registered cs-starvation mode — the positive loss term is scaled by cs (ring
  ~0.17, segment ~0.04), so the model is trained to under-score true peaks and the ranking collapses
  (visible as `train_loss_ce` ~0.003, ≈10× below baseline focal). lr_drop@280 cannot close a 0.14 gap.
- **β=0.5** (`dino_boost2`, job 2656944): fresh ssl1 warm start; ran to ep310, converged after the
  lr_drop@280 at organic **0.525** (stable, range 0.515–0.534) / 41 **0.706** — a mild but real net
  deficit, still below the continued-train band (organic 0.55–0.58).
- **Monotonic recovery toward β=0:** 1.0→0.42, 0.5→0.525, 0 (≡ plain focal ≡ ssl1)→0.586. Every step
  that turns the size-weighting *down* buys organic back. In our domain — all objects uniformly
  tiny-cs (elongated boxes; no size diversity for the method to exploit) — the Category-Size weighting
  is **pure tax**, and the best operating point is "boost off". β=0.25 was skipped: the 2-point
  monotonic trend predicts ~0.55 (still ≤ baseline), i.e. not a win.
- **Decision:** Boost Loss declined. `use_boost_loss` stays default-off (all non-boost configs
  unchanged; the code + unit test remain in the tree). Proceed to Exp B (CCTM) — the doc §7 stop-rule
  requires BOTH portable modules to fail before Cross-DINO is fully declined.

## K. Cross-DINO CCTM feature-enrichment module (Exp B)
Second (and final) portable Cross-DINO piece after Boost Loss (Exp A) was declined. **CCTM** (Cross
Coding Twice Module, arXiv:2505.21868 §III-C) reinjects the pre-encoder backbone feature into the
encoder memory before decoder query selection, giving the decoder a finer "cross feature".
- **`models/dino/deformable_transformer.py`** — new `CCTM(nn.Module)`: two rounds of elementwise
  sigmoid-gated fusion of encoder memory `E` (`memory`) and the input-projected backbone `B`
  (`src_flatten`), which are token-aligned `(bs, Σhw, 256)`. Inserted right after the encoder returns
  `memory`, before `gen_encoder_output_proposals` (the `two_stage='standard'` query selection). A
  `use_cctm` flag is threaded through `__init__` and `build_deformable_transformer`
  (`getattr(args,'use_cctm',False)` → all existing configs unchanged). The module is created AFTER
  `_reset_parameters()` so its custom init is preserved. Warm-start load is `strict=False`, so the
  new `cctm.*` keys (absent from the ssl1 checkpoint) simply keep their init.
- **Warm-start design (the lesson from Exp A):** a per-channel zero-init LayerScale (`gamma`) wraps
  the fusion, so CCTM is an **EXACT identity at init** (unit-tested: `max|out−E| = 0`). The fine-tune
  therefore *starts* precisely at ssl1's operating point and learns how much fusion to add — no
  epoch-0 objective shock (the thing that made Boost a net tax). `gamma` receives gradient first
  (ReZero-style), which then unlocks the gate projections.
- **Fidelity note:** the paper's Eq. 2-3 are reconstructed from text/figure (no public code) and
  include an unbounded `2·E·B'·E'` term; we implement the **bounded, convex** realization of the same
  described intent (stability matters for a warm-start fine-tune, not from-scratch). The structural
  fidelity the investigation doc calls well-confirmed — post-encoder / pre-decoder, elementwise gated
  reinjection, applied twice — is preserved.
- **ONNX-safe (verified):** only `Linear`/mul/add/sigmoid; opset-16 export of the module traces
  cleanly; it does not touch the MSDeformAttn custom op or change the token/feature count. Unlike Boost
  Loss (training-only), CCTM **is** on the exported path, so a full-model ONNX parity check on the
  first checkpoint remains the pre-deploy gate.
- **`config/DINO/DINO_4scale_swin_cctm.py`** (`use_cctm=True`) + **`run_detector_cctm.sbatch`**
  (fine-tune from ssl1). A/B organic vs ssl1 0.586; gate on organic. STOP RULE (doc §7): if CCTM also
  fails to move organic → Cross-DINO investigated-and-declined; do **not** port the Strip-MLP backbone.
- **Identity-init validated empirically** (`dino_cctm1`, job 2659048, uniform 1e-5, aborted): epoch-0
  eval organic **0.561** / 41 **0.759** ≈ ssl1 (0.586/0.762) — no epoch-0 shock (contrast Boost β=1.0
  which dipped to 0.511). Confirms the zero-init LayerScale starts the model exactly at ssl1. It also
  showed the flip side: at the body's 1e-5 rate `gamma` barely moves (still ≈ssl1 at ep2), so an
  architectural module grafted onto a converged model can't gain traction → a null would be a false
  negative.
- **Higher CCTM LR (the real run):** `util/get_param_dicts.py` gains an optional third param group —
  `cctm.*` at `args.lr * lr_cctm_mult` while backbone/encoder stay at `lr`/`lr_backbone`. Config sets
  `lr_cctm_mult=10.0` → CCTM trains at 1e-4 (the usual from-scratch rate), body at 1e-5. Default
  (`lr_cctm_mult=1.0`/absent) is byte-identical to the original two-group split — no other config
  affected. Relaunched as **`dino_cctm2`, job 2659055** (2026-07-06). Escalation: a null even at 10×
  CCTM-LR → one co-trained run (SSL backbone init, CCTM from epoch 0) before invoking the stop rule.

**Exp B VERDICT (2026-07-08): DECLINED — trustworthy null; the mechanism misses our ceiling.**
- `dino_cctm2` (CCTM @10× LR) converged post lr-drop@280 at organic **0.567** (0.556–0.577) / 41
  **0.760** — organic in-band but below ssl1 0.586 (no lift on the gate); 41 held at baseline. The 10×
  LR worked (curve moved/varied ⇒ gamma grew, the module genuinely trained), so the null is trustworthy.
- **Mechanistic diagnostic** (`diagnose_C`-style, organic, 817 GT peaks, cctm2 vs ssl1) settles the
  co-train question: CCTM raises recall on ALREADY-EASY peaks — bright (vis=3) 0.693→0.726, ring
  0.833→0.875, inner-q 0.567→0.598 — but does NOTHING for the two modes that cap organic: **faint
  (vis=1) 0.330→0.330 and high-q (682–1024) 0.436→0.436, both identical**; precision slips 0.841→0.833
  (FP/img 10.4→11.2, more confident FPs). Net AP flat = easy-peak gains cancelled by the precision cost.
- **Why co-train won't rescue it:** CCTM reinjects `B` (the backbone feature); faint/high-q peaks are
  weak in `B` itself (the representation/sensitivity ceiling the SSL effort already identified).
  Reinjecting detail sharpens what is already visible — it cannot manufacture sensitivity, and
  co-adaptation cannot use signal that is not there.
- **DECISION: Cross-DINO investigated-and-declined.** Both portable modules failed the organic gate
  (Boost declined; CCTM null + off-target). Do NOT port the Strip-MLP backbone (doc §7 stop rule).
  CCTM code + the `lr_cctm_mult` param group stay in the repo **default-OFF** (`use_cctm=False` /
  `lr_cctm_mult=1.0` → byte-identical no-op; all other configs unchanged). **NEXT LEVER:**
  Co-DETR/Co-DINO training-only auxiliary heads — adds faint-peak *sensitivity* (the right axis the
  diagnostic points to), zero inference cost, ONNX-safe (heads dropped at export). Chosen 2026-07-08.

## L. Co-DINO collaborative auxiliary head (Co-DETR, arXiv:2211.12860) — TRAINING-ONLY
Next lever after Cross-DINO declined (`docs/CO_DINO_INVESTIGATION.md`). Our ceiling is faint/high-q
peak SENSITIVITY (recall); Co-DETR's aux head injects dense one-to-many supervision into the *encoder*
features → the most on-thesis lever (adds sensitivity, not detail-routing/re-weighting). Phase-0 MVP:
ONE FCOS center-sampling head, encoder-supervision only, no customized positive queries.
- **`models/dino/co_heads.py`** (new) — `memory_to_pyramid` (inverse of the encoder flatten,
  unit-tested); `CoHeads` (shared 3×3-conv stem + cls/box heads over the 4 encoder pyramid levels);
  `CoCriterion` (FCOS center-sampling one-to-many assignment, smallest-area tie-break, focal-cls +
  GIoU/L1-box). FCOS not ATSS: our elongated arcs would starve ATSS anchor-IoU.
- **Wiring (all default-off via `getattr(args,'use_co_heads',False)`):** `deformable_transformer.py`
  stashes the (post-CCTM) encoder memory as a 2-D pyramid, training-only; `dino.py` runs the head
  (`out['co_head_outputs']`, `self.training`-gated), adds `loss_co_{cls,bbox,giou}` to `weight_dict`,
  attaches `model.co_heads` + `criterion.co_criterion`; `SetCriterion.forward` folds the co-losses
  into its returned dict → **`engine.py` unchanged** (they ride the existing weighted sum).
  `get_param_dicts.py` generalized to optionally split `co_heads.*` into a higher-LR group.
- **Deploy-safe (verified by GPU smoke):** strictly training-only — eval/export produces NO
  `co_head_outputs` (the `self.training` gate + the ONNX whitelist reading only
  `pred_logits`/`pred_boxes`); eval-mode output keys are byte-identical to baseline. Warm-start load
  leaves only the 12 `co_heads.*` tensors fresh. Unit tests: dense loss finite, gradient reaches the
  encoder features, empty-GT and smallest-area tie-break correct, pyramid reshape is the exact inverse.
- **Two recipes built:** `DINO_4scale_swin_codino.py` (warm-start fine-tune of ssl1, co-heads @10× LR)
  and `DINO_4scale_swin_codino_scratch.py` (**FROM-SCRATCH co-train**: ssl1's exact recipe — SSL
  backbone via `backbone_dir`, uniform 1e-5, no amp — + co-heads from epoch 0). **Chose from-scratch**
  (the faithful test: a training-scheme change that shapes *encoder* feature learning needs the encoder
  to co-adapt from init, which a warm-start under-tests). Clean A/B: the only difference vs ssl1
  (organic 0.586 / 41 0.762) is the aux head. Run `dino_codino_scratch1`. **GATE = organic AP AND the
  faint/high-q recall probe** (`diag_compare.py`); AP-up-but-recall-flat = CCTM-null shape → decline.

**Co-DINO VERDICT (2026-07-10): DECLINED.** `dino_codino_scratch1` ran to ep286 (past lr-drop 280) and
held organic **~0.03–0.04 below ssl1** (mean Δ ep82–300 ≈ −0.035) with **41 even** — the sim2real
signature: dense one-to-many supervision on synthetic holds on synthetic-like 41 but drags real organic.
Recall probe (`diag_compare`, codino ckpt vs ssl1, 817 organic GT, `base` mode — co_heads are
training-only, 12 keys ignored at inference) shows **no faint/high-q gain**: recall 0.537→0.435;
**vis=1 0.330→0.244**, **high-q 0.436→0.317** (both WORSE — the target axes), ring unchanged 0.833.
(The −0.10 recall-at-0.3 overstates vs AP −0.03 — codino shifted to a higher-precision/lower-recall
point, 64 vs 83 FP — but the target axes show no gain regardless.) So even the *faithful* test (encoder
co-adapting from ep0) does not lift the ceiling. `use_co_heads` stays default-off. Cut ep286. Co-DINO
joins semi / Boost / CCTM as a clean documented negative.

## M. Style-transfer input matching (synthetic→real appearance) — TRAINING-ONLY
Pivot after four detector-side levers (semi, Boost, CCTM, Co-DINO) all came back null/negative — the
finding is the bottleneck is the DATA (sim2real), not the detector. Attack it directly
(`docs/STYLE_TRANSFER_INVESTIGATION.md`): the faint/high-q recall ceiling is capped by training on
synthetic peaks whose faint ones don't *look* real.
- **`datasets/style_match.py`** (new) — monotone per-image CDF/histogram match of SYNTHETIC images onto
  a reference intensity distribution pooled from REAL corpus frames (`backbone_ssl_corpus.h5`, 12,991
  frames, disjoint from eval, in the model's `[0,1]` `to_model_input` space). Rank-preserving →
  no-data (0) stays 0, intensity ordering preserved. `build_reference` (offline, run once → `style_ref.pt`),
  `load_reference`, `cdf_match`.
- **`main.py`** — `SimulationDataset.__getitem__` applies `cdf_match` before the channel-repeat, gated by
  `use_style_match` (default off). PURELY a training-data transform: **no model/loss/export changes**.
  Eval uses `PyGIDDataset` (`SimulationDataset` is training-only, `main.py:246/423`) → real-image
  preprocessing + ONNX byte-identical.
- **Distinct from reverted Path A** (phase H): Path A fixed mask geometry + quantization *level count*;
  this matches the distribution *shape* to a real reference, and gates on the faint/high-q **recall
  probe** (not AP — Path A's AP-only test could miss a faint-recall change).
- **Verified:** unit tests (weakly monotone, 0 inversions; zeros + local-max preserved; histogram-match
  decile-Δ 0.0006). Faint-peak contrast retention (1,628 simulated peaks): median contrast unchanged,
  faint tail slightly *lifted* (p10 0.0068→0.0078), +2.4pp peaks below 1 8-bit level; faintest quartile
  ~29% flatten to ≤1 level (accepted — real peaks are low-contrast; the recall probe is the judge).
  Dataset integration + inference-untouched confirmed on GPU.
- **`config/DINO/DINO_4scale_swin_stylematch.py`** + **`run_detector_stylematch.sbatch`** — warm-start
  screen from ssl1 (fair here: a DATA change needs no architectural co-adaptation). Run `dino_stylematch1`.
  GATE = faint(vis=1)/high-q recall probe; stop after Phase 0 if flat.
- **VERDICT — DECLINED (documented negative).** `dino_stylematch1` ran the full 500 epochs (ep499).
  - **AP: dead-even wash.** Post-drop plateau (ep282–436, n=78 matched evalpoints vs ssl1): organic
    Δ **−0.0017** (0.5604 vs 0.5621), 41 Δ **+0.0049** (0.7504 vs 0.7456); organic peak 0.5848 vs 0.5860.
    Neither the tax the other four levers were, nor a win.
  - **Recall probe (the gate): faint flat, high-q worse.** Same 8-img organic probe (`diag_compare.py`,
    GT=817), ssl1 → style-match: overall recall 0.537→0.518; **vis=1 (faint) 0.33→0.32 (flat — ceiling
    unmoved)**; **q 682–1024 (high-q) 0.436→0.37 (−0.066, the largest move, wrong direction)**; vis=2
    0.53→0.492, vis=3 0.693→0.673. Only positive is a marginal precision/FP tick (0.841→0.849,
    FP/img 10.4→9.4 — the expected side effect of detecting slightly less).
  - **Conclusion:** global 1-D intensity-distribution (CDF/histogram) matching does **not** crack the
    faint ceiling. The sim2real gap that matters for faint/high-q recall is **structural** (noise
    texture, background morphology, detector artifacts), not the intensity histogram shape. Machinery
    stays in the repo, default-off (`use_style_match=False`); training path for all other configs
    unchanged, inference/ONNX byte-identical.
  - **Series status:** FIVE single-variable levers now closed as documented negatives (Semi-DETR
    pseudo-labeling · Cross-DINO Boost · CCTM · Co-DINO · style-match). Four detector-side + one
    data-appearance all null/negative ⇒ the remaining plausible cracks are (a) *structural* sim2real
    realism (learned/GAN-style or physics-based noise+background injection, not histogram matching),
    and (b) the persistent eval label-incompleteness confound (faint "misses" partly unlabeled in GT —
    see "Diagnostics & roadmap" KEY FINDING). Next cheap independent test: label-completeness re-eval
    (expert review of `viz_fp.png` / recall-at-fixed-FP) before spending another training lever.

## N. Structural sim2real noise injection (real-level pixel grain) — TRAINING-ONLY
Step 2 of the sim2real track, after style-match (phase M) closed the 1-D intensity HISTOGRAM gap
and did NOT move faint recall. Direct structural measurement (tmp_diag/{struct_gap,robust_grain,
mf_snr}.py, on real corpus frames vs current synthetic output, both in the model's [0,1] space)
found the remaining gap is SPATIAL TEXTURE — invisible to the 1-D histogram:
- **REAL** frames carry a heavy WHITE pixel-grain floor: robust (MAD) high-pass residual **0.121**,
  autocorrelation length ~1 px (white), isotropic. **SYNTHETIC** frames are ~4x smoother
  (MAD **0.032**) — the sim's noise (Perlin/Poisson) is added before a 3x3 smoothing kernel and is
  spatially correlated (~3 px). Visually: real = pervasive sandpaper grain; synth = smooth
  gradients with sharp edges (tmp_diag/{montage,texture_montage}.png).
- MECHANISM: the detector only ever trains on smooth synthetic backgrounds -> its low-level
  features never adapt to the real noise floor -> misfires on grainy real images (missed faint/
  high-q peaks). This is a DOMAIN-ADAPTATION gap, distinct from the phase-M marginal gap.
- **`datasets/struct_noise.py`** (new) — `add_grain`: per-image white Gaussian grain, std ~
  U[0.05, 0.13] (0.13 -> synth MAD grain ~0.13, matching real 0.121), added to the FINAL [0,1]
  image AFTER the sim's smoothing so it survives; no-data (0) preserved, valid pixels in [1/255,1].
- **NO peak-boosting (verified unnecessary).** Naive per-pixel contrast-vs-sigma suggested grain
  would bury 63% of peaks; but a detector INTEGRATES over a peak's footprint, so the correct
  detectability metric is matched-filter SNR ||s||_2/sigma ~ amplitude*sqrt(N). Measured: peaks
  have median footprint ~210 px, and at real grain sigma=0.12 even the faintest quartile sits at
  MF-SNR ~26 (only ~1% truly buried). Grain-only keeps peaks detectable via spatial coherence; an
  explicit boost moved MF-SNR 26.2 -> 26.4 (no-op). Optional `boost_peaks`/`grain_with_peak_floor`
  kept in the module, off by default (`struct_noise_boost=False`).
- **`main.py`** — `SimulationDataset.__getitem__` applies `add_grain` before the channel-repeat,
  gated by `use_struct_noise` (default off). PURELY a training-data transform: **no model/loss/
  export changes**. Eval uses PyGIDDataset -> real-image preprocessing + ONNX byte-identical.
- **Distinct from style-match (M) and the reverted Path A (H).** M matched the intensity histogram
  SHAPE; H changed mask geometry + quantization level count; this matches the 2-D noise TEXTURE
  (spatial grain), the one thing the 1-D metrics are blind to.
- **Verified:** grain reaches real level (synth MAD 0.032 -> 0.135 at sigma_hi vs real 0.121);
  peak survival by MF-SNR (above); no-data preserved; dataset integration produces valid [0,1]
  samples; py_compile + config-load OK; inference path untouched.
- **`config/DINO/DINO_4scale_swin_structnoise.py`** + **`backbone_curation/ssl/
  run_detector_structnoise.sbatch`** — warm-start screen from ssl1 (fair: a DATA change needs no
  architectural co-adaptation, same as M). Run `dino_structnoise1`, 500 ep, lr-drop 280.
  GATE = faint(vis=1)/high-q recall probe (diag_compare.py) vs ssl1; stop after Phase 0 if flat.
- **VERDICT — PENDING** (run launched, awaiting plateau + recall probe).

## Results so far (run `ringseg_2class_20260603-142434`, ep360 of 500; baseline also ~ep350)
| set | new 2-class @ep360 | old 91-class baseline | notes |
|---|---|---|---|
| organic (pygid) | **0.554** (still rising) | 0.552 | even / slight edge new |
| 41 (roi_data)   | **0.758** (peak 0.768) | ~0.751 | slight edge new |
Plot: `train_output/ringseg_2class_20260603-142434/ap_curves.png`.

**Superseded by the SSL-backbone effort (2026-06):** round-1 SSL backbone `ssl1` = best single
model (organic **0.586** vs 0.554 scratch); deployed best = **ssl1+baseline ensemble**
(organic **0.605** / 41 **0.780**); all three SSL refinements (recipe v2 / freezing / 5-scale)
were NEGATIVE — full record incl. run paths in **`backbone_curation/RESULTS.md`**,
deployment in `backbone_curation/ENSEMBLE_DEPLOY.md`.

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
