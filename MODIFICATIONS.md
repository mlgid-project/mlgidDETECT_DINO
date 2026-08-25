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
- **VERDICT — DECLINED (2026-07-21).** Run `dino_structnoise1` (job 2677809, warm-start ssl1)
  evaluated at the post-lr-drop plateau (ep282-302, 11 eval points) + recall probe (diag_sn-2679464):
  - **AP:** organic post-drop mean **0.5414 vs ssl1 0.5621 (-2.1 pts)**; 41 wash (0.7510 vs
    0.7456, +0.5). Organic's best value (0.5694) was at ep10 — right after the warm start,
    i.e. grain training actively pulled organic AP DOWN from the ssl1 state.
  - **Recall probe** (organic, score>0.3, checkpoint ep302 vs ssl1): overall **0.512 vs 0.537**;
    faint vis=1 **0.32 vs 0.33** (no gain on the target stratum); high-q third **0.395 vs 0.436**
    (-4.1 pts, worse on the other target stratum); precision 0.821 vs 0.841, FP/img 11.4 vs 10.4.
    Only ring recall ticked up (0.875 vs 0.833) at the cost of segments (0.501 vs 0.528).
  - **Interpretation:** adding real-level white grain to smooth synthetic images does NOT teach
    real-noise robustness — it just makes training harder/noisier (uniformly slightly worse, same
    signature as phase M's high-quantization arm). Together M+N close the sim2real appearance
    hypothesis: neither matching the 1-D intensity histogram nor the 2-D noise texture moves
    faint/high-q recall. The remaining candidate is peak CONFIGURATION realism (structures/
    intensities), tested by the physics-CIF track (docs/PHYSICS_SIM_INVESTIGATION.md).
  - Machinery kept (off by default): `use_struct_noise=False` everywhere; config + sbatch retained
    for reproducibility. No production impact (training-only lever, never deployed).
  - Run went the full 500 ep (completed 2026-07-21 23:52); ep498 organic 0.5423 — verdict unchanged.

## O. q-decay intensity envelope — DECLINED BY MEASUREMENT, NEVER RUN (2026-07-20)
Pre-registered cheap test of the strongest single physics mechanism behind the sim2real gap:
"real high-q peaks are systematically weak (form factors / Debye-Waller), synthetic ones are
randomly bright, so the detector never learns the real high-q regime". Planned as a training-only
multiplicative envelope on the final image (`datasets/q_decay.py`, mirroring `struct_noise.py`).
**Killed by two measurements BEFORE building the run** (~30 GPU-min instead of a 1.7-day run):
- `tmp_diag/qprofile.py`: column-level radial profiles are nearly FLAT in both domains — log+HE
  in the shared preprocessing chain removes physical q-decay from real AND synthetic images.
- `tmp_diag/peak_contrast_vs_q.py`: the peak-level relation is INVERTED vs the hypothesis. Real
  labeled peaks are MOST contrasty at high q (organic high/low median ratio **14.3**, 41 **1.97**)
  while synthetic is backwards (**0.70**). Organic's FAINTEST peaks live at LOW q.
Implications: (a) an attenuation envelope would push synth the WRONG way — no run; (b) the high-q
recall deficit is not labeled-peak faintness; (c) the mismatch is in the JOINT (q, intensity)
distribution, which no hand-tuned 1-D envelope can fix — motivating phase P. `datasets/q_decay.py`
was deleted rather than left as dead code.

## P. Physics-based CIF simulation (peak configuration from real crystallography) — TRAINING-ONLY
User-proposed direction after M+N closed the sim2real APPEARANCE hypothesis. The standard sim
places peaks at RANDOM q with RANDOM intensities; this dilutes training with images whose peak
configurations come from real structures (CIF -> structure factors -> orientations), giving the
correct joint (q, chi, intensity) statistics and real morphology (powder rings vs textured arcs)
automatically. Full design record: `docs/PHYSICS_SIM_INVESTIGATION.md`.
- **`physics_simulation.py`** (NEW sibling; `simulation.py` NOT modified) — `PhysicsSimulation.
  simulate_img()` honors the exact FastSimulation 4-tuple contract and REUSES the entire standard
  appearance chain (internal FastSimulation for detector mask / dark areas / detector-gap filtering
  / `img_from_labels`, then the same module-level chain in the same order). A physics image differs
  from a standard one ONLY in peak placement and relative intensities. Composition per image (user
  spec): 0-1 powder + 1-2 oriented entries; per-entry scale 0.08-1.0 (minor/major phases).
- **`physics_sim/build_exclusion_list.py`** — eval-exclusion via mlgidMATCH at a lowered threshold
  (0.05; user: "really find all candidates"), over 41.h5 + organic_labeled.h5 (45.h5 excluded per
  user). Segments use the library tree matcher with AUDITED caps (top-64 screen candidates per
  node, top-6 branch recursions; every cap event counted into the output) plus a full uncapped NN
  screen stored as an audit layer; rings use a two-sided 1-D q-coverage test done here (the library
  rings path needs `create_all` caches we deliberately don't build). A SELF-TEST (simulate from
  known CIFs, require self-recovery) gates the whole run.
- **`physics_sim/generate_bank.py`** -> `bank.npz` — per CIF: 1 powder entry + 8 recorded random
  fiber orientations, top-200 peaks each, stored as (|q|, chi, intensity). Exclusions: matched
  powder -> powder mode dropped; matched orientation -> entries within 10 deg dropped, OTHER
  orientations of the same structure stay usable (user spec). `physics_sim/merge_shards.py` merges
  array output and REFUSES an incomplete set.
- **`main.py`** — `SimulationDataset` gains `use_physics_sim` / `physics_sim_fraction` /
  `physics_bank_path` (default off); `__getitem__` picks physics vs standard per sample. No model,
  loss, or export change; eval path and ONNX byte-identical.
- **`config/DINO/DINO_4scale_swin_physics.py`** + `run_detector_physics.sbatch` — warm-start ssl1,
  **fraction 0.5 for the first run** (user decision: maximize measurable impact, tune afterwards).
- **THREE BUGS found and fixed during bring-up** (each would have silently corrupted the result):
  1. `mlgidmatch 0.1.3` + `pygidsim 0.1.5` are API-incompatible (`giwaxs_2d` kwarg `q_range` ->
     `q_xy_range`/`q_z_range`); exact-translation shim in `patch_pygidsim_compat`.
  2. Eval peak extraction silently yielded ZERO samples — h5py cannot build a direct-read
     conversion path for these datasets (`np.asarray(dset)` raises), `qz_qxy_range` is sometimes a
     byte string `b'(0, 3.2, 0, 3.2)'` (same workaround as `util/labeleddataset.py`), and organic's
     `fitted_peaks` is a COMPOUND dataset, not a group.
  3. Exclusion-margin angles were computed in MILLER space. mlgidMATCH reports orientations as
     Miller directions and pygidsim maps them via `orientation @ rec`; for a realistic non-cubic
     cell the naive angle reads 45 deg where the physical angle is 70.8 deg — far past the 10 deg
     margin, so matched eval orientations could have leaked into training. Angles are now measured
     after the Cartesian map.
  Also: a `multiprocessing` fork pool DEADLOCKED against torch/OpenMP (8 h, zero units); parallelism
  is now Slurm array shards. Do not reintroduce multiprocessing there.
- **FOURTH BUG (post-launch crash, fixed 2026-07-30).** Both 2026-07-22 runs died on epoch 0,
  iteration 1 with `AssertionError` in the matcher (`util/box_ops.py:53`, `x2>=x1 & y2>=y1` on
  targets). `filter_dark_area`'s polar/quazipolar branch clamps `y2` down (toward the quazipolar
  line ~0.77*x1) and `y1` up, which can invert a box, and its `polar_indices` keep-mask is computed
  BEFORE the quazipolar clamp so it does not catch it. The standard sim survives via a final
  ordering guard (`simulation.py:416-420`) that `physics_simulation.py` was missing; that guard
  (drop `x2<=x1`/`y2<=y1`, re-clamp) is now applied after the dark-area filter. The pre-launch gate
  only checked box ALIGNMENT, not ORDERING — it now also runs a 300-draw box-ordering stress
  (0 inverted over 10,883 boxes after the fix). Verified live: both relaunched runs stepped past the
  crash point cleanly (warm-start to epoch 2, from-scratch stepping, 0 tracebacks).
- **GATE** (as every lever): organic/41 AP per epoch; decisive = faint(vis=1)/high-q recall probe
  vs ssl1.
- **VERDICT — DECLINED (2026-08-03).** 50% physics-CIF dilution is negative on every gate.
  - **AP (from-scratch `dino_physics_scratch1` 2701979, ssl1 backbone + random head, no-amp faithful
    test; TIMEOUT at ep474, well past lr-drop@280 so the plateau is settled).** Converged (ep300+)
    vs ssl1 baseline `dino_ssl1`: organic **0.5395 vs 0.5634 (Δ −0.024)**, 41 **0.6255 vs 0.7454
    (Δ −0.120)**. Even physics's all-time-peak organic (0.580 @ep166, pre-drop) sits below ssl1's
    peak (0.586 @ep238). A transient pre-drop organic lead (Δ +0.013…+0.023 over ep140–210) was pure
    early-convergence speed — ssl1 overtakes at its lr-drop and physics never recovers.
  - **AP (warm-start `dino_physics1` 2701978, fine-tune ssl1 ckpt, COMPLETED ep498).** Brief spike
    (organic 0.617 @ep10) then decays under dilution to converged **0.565 / 0.616** — below the
    deployed ssl1 single-backbone (0.586 / 0.762) on both gates; 41 collapses ~−0.15.
  - **Decisive recall probe (`tmp_diag/diag_sweep.py`, organic, converged ckpts ep474 vs ep436).**
    At a FIXED score 0.3 physics *appears* to win recall (0.668 vs 0.537, incl. faint 0.485 vs 0.33,
    high-q 0.551 vs 0.436) — but that is pure miscalibration: physics fires ~5.5× more boxes
    (56.9 vs 10.4 FP/img, precision 0.545 vs 0.841). At a **matched operating point** the win
    inverts on every stratum. Matched precision (~0.83): overall **0.324 vs 0.530**, faint
    **0.189 vs 0.326**, high-q **0.263 vs 0.432**. Matched FP/img (~11): overall **0.381 vs 0.528**,
    high-q **0.309 vs 0.432**. ssl1 dominates physics at every precision target (0.60–0.85) on
    overall/faint/high-q. Physics dilution made the detector trigger-happy and *worse* precisely on
    the high-q stratum it was built to fix (−0.17 high-q recall at matched precision).
  - **Conclusion:** the seventh single-variable lever to fail (after Semi-DETR, Boost, CCTM, Co-DINO,
    style-match, struct-noise). Physical peak-configuration priors from real crystallography did not
    close the sim2real gap — the ceiling is representation/sensitivity, not peak realism. Deployed
    best unchanged = **ssl1 + baseline ensemble** (organic 0.605 / 41 0.780). Runs
    2701978/2701979 and `bank.npz` retained; no code reverted (all physics paths are opt-in and
    default off, so main is byte-identical at inference).

## Q. Label-completeness diagnostic (Step 1) — **VERDICT RETRACTED; the eval is NOT label-limited** (2026-08-05, retracted 2026-08-23)
New plan after the 7 declined levers: before spending another training lever, check whether the
organic GATE itself is trustworthy. Prior hint ("Diagnostics & roadmap" KEY FINDING): the best
single model's confident FPs sit ON rings. This quantifies it on the *deployed* model.
- **NEW `diagnostics/label_completeness.py`** — runs the deployed ENSEMBLE (ssl1 + baseline,
  detection-level NMS fusion, exactly `ensemble_eval.py`) on organic; classifies each unmatched
  detection (FP) as ON-RING (q-dist to nearest GT peak < 8px AND I(q) percentile > 0.5 → candidate
  real unlabeled peak) vs OFF-RING (candidate genuine error); reports standard vs label-adjusted
  precision + an expert-review montage (`diagnostics/label_completeness.png`). GPU sbatch:
  `tmp_diag/run_label_completeness.sbatch` (job 2721947).
- **RESULT (organic, 817 GT, score>0.3):** recall 0.605, precision(standard) **0.764**,
  precision(label-adjusted, on-ring FPs as ignore) **0.890**. On-ring share of FPs 0.60; of
  HIGH-confidence (>0.5) FPs **0.74**; median FP q-dist to nearest GT peak **1.8px**; only 10% of
  FPs off-ring (>20px). Montage: on-ring FPs land on real arcs at unlabeled χ; off-ring FPs are the
  spurious tall segment-bars / low-q noise.
- **VERDICT RETRACTED (2026-08-23).** The original conclusion — "the organic eval is label-LIMITED,
  true precision ~0.89 vs measured 0.76" — is **WRONG**. It rested on the assumption that an
  unmatched detection landing on a real ring is an unlabeled real peak. **The organic labels are
  COMPLETE: there are no unlabeled peaks** (user, 2026-08-23; cf. the standing fact that the
  hand-labelled peaks are correct and not over-segmented). The "label-adjusted precision 0.890" is
  therefore meaningless and must not be quoted.
- **What the measurement actually shows.** Precision on organic is genuinely **0.764**. The on-ring
  unmatched detections are **real false positives**: the model fires at ring positions where no peak
  exists. That makes them a MODEL problem, and a sizeable one — 0.60 of all FPs and **0.74 of the
  high-confidence (>0.5) FPs** are of this kind, with a median q-distance of only **1.8 px** to the
  nearest labeled peak.
- **Reading it correctly:** the model puts boxes at very nearly the right *radial* position (1.8 px in
  q) but at a χ where nothing is. Combined with the phase V/W finding that χ-separation is where the
  model fails, this is consistent with a general weakness along χ — merging genuine close pairs in χ
  *and* emitting spurious detections at wrong χ on a correct ring. **Not yet established:** the FP
  χ-distance distribution was never measured, only the q-distance. That measurement is cheap and
  should be made before the connection is claimed.
- **CONSEQUENCE for every future lever:** the earlier guidance ("raw AP can stay flat on a genuine
  improvement, so judge on the label-adjusted view") is void — AP and precision on organic mean what
  they say. Keep the matched-operating-point discipline (phase-P lesson) for a different reason: score
  distributions differ between models, not because the labels are incomplete.

## R. Higher input resolution 512×2048 (Step 2) — FROM-SCRATCH — **DECLINED (8th lever negative)**
First Step-2 "representation sensitivity" lever (`docs/HIRES_INVESTIGATION.md`). Attack the faint/
high-q sensitivity ceiling directly: double the q-axis resolution (1024→2048; χ/HEIGHT unchanged) so
faint/small high-q peaks get 2× the samples. **RAW-DATA GATE passed:** organic native q-image is
1641×1641, 41 is 1350×1350 — both finer than 1024, so 2048 exposes REAL detail (verified: real
frames resample crisply, GT boxes on peaks full-range, `tmp_diag/hires_real_montage.png`).
From-scratch (SSL-backbone init + random head + ssl1 recipe; single variable vs ssl1 =
q-resolution). NEW MODEL LINE (ONNX input → (1,1,512,2048); the 512×1024 deployment is untouched).
- **Config-gated, byte-identical to all prior runs at 1024:** new key `polar_shape=[512,2048]`
  (`config/DINO/DINO_4scale_swin_hires.py`) drives both the sim (`simulation.HEIGHT/WIDTH` via the
  `main.py` `SimulationDataset`) and the real-eval resample (`evaluate_giwaxs_ap`). Absent elsewhere
  → [512,1024].
- **`simulation.py` (5 edits, each byte-identical at WIDTH=1024):** (1) removed the dead global-WIDTH
  reset that force-clobbered WIDTH→1024; (2) hardcoded bg ring box `[116,0,128,512]` → q-coords ×
  WIDTH/1024, χ=HEIGHT; (3) quazipolar image-mask factor `(1-(WIDTH-512)/1024)` → `512/WIDTH`; (4)
  quazipolar BOX-CLAMP factor (same) — **THE label-corruption hazard**: old factor goes NEGATIVE at
  2048 → inverted boxes → matcher crash (phase-P bug 4); `512/WIDTH` is the resolution-invariant form
  (0.5@1024, 0.25@2048); (5) detector-gap radius `self.rs`/`ws` × WIDTH/1024 (were absolute → clipped
  a different SEGMENT set at 2048, shifting the ring/segment mix). Model/transformer/PE/postproc: NO
  change (resolution-agnostic; SSL backbone keys match at 2048).
- **Pre-launch verification** (`tmp_diag/hires_smoke.py`, `hires_compare.py`; jobs 2721954/56/59/63):
  [A] 8436 boxes/200 draws, **0 inverted / 0 oob**; [B] synthetic (C,512,2048) in [0,1], boxes in
  range; [C] organic+41 resample to (1,1,512,2048), GT 0 inverted/0 oob, boxes on peaks; [D] real
  train step batch=2 peak **8.4GB/40GB** → batch stays 2 (no confound); [E] 1024-vs-2048 distribution
  PASS (zero_frac diff 0.03, box-align rel-diff 0.15, q-hist L1 0.16, box count matched after edit 5).
- **Config + launcher:** `config/DINO/DINO_4scale_swin_hires.py`, `backbone_curation/ssl/
  run_detector_hires.sbatch` (from-scratch, 500ep, lr-drop 280, `--exclude=galvani-cn203`). Run
  **`dino_hires1`, job 2721965** (launched 2026-08-05, auto-resubmit chain 2721990–93 across the 72h
  walls). GATE = organic/41 AP + faint/high-q recall probe vs ssl1 (0.586/0.762) AND the deployed
  ensemble (0.605/0.780); apply the phase-Q label-limit caveat when reading AP.

### VERDICT: NEGATIVE — declined (cancelled at ep405/500, 2026-08-10)
**AP gate (post-lr-drop plateau, ep300–404 mean):** organic **0.444** vs ssl1 0.563 (**−0.12**);
41 **0.641** vs ssl1 0.745 (**−0.10**). hires best-ever organic 0.468 @ep256 / 41 0.680 @ep250, both
below its OWN pre-drop values — the lr-drop produced no gain. The deficit was ~0.13 at ep82 and
~0.12 at ep404: **it never closed at any point in the schedule**, so this is not an undertraining
artifact. Remaining 95 epochs at lr 1e-6 could not move it; run cancelled to free the GPU.

**Decisive recall probe** (`tmp_diag/hires_probe.py`, job 2730643; ep405 snapshot; organic; matched
operating point per the phase-P calibration lesson — ssl1@0.30 det=522 vs hires@0.15 det=538):

| stratum | ssl1 (1024) | hires (2048) | Δ |
|---|---|---|---|
| **high-q third** | 0.434 | **0.236** | **−0.198** |
| low-q third | 0.567 | **0.685** | **+0.118** |
| mid-q third | 0.585 | 0.549 | −0.036 |
| **ring** | 0.833 | **0.583** | **−0.250** |
| segment | 0.528 | 0.474 | −0.054 |
| faint (vis=1) | 0.330 | 0.271 | −0.058 |
| recall / precision | 0.537 / 0.841 | 0.477 / 0.725 | −0.060 / −0.116 |

**The lever's own mechanism is refuted, not merely unproven.** The hypothesis was "2× q-samples →
better high-q sensitivity". The measured effect is the OPPOSITE and is *stratum-specific*: high-q
recall roughly halved (−0.198) and ring recall collapsed (−0.250), while **low-q recall IMPROVED
(+0.118)**. Doubling the q-axis did not extend sensitivity outward — it redistributed it toward low
q. Probable cause: a peak's fixed integrated intensity is spread over 2× the q-pixels, so per-pixel
contrast DROPS, which hurts precisely the faint low-contrast high-q peaks the lever targeted; rings
(already the most q-elongated objects) become 2× wider against an unchanged query budget.

**The phase-Q label-limit caveat cannot rescue this**: unlabeled real peaks inflate FPs, i.e. they
depress *precision*, but the failure here is in **recall**, which is computed against labeled GT and
is immune to un-annotated peaks. Both the AP gate and the decisive probe agree.

**Transferable lesson (contradicts the pre-registered rationale):** "the raw data carries finer
detail than the grid exposes" (native 1641×1641 / 1350×1350) was a *necessary* condition for this
lever and was correctly verified — but it is NOT sufficient. Resolution helps only if per-pixel
CONTRAST survives the resample; for spread-out low-contrast features it is actively harmful. Any
future resolution lever must gate on measured post-resample contrast in the target stratum, not on
native-vs-grid sampling alone. Follow-ups from the Step-1/2 plan remain open: Step 3 = dense
heatmap/segmentation head, Step 4 = physics analysis-by-synthesis.

## S. Prominence probe — the residual recall gap is NOT contrast-limited (2026-08-18)

**Motivation.** Eight consecutive levers declined (phases I–R), and the last one failed specifically
on *recall* of faint/high-q peaks. Before spending a ninth run, measure whether the peaks the
deployed ensemble misses are physically distinguishable in the input at all. If they are not, the
eval is at the information floor of the data and every further modelling lever will also fail.

**Method** (`diagnostics/prominence_probe.py`, job 2754438, ~4 min on one A100). For every labeled
peak, its **topographic prominence** = 0-dimensional superlevel-set persistence: sweep a threshold
downward; each local maximum births a component at its own height; when two merge the ELDER (higher
birth) survives and the younger dies at the merge level, so `prominence = birth − death`. This is
"how far must I descend from this summit before reaching ground that leads somewhere higher" — a
measure of how far a peak stands out from its *local* surroundings, independent of the absolute
background level. No smoothing and no threshold needed: a noise spike simply births a component
with negligible prominence. Measured on the exact HE'd image the network receives (which is
quantised to ~110–140 distinct levels, so every distinct value is used as a sweep level and the
persistence is **exact**, not approximated). Model = deployed ensemble, score>0.3, same q-matcher
as every prior probe. Core routine verified against hand-computed cases
(`tmp_diag/prom_core_selftest.py`), including a 0.9-high bump on a 0.5-high hill next to a taller
peak: naive height says 0.83, correct prominence is 0.33, the code returns 0.33.

**Result — the information-floor hypothesis is REJECTED.**

| set | peaks | recall | separation AUC | median prominence found | missed |
|---|---|---|---|---|---|
| organic | 817 | 0.605 | **0.489** | 0.208 | **0.247** |
| 41 | 1680 | 0.834 | **0.597** | 0.259 | 0.173 |

AUC = P(a random *found* peak is more prominent than a random *missed* one); 0.5 means prominence
explains nothing. On organic it is **0.489** — below 0.5, i.e. the missed peaks are if anything
*more* prominent than the found ones. Detection rate is flat right across the prominence deciles
(0.54–0.66, no trend). On 41 the only dip is the faintest decile (0.599 vs ~0.85 elsewhere).

**What does predict recall: the annotator's confidence, at matched prominence.**

| organic | n | recall | median prominence |
|---|---|---|---|
| conf 0.1 | 291 | 0.375 | 0.235 |
| conf 0.5 | 132 | 0.621 | 0.233 |
| conf 1.0 | 394 | 0.769 | 0.200 |

Recall doubles across the tiers while prominence is flat — the low-confidence peaks are the *more*
prominent ones. Cross-tabulating settles it: within each prominence tercile confidence still
separates hard (organic brightest tercile: 0.326 vs 0.814), while prominence does ~nothing within a
confidence group (organic conf=1.0 across terciles: 0.755 / 0.729 / 0.814). Same pattern on 41
(brightest tercile 0.663 vs 0.977).

**Headroom.** Misses more prominent than the median *found* peak: 173 on organic (**+0.212** of GT),
119 on 41 (**+0.071**). If conf=0.1 peaks reached the conf=1.0 rate: organic 0.605→0.745,
41 0.834→0.934.

**organic high-q INVERSION.** High-q peaks there are the *most* prominent (0.290 vs 0.078 at low q)
yet have the *worst* recall (0.475 vs 0.685). Phase R's founding premise — "high-q peaks are faint,
more q-resolution will help" — was wrong about organic from the start, consistent with its failure.

**Free confirmation of phase Q.** Unmatched detections are not noise: a typical one is more
prominent than 35–40% of *labeled* peaks.

**Transferable lesson.** Nearly every declined lever (hires, struct-noise, style-match, q-decay,
physics-sim realism) attacked CONTRAST or appearance realism. This probe says that axis is not
binding — a coherent post-hoc explanation for the whole 8-negative streak. Do not build another
contrast lever.

**Caveats.** Prominence is measured in the histogram-equalised model input — the correct domain for
"can the network see it", but not raw physical contrast. organic is only 8 frames (817 peaks), so
its effective n is well below 817. The 41 ring/segment split in the probe output is INVALID (n=2):
`H5GIWAXSDataset` never populates `polar_labels.is_ring`; organic's (24 rings) is fine.

Figure: `diagnostics/prominence_probe.png`. Per-peak records: `tmp_diag/prominence_{organic,41}.npz`.

## T. Near-miss probe — the gap is a χ-SEPARATION problem, not sensitivity (2026-08-18)

**Motivation.** Phase S established there is headroom but not what to build: a missed peak can be
lost at four different places, each needing a completely different fix.

**Method** (`diagnostics/nearmiss_probe.py`, job 2754964). Replay the deployed ensemble's own
stages — `900 queries × 2 classes → top-225 → pooled 450 → NMS → score>0.3` — and for every missed
GT record the best score of a *compatible* box (the q-matcher's own criterion: IoU>0.1 and
|Δq|<10 px) at each stage, then bucket it. Sanity check passed: recall@0.30 reproduces phase S
exactly (0.605 / 0.834), so the staging is faithful.

**Result — the model responds at most of the peaks it "misses".**

| bucket (organic, 323 misses) | n | median best score | meaning |
|---|---|---|---|
| `ASSIGNMENT` | 69 | **0.90** | qualifying box IS in the final output; Hungarian matching gave it to a neighbouring GT |
| `BELOW_THRESH` | 84 | 0.17 | survives NMS, scores under 0.30 |
| `RANK_CUT` | 82 | 0.07 | a query responded but did not survive top-225 |
| `NO_RESPONSE` | 74 | 0.00 | genuine blindness |
| `NMS_KILLED` | 14 | 0.41 | NMS removed it |

**77% (organic) / 70% (41) of misses have a model response somewhere.** True blindness is only
**9.1% of organic GT / 5.0% of 41 GT**. Median best raw query score: detected 0.853, missed 0.134.
On 41: `BELOW_THRESH` 112, `NO_RESPONSE` 84, `ASSIGNMENT` 46, `RANK_CUT` 30, `NMS_KILLED` 7.

**THE MECHANISM — 84.5% of organic misses sit within 8 q-px of a peak the model DID detect**
(91.3% within 8 q-px of *any* labeled peak). For `ASSIGNMENT` it is absolute: 100% have a same-q
neighbour, 94% of those were detected, and the median separation is **3.9 px in χ** — while GT
boxes are only ~**8 px tall in χ** (of 512). Azimuthally adjacent labels overlap heavily, the model
emits ONE box, and the matcher scores the other peak as a miss.

Two populations: tightly-packed χ siblings (`ASSIGNMENT`+`NO_RESPONSE`+`NMS_KILLED` = 49% of
misses, Δχ ≈ 4–6 px) and well-separated same-ring peaks (`BELOW_THRESH`+`RANK_CUT` = 51%,
Δχ ≈ 36–52 px, unambiguous and about scoring/ranking).

**The labels are correct.** Two labels 3.9 px apart in χ are 0.69° apart, which raised the question
of annotation over-segmentation. **Confirmed by the user (2026-08-18): the fitted peaks in the
verification set are correct, hand-labeled, not over-segmented.** The tight-χ population is
therefore a real model defect, not a labeling artifact — the separation failure stands.

**This explains phase R.** The binding constraint is separation along **χ**. Phase R doubled **q**
(1024→2048) and deliberately held χ at 512 — the wrong axis — and paid per-pixel contrast for it.
The observed halving of high-q recall follows directly.

**Operating point.** Dropping 0.30 → 0.05 gives recall 0.605→0.720 (organic) and 0.834→0.911 (41),
and phase Q means the measured precision cost is overstated. **But this does NOT move the AP gate** —
AP integrates over thresholds, so re-thresholding is a deployment decision about the shipped
product, not a lever result. Do not report it as a metric win.

**Not worth pursuing.** `RANK_CUT` boxes are weak (only 23/82 organic above 0.1), so raising
`num_select` recovers little. NMS tuning is worth ~4% of misses. **Live inconsistency noted:**
`util/postprocessing.onnx_to_xyxy` hardcodes `num_select=225` while `config/DINO/DINO_4scale_swin.py`
sets `num_select = 150`.

**INDICATED NEXT LEVER (#9, not yet run): raise the χ/HEIGHT resolution, NOT q.** The χ axis is
512 px over 90° = 0.176°/px; at feature strides 8/16/32 an 8-px box is ≤1 feature pixel, so the
network physically cannot separate two peaks 3.9 px apart. Reuse phase R's config-gated
`polar_shape` machinery (it already generalises). Pre-register the gate as the *separation* metric
this probe measures — `ASSIGNMENT`-bucket count and recall among same-q sibling peaks — not only
organic/41 AP. Raw-data gate to verify first: at radius r px the native angular sampling is ~1/r
rad/px, so finer χ is supported at mid/high q but not near the beamstop.

Per-peak records: `tmp_diag/nearmiss_{organic,41}.npz`.

### T2. Does this DETR need NMS at all? — **NO CHANGE; NMS is not the bottleneck** (2026-08-18)

Direct follow-up to T. NMS encodes "high overlap ⇒ duplicate", which is false for genuinely
adjacent peaks: two 8.5 px boxes at separation *d* have IoU (8.5−d)/(8.5+d), crossing the deployed
`POSTPROCESSING_NMSIOU_SEG = 0.4` at d ≈ 3.6 px. So NMS cannot be tuned to keep real close pairs and
drop real duplicates — only to trade one for the other. DETR-family models are trained with
one-to-one Hungarian matching so that duplicate suppression is *learned*; DINO's reference inference
runs no NMS. `diagnostics/sweep_nms.py` had only ever swept DOWNWARD (0.4→0.1); the loosening
direction was untested.

`diagnostics/nms_sweep_single.py` (job 2755004), **single model ssl1, no ensemble** — caches each
frame's pre-NMS top-225 once, then re-runs NMS at 11 settings including fully off.

| organic | ap_total | recall | precision | tight-pair recall (χ-gap <5 px, n=165) |
|---|---|---|---|---|
| seg_iou = 0.40 (deployed) | 0.5683 | 0.537 | 0.841 | 0.352 |
| seg_iou = 0.70 (best AP) | 0.5710 | 0.545 | 0.833 | 0.352 |
| NMS FULLY OFF | 0.5413 | 0.548 | 0.797 | 0.358 |

| 41 | ap_total | recall | precision | tight-pair recall (n=176) |
|---|---|---|---|---|
| seg_iou = 0.40 (deployed) | 0.7441 | 0.772 | 0.705 | 0.449 |
| NMS FULLY OFF | 0.6902 | 0.774 | 0.664 | 0.455 |

**VERDICT: keep NMS at the deployed settings.** Removing it entirely buys **+0.006** tight-pair
recall on both sets while costing **−0.027 / −0.054 ap_total** and ~4 points of precision. The AP
optimum (organic 0.70, 41 0.20–0.30) is within noise of 0.40 on 8 / 41 frames, so no change is
warranted. Postprocessing and the ONNX path stay untouched.

**What this proves, and it is the useful part:** the model is not emitting a second box for tight
pairs *at all* — there is nothing for NMS to suppress. Combined with T (`NMS_KILLED` only 4.3% /
2.5% of misses), the fix must be UPSTREAM of postprocessing. That is what phase U tests.

Incidental: top-k duplicate queries (one query selected as BOTH classes, which per-class NMS cannot
remove) are only 0.9% / 0.8% of selections — a non-issue.

**Single-model ssl1 baseline established here** (`checkpoint.pth` = last epoch, so slightly below
the recorded best-AP 0.586): organic ap_total **0.5683** / ap_high 0.7004; 41 ap_total **0.7441** /
ap_high 0.8569. This is the control for phase U — all prior ensemble numbers are superseded for
research purposes (user directive 2026-08-18: single model only until further notice).

## U. Azimuthal peak clusters in the simulator (Step 3) — FROM-SCRATCH — **DECLINED (9th lever negative)**

**Motivation (mechanism established, not guessed).** Phases S/T showed the recall gap is a
peak-SEPARATION failure along χ: 84.5% of missed peaks sit within 8 q-px of a peak the model DID
detect, median χ-separation 3.9 px against ~8.5 px-tall boxes. T2 then ruled out postprocessing.
That leaves the training data.

**Root cause, located in `simulation.py`.** The stock simulator *structurally cannot* produce a
tight azimuthal cluster — two independent mechanisms remove it:
1. `add_peaks_on_rings` places peaks on a fixed `[0, 1/6, 2/6, 3/6]` grid of the ring's χ-extent,
   gated to rings with `max_a_width >= 100` (≥200 px tall), so consecutive peaks are **≥33 px
   apart**; it fires on only **10% of images** and caps at **4 peaks per ring**.
2. `filter_nms` with `min_ring_seg_nms = 0.0` deletes *any* overlapping label pair, on boxes
   inflated to ±3.5·`a_width` in χ.

**Measured target (the spec).** Real labeled data, same ±8 q-px tolerance as the evaluation matcher:

| | organic | 41 |
|---|---|---|
| peaks sharing a q-position | **85.9%** | 54.9% |
| largest cluster | 11 peaks | 8 |
| median χ-gap in a cluster | 49.5 px | 93.4 px |
| gaps **< 5 px** | **12.5%** | 13.2% |
| gaps < 33 px (below the simulator's floor) | 34.9% | 23.6% |
| GT box χ-height | median 8.5 px | median 79.3 px |

Box heights are already right (`max_a_width/40` yields 5–10 px vs a real median of 8.5), so only
**spacing, frequency and count** change.

**Design** — `use_peak_clusters`, opt-in, default off (`sim_config=None` ⇒ byte-identical to every
prior run). `FastSimulation.add_peak_clusters` spawns same-q azimuthal siblings for a fraction of
segment peaks, appended **after** the generation-time NMS exactly as `add_peaks_on_rings` already
does — so no existing filtering behaviour changes. Sibling count from the measured cluster-size
histogram (up to 7 extras vs the stock cap of 4); χ-offsets are a segmented cumulative sum so a
cluster spreads along the ring rather than stacking. Seed probability is normalised by
`cluster_extra_ratio` so the peak count is independent of `obj_num`.

**Leakage discipline.** The χ-gap prior is a **parametric** two-component mixture (18% uniform
1–7 px; lognormal median 55 px, σ 0.95) with constants rounded from **organic only** — not an
empirical resample of eval labels. **41 is therefore an uncontaminated gate.** (45.h5 would have
been the clean third set but is gone, and the user directed it be ignored on 2026-07-20.)

**Control** = ssl1 under identical settings (T2): organic ap_total 0.5683 / 41 0.7441; tight-pair
(<5 px) recall 0.352 / 0.449.

**PRE-REGISTERED GATE.** AP over all peaks will barely move even if the lever works — the tight
population is only ~12% of gaps — so AP alone is the wrong read:
- **PRIMARY: same-q sibling recall by χ-gap bucket** (<5, 5–10, 10–20, 20–33, ≥33 px) at a
  **matched operating point**, vs the control table above. Success = a material gain in the <5 and
  5–10 px buckets.
- **SECONDARY: organic/41 ap_total must not regress.**
- Matched-operating-point comparison throughout (phase-P calibration lesson).
- **41 carries the verdict** (uncontaminated); organic is secondary.

**Pre-launch verification** (`diagnostics/verify_clusters.py`, hard gates; the training job is
submitted `--dependency=afterok` so a failure cannot start the run): G1 the default path is
bit-identical to pre-patch `simulation.py` under matched RNG seeds; G2 the synthetic χ-gap
distribution matches the real one (p10/p25/p50/p75 within 2×, frac<5 px in [0.06, 0.25]);
G3 clustering present with size ≥5; G4 box sanity and peaks/frame; G5 rendered signal actually
appears inside the new boxes; G6 `SimulationDataset` smoke through the real config.
Figure: `diagnostics/verify_clusters.png`.

Files: `simulation.py` (gated additions only), `main.py` (one gated block),
`config/DINO/DINO_4scale_swin_clusters.py`, `backbone_curation/ssl/run_detector_clusters.sbatch`
(chained, completion-guarded), `diagnostics/verify_clusters.py`.
Run: `detector_runs/dino_clusters1`, 500 ep, lr-drop 280, 512×1024.

### VERDICT: NEGATIVE — declined (cancelled at ep446/500, 2026-08-21)

**The pre-registered primary gate fails.** `diagnostics/clusters_gate.py` (job 2764300) compares
ssl1 vs `dino_clusters1` at a **matched operating point** — sweeping the score threshold per model
and comparing where the detection count (and separately the precision) matches.

Organic, matched detection count (ssl1@0.30 = 65.2/fr ↔ clusters@0.50 = 64.1/fr):

| χ-gap | n | ssl1 | clusters | Δ |
|---|---|---|---|---|
| **< 5 px** | 165 | 0.352 | 0.321 | **−0.030** |
| **5–10 px** | 52 | 0.423 | 0.385 | **−0.038** |
| 10–20 px | 59 | 0.390 | 0.458 | +0.068 |
| 20–33 px | 92 | 0.620 | 0.576 | −0.043 |
| ≥ 33 px | 358 | 0.612 | 0.567 | −0.045 |

41 (the uncontaminated gate), matched at 44.9 ↔ 45.0/fr: `<5` **+0.028**, `5–10` **−0.080**,
`10–20` 0.000, `20–33` +0.054, `≥33` −0.003. Opposite signs to organic, all within a handful of
peaks (organic `<5` n=165 ⇒ −0.030 is ~5 peaks). **No effect on the target stratum.**
At matched PRECISION organic is worse still: every bucket negative, `<5` −0.091.
Secondary gate also fails: ap_total −0.024 organic / −0.040 on 41.

**The trap this avoided.** At the *deployed* threshold the lever looked like a clear win — every
χ-gap bucket up, `<5` +0.030, `10–20` +0.220. But the cluster model emits **99.9 det/frame vs
ssl1's 65.2** at that threshold, with precision 0.635 vs 0.841. The lift appeared in EVERY stratum
including `≥33 px`, which this lever cannot plausibly affect — the tell that it was a calibration
artefact, not a real gain. Reading the fixed-threshold table would have recorded a false positive.
This is the phase-P lesson holding under a case where it mattered.

**What the model actually learned.** It emits 53% more detections but is no better at separating
close pairs at equal volume. Training on tight pairs taught it *that more peaks exist*, not *how to
resolve them*.

**Status of the χ-resolution hypothesis — ELIMINATIVE, not positive.** Phase U rules out the
training-data explanation ("the model never saw tight pairs"), which was the cheaper of the two
candidates. It does NOT independently demonstrate that finer χ sampling would help. The remaining
argument is architectural and a priori: `return_interm_indices = [1,2,3]` makes the finest feature
level stride 8, so peaks 3.9 px apart fall inside one feature cell. Suggestive but weak supporting
detail: on organic the only positive bucket is `10–20 px` (+0.068, n=59 ⇒ ~4 peaks) — 1–2.5 feature
cells, i.e. resolvable — while `<5 px` (half a cell) is negative. Too small to lean on.

**Do NOT launch a resolution run on this basis.** Nine levers have now been declined, several of
them on a priori mechanism arguments that measurement later refuted (phase R most directly). The
next step is the cheap discriminating test in phase V below, not another multi-day run.

**Artefacts.** `diagnostics/clusters_gate.py` is the reusable matched-operating-point gate for the
sibling strata — use it for every future lever aimed at peak separation. Run kept at
`detector_runs/dino_clusters1` (ep446); snapshot at `tmp_diag/clusters_probe_ckpt.pth`.
The simulator changes stay in-tree, gated off by default (`use_peak_clusters=False`), so the
default path is unaffected — gate G1 proved bit-identity.

**Two real bugs found while building this, both fixed and both pre-existing hazards:**
1. `simulate_img` called `self.__init__()` with NO arguments on 50% of calls, resetting
   `sim_config` to defaults and `device` to `'cuda'` — it would silently discard ANY configured
   simulation on half of all images. Any future sim-config lever would have hit this.
2. `img_from_labels` divides by `a_widths ** 2`, and `clamp_boxes` runs AFTER the `y1 < y2`
   validity filter, so a box lying wholly outside the image clamps to ZERO height ⇒ `0/0` ⇒ NaN ⇒
   `simulate_img` silently discards and regenerates the frame. Unfiltered, 31% of siblings landed
   out of χ range and NaN'd 2 frames in 3, biasing survivors sparse. **This landmine is still live
   for anyone who generates an out-of-range label.**

**EXONERATED AS A FAIR TEST (2026-08-24, phase X.3).** A recurring worry about this verdict was that
the lever might have been under-dosed — that the simulator never really produced close pairs. Measured
directly: with clusters ON the training χ-gap distribution reaches `<5 px` **0.177** vs real **0.125**
(base simulator: 0.029). Phase U slightly OVERSHOT the real rate. The negative verdict therefore stands
as a genuine test of "train on close pairs", and the conclusion is stronger than first written: the
model saw close pairs at 1.4× the real rate and still merges them (phase X.2/X.3 measure clusters1's
proposal boxes at 1.84 px apart for a true 8 px gap — 2.4× better than ssl1's 0.77, still far wrong).
Training exposure is necessary but NOT sufficient. Related: relaxing the simulator's NMS is a no-op —
the base generator places peaks independently at random and simply never creates close pairs, so
`add_peak_clusters` is the only mechanism that produces them (see phase X.3).

## V. Synthetic separation ladder — **RUN; KILLS BOTH RESOLUTION ROUTES** (2026-08-21)

Before spending 3 days on 512×1024 → 1024×1024, measure the architecture's actual resolution limit
directly. Feed **synthetic** frames containing peak pairs at controlled χ-separations
(2, 4, 6, 8, 12, 16, 24, 32 px) and measure, per separation, whether the model emits TWO boxes or
one. No training; ~1 GPU-hour.

Run it on **`dino_clusters1`**, which was trained on exactly this distribution — that is what makes
it decisive:
- limit ≈ 8 px (one feature cell) and clean above it → stride-limited ⇒ finer χ should move the
  limit to ~4 px ⇒ the resolution lever is justified;
- limit ≫ 8 px, unrelated to stride → something else binds; resolution will not fix it;
- **resolves 4 px pairs fine in SYNTHETIC but not in real data** → the limit is not resolution at
  all but the sim-to-real gap, and the resolution lever would be the 10th wasted run.

The third outcome is the one that makes this worth running first, and no measurement so far can
rule it out.

### RESULT — the limit is ~16 px and NOTHING we control moves it

`diagnostics/separation_ladder.py` (jobs 2764330, 2764332). Two peaks planted at a known χ-gap,
8 pairs/frame × 20 frames × 10 rungs, rendered through the FULL appearance pipeline (patch
`simulate_labels`, call the real `simulate_img`), box size = measured real organic GT medians
(q-width 10.6 px, χ-height 8.5 px). One fixed image set for every model. Matched operating point =
the threshold whose detection count is nearest the true planted-peak count.

| χ-separation | ssl1 | clusters | 5scale (stride-4) |
|---|---|---|---|
| 4 px | 0.130 | 0.087 | 0.203 |
| 8 px | 0.037 | 0.029 | 0.059 |
| 12 px | 0.049 | **0.392** | 0.273 |
| **16 px** | 0.662 | 0.654 | 0.579 |
| 32 px | 0.761 | 0.694 | 0.716 |
| 64 px | 0.702 | 0.718 | 0.580 |

**RESOLUTION LIMIT (first rung ≥0.5 resolved): 16 px for ALL THREE MODELS.**

**Below 16 px the failures are MERGES, not misses** (0.52–0.87 merged) — the model sees the feature
and emits a single box. A pure resolution failure.

**What this kills.**
1. **The χ-resolution lever as scoped (512→1024) would NOT have worked.** Real gaps are 3.9 px
   median; doubling makes them 7.8 px, still far below the 16 px wall. You would need **4×**
   (χ=2048, ~4× the pixels, ~2 weeks) to reach it. Had we launched on the a priori stride argument
   this would have been the 10th negative lever.
2. **Feature stride is NOT the mechanism.** `dino_5scale_scratch` (`return_interm_indices=[0,1,2,3]`,
   a STRIDE-4 level, same backbone init and recipe as ssl1, same 512×1024 grid) has the *same*
   16 px limit — 4× the feature resolution changed nothing, and it is worse at large separations.
   This re-examines a model previously declined on AP, which we now know cannot see this effect.
3. **Training is not the mechanism** — clusters1 saw this exact distribution (phase U). Its only
   gain is at 12 px (0.392 vs 0.049), i.e. right at the boundary: training helps you exploit
   resolution you have, it cannot create resolution you lack.

**Also eliminated:** NMS (two 8.5 px boxes at 8 px separation have IoU 0.03, far under the 0.4
threshold — though NMS DOES explain the 2 px rung, IoU 0.62); and image information (at 8 px the
rendered Gaussians, σ≈2.4 px, are 3.3σ apart and plainly separable in the pixels, yet 87% merge).

**Sim-to-real is not the escape either:** converting to per-peak recall, synthetic 4 px gives ssl1
0.49 vs real `<5 px` 0.352 — same ballpark, so the model behaves on synthetic tight pairs much as
on real ones.

**MEASUREMENT CAVEAT.** ssl1 gave 0.176 (job 2764330) then 0.130 (2764332) at 4 px: run-to-run
noise ≈ ±0.05 at small separations, because `np.random` is not seeded per frame in the appearance
pipeline (only `random` and `torch` are). The 16 px crossing is stable (0.664 / 0.662) so the LIMIT
is solid, but small-separation deltas are not — in particular the 5-scale "gain" at 4 px
(0.203 vs 0.130) is **within noise and is not claimed**. Fix the seeding before reusing this script
for fine comparisons.

**OPEN — the mechanism is now genuinely unknown.** Not training, not stride, not NMS, not image
information. Something in the detection head collapses two clearly-separable peaks into one output.

### V.1 Localisation discriminator — the failure is genuine perception (2026-08-22)

The `merged` bucket conflated two failures needing opposite fixes: (a) the model emits **ONE** box
(query/assignment capacity) vs (b) it emits **TWO** both piled near the midpoint, so the one-to-one
matcher can only claim one (regression precision). `localisation()` counts raw detections in each
pair's neighbourhood **before** matching and records their separation.

**Verdict: (a).** At 8 px χ the split is **0.94 one-box / 0.02 two-box** — the model simply does not
emit a second detection. And regression is *not* the weak link: above the wall the two boxes land at
**31.9 vs a true 32** and **64.0 vs a true 64**, sub-pixel. Below 8 px, when two *are* emitted they
sit **0.2 px apart** — stacked duplicates that survive only because class-aware NMS never compares
them (one is scored ring, one segment, so they go through different IoU pools).

So this is a real resolution failure, not a matcher artefact and not a regression-precision problem.
Correction to an earlier framing of mine: I had said the two-box branch would show "this was never a
resolution problem" — the discriminator showed the opposite. It **is** a resolution problem, just not
one that stride or training moves.

### V.2 q-axis ladder — CONFOUNDED, decides nothing (2026-08-22, superseded by V.3)

The only strongly anisotropic thing in the architecture is the **Swin window**: `window_size_h = 48`
(χ) vs `window_size_w = 6` (q), 8:1, and `window_partition` confirms `h` indexes χ. If the window sets
the wall, the q limit should be ~8× finer than the χ limit. No pre-window-change checkpoint exists
(all four have a 1045-row `relative_position_bias_table` = 48×6), so the free test is the q ladder.

Raw result — RESOLUTION LIMIT on q: **ssl1 32 px, clusters 24 px, 5scale 24 px**, all *coarser* than
χ's 16 px. Read literally that refutes the window. **It does not, because the stimulus was wrong.**

`img_from_labels` (`simulation.py:816`) uses an **anisotropic box→Gaussian convention**:

    sigma_q   = box_width  / w_coef     w_coef = 1.0
    sigma_chi = box_height / a_coef     a_coef = 3.5

so the real median box (10.6 × 8.5) renders a peak **4.4× broader along q** (σ_q = 10.6) than along
χ (σ_χ = 2.43). The q-separated pairs were a much harder stimulus for reasons having nothing to do
with the model. The two readings disagree in sign:

| separation in units of the rendered σ | χ resolved | q resolved |
|---|---|---|
| ~3.0–3.3 σ | 0.037 (Δχ=8) | **0.778** (Δq=32) |
| ~4.5–4.9 σ | 0.049 (Δχ=12) | **0.740** (Δq=48) |
| ~6.0–6.6 σ | 0.662 (Δχ=16) | 0.677 (Δq=64) |

In raw pixels q looks worse; in σ units q looks ~2× **better** (resolves at ~3σ where χ needs ~6.6σ).
The conclusion depends entirely on the choice of normaliser, so **the experiment decides nothing** —
the window hypothesis is neither confirmed nor refuted. Supporting signs the two ladders are not
measuring the same thing: the q curve is shallow and gradual (0.233 at 2 px → 0.778 at 32 px) where
the χ curve is a sharp step (0.037 at 8 px → 0.662 at 16 px), and the q 1box/2box split is balanced
(≈0.47/0.41 at 2 px) where χ's is lopsided (0.94/0.02 at 8 px).

A second, smaller confound: the evaluation matcher is q-based (`get_matcher('q', min_iou=0.1)` —
costs on |Δq|, gates on IoU), so for χ-separated pairs the cost is degenerate and the IoU gate does
the work, while for q-separated pairs the cost is informative. The axes are not matched in matcher
difficulty either.

### V.3 Isotropic-stimulus ladder — the corrected anisotropy test (LAUNCHED 2026-08-23)

`LADDER_ISO=1` plants the box that renders an **isotropic** peak, σ_q = σ_χ = **2.43 px**
(box 2.43 × 8.5 = SIGMA·w_coef × SIGMA·a_coef), so both axes carry an identical stimulus and any
surviving difference is the model. SIGMA is the real χ width, which leaves the χ profile unchanged
from V — so the iso-χ run doubles as a **control that must reproduce the 16 px wall**.

`LADDER_NMS=0` additionally sets both class IoU thresholds to 1.0. The *boxes* stay anisotropic
(2.43 wide × 8.5 tall) even when the render is isotropic, and two boxes offset by d along an axis of
extent e have IoU = (e−d)/(e+d): at d = 2 px the χ pair sits at IoU 0.62 (suppressed) while the q
pair sits at 0.10 (kept). That asymmetry only bites at d ≤ 3 px, far below the 16 px wall, but the
NMS-off arm removes it so the wall can be confirmed on raw model output.

Caveat carried into it: the 2.43 × 8.5 box is out-of-distribution for the box **regression** target
(models were trained on ≈10.6 × 8.5). Acceptable, because the hypothesis under test — the Swin
window's 48:6 anisotropy — lives in the backbone, not the box head.

Jobs 2776881–2776884 = {χ, q} × {NMS on, off}, all three models, 2–2.5 min each.

**CONTROL PASSES.** The iso-χ arm reproduces the wall — ssl1 0.000 at 8 px, 0.282 at 12, **0.986 at
16** — and more cleanly than the anisotropic run: `missed` is 0.000 at every rung, so what is measured
is purely resolution, not detection.

**RESULT: a real but SMALL anisotropy, ~1.5–2×, nowhere near the window's 8:1.**

RESOLVED fraction, matched stimulus (σ_q = σ_χ = 2.43 px), NMS on = deployed:

| sep px | χ ssl1 | χ clusters | χ 5scale | q ssl1 | q clusters | q 5scale |
|---|---|---|---|---|---|---|
| 6 | 0.000 | 0.016 | 0.000 | 0.177 | 0.162 | 0.008 |
| 8 | 0.000 | 0.060 | 0.000 | **0.822** | **0.689** | **0.659** |
| 12 | 0.282 | 0.984 | 0.863 | 1.000 | 1.000 | 1.000 |
| 16 | **0.986** | 0.993 | 0.993 | 0.993 | 0.993 | 0.985 |

RESOLUTION LIMIT (first rung ≥ 0.5):

| arm | χ | q | ratio |
|---|---|---|---|
| NMS on (deployed) | 16 / 12 / 12 px | 8 / 8 / 8 px | ~1.5–2× |
| NMS off (control) | 12 / 12 / 12 px | 8 / 12 / 8 px | ~1–1.5× |

The direction matches the window (χ is the coarse axis, `window_size_h = 48`), but the magnitude is
**1.5–2×, not 8×**. The discriminator agrees and is sharper than the RESOLVED column, which is the
number to trust since it bypasses matching entirely: the one-box→two-box switch flips between 8 and
16 px on χ (ssl1 at 12 px: 0.72/0.28) and between 6 and 8 px on q (ssl1 at 8 px: 0.06/0.94). Above the
flip, localisation is sub-pixel on both axes (24.0/32.0/48.0/64.0 against true 24/32/48/64).

### V.4 — the window lever is DECLINED WITHOUT RUNNING IT (2026-08-23)

Not because the anisotropy is absent — it is real — but because **its ceiling is too low to matter.**

1. The best case for a squarer window is that χ becomes as good as q: a wall at **8 px**.
2. Real organic χ-gaps are **3.9 px median** (phase T), which is where recall is 0.352.
3. **8 px is still 2× above 3.9 px.** A perfect window fix would not move the pairs that are actually
   being missed.

So a 3-day retrain buys, at absolute best, a wall that is still on the wrong side of the data. That is
a quantitative decline, not a guess — and it is the second lever now declined by measurement instead
of by a training run (cf. phase O).

**Also killed by this run:** the window is not the *dominant* mechanism either. Something
axis-independent sets a floor at ~8 px on both axes, and it is not stride (5scale matches ssl1 on both
axes), not training (clusters matches ssl1), not NMS (the NMS-off arm shows the same walls), not
regression (sub-pixel above the flip) and not image information (at 8 px two σ=2.43 Gaussians are
3.3σ apart).

**Read the NMS-off small-separation rungs with care:** they carry a spurious floor of ≈0.15–0.20
RESOLVED at 2–8 px, because with suppression disabled two stacked near-duplicate boxes can both be
claimed by the one-to-one matcher. That floor is an artefact; the ≥0.5 crossings are not affected.

**Status: the mechanism remains open, but the resolution route as a whole is now closed** — stride,
input resolution, training distribution and window anisotropy have each been measured and each is
either ineffective or capped below the 3.9 px the data needs. Ten levers declined. Any further work
here should target *why the head refuses to emit a second query* below the wall, not the features
feeding it.

## W. Query-suppression probe — the second peak is **NEVER PROPOSED**, not suppressed (2026-08-23)

`diagnostics/query_suppression_probe.py`, job 2776912, 2 min, existing checkpoints, no training.

Phase V left one question: the head emits ONE box below the wall — is the second peak *scored down*
(classification suppression) or *never put forward*? The leading hypothesis was DINO's contrastive
denoising: with `dn_box_noise_scale = 0.4` (`dn_components.py:81-89`) negative DN queries sit
0.2–0.4 × box-dimension from truth — **1.7–3.4 px** for an 8.5 px χ box, right where real second peaks
live (median real χ-gap 3.9 px). If that were the mechanism a box would land on peak 2 and be scored
as background.

Because DINO is two-stage, both stages are readable — `out['interm_outputs']` is the encoder's
proposal, `out['pred_logits']` the decoder's output. All 900 queries read raw: no top-k (deployed
keeps 225), no NMS, no threshold, so a query at score 1e-4 is still visible.

**ssl1 — the answer is (C), and it is not close:**

| sep px | (A) suppressed | (B) dec absent | (C) enc absent | med score 2nd peak | med score 1st peak |
|---|---|---|---|---|---|
| 6 | 0.008 | 0.992 | 0.969 | 0.0050 | 0.0080 |
| 8 | 0.022 | 0.978 | 0.948 | 0.0031 | 0.0077 |
| 12 | 0.169 | 0.548 | 0.234 | 0.2144 | 0.1836 |
| 16 | 0.007 | **0.000** | **0.000** | 0.7640 | 0.8990 |

Below the wall **neither** peak carries a query. The scatter of every decoder query near a pair shows
why: at 16 px the high-scoring queries form two clean columns at ±8 px; at 8 px they collapse into a
single pile at **offset ≈ 0 — the midpoint** — with nothing at ±4. The encoder proposes *one object,
centred between the two peaks*, and the decoder faithfully reports it.

**So the DN lever is REFUTED before spending anything.** You cannot un-suppress a detection that was
never proposed. Shrinking `dn_box_noise_scale` would have been a 3-day null. (Step 1's 12 h screening
fine-tune is moot too.)

**Nuance — clusters1 behaves differently, and it is informative.** At 4–8 px it splits ~(A) 0.33–0.48
/ (C) 0.29–0.52: training on tight pairs *did* partly teach the encoder to propose the second peak,
and those proposals localise. But their scores land at **0.07–0.11 against a 0.10 threshold** — right
on the boundary. That is exactly why phase U looked like a win at the fixed threshold and vanished at
a matched operating point: the recovered peaks are only recoverable by lowering the threshold, which
costs precision elsewhere. Phase U's verdict stands, but we now know *why* it behaved that way.

**Where this points.** The merge happens **at or before the encoder's objectness/proposal stage**, not
in the decoder's learned duplicate suppression. Note this does not contradict phase V: stride-4
features did not help, so it is not feature *resolution* — the encoder is handed adequate resolution
(at stride 8 two peaks 8 px apart already occupy adjacent tokens; at stride 4, two tokens apart) and
still produces a single objectness maximum at the midpoint.

**Caveat.** The acceptance radius scales with separation (`R = max(1.5, 0.30·sep)`), so small rungs are
judged with a tight radius and could in principle over-report "absent". The scatter defeats that
concern — at 8 px the query pile sits ~4 px from each peak, outside any plausible R — but a fixed-R
rerun would settle it if this line is pursued. Rungs below 6 px are flagged `[zones overlap]` and were
not read: there the three zones cannot be made disjoint.

**Status: DN declined without running it (11th lever).** Two levers have now been killed by cheap
measurement in two days (window, DN) where each would have cost ~3 days to run.

## X. Objectness + proposal-box probes — the head describes the close pair as ONE elongated object (2026-08-23/24)

Two free probes on existing checkpoints. Together they move the failure from "somewhere in stage 4"
to a single MLP, and they **correct phase W**.

### X.0 — what phase W actually measured (correction)

`out['interm_outputs']` is built from `hs_enc`, i.e. `tgt_undetach`, the **already-gathered top-900**
(`deformable_transformer.py:420`). The full objectness map over all tokens,
`enc_outputs_class_unselected` (line 409), is never returned. So phase W's "(C) the encoder never
proposes it" is really "**the second peak is not among the 900 selected**" — a weaker claim. The DN
refutation is unaffected (nothing is scored down), but the mechanism was not established.

### X.1 — neither the token grid nor the top-k selection is the limit (`diagnostics/objectness_probe.py`, job 2778786)

Hooks `enc_out_class_embed` to capture the unselected map; asks on the TOKEN GRID whether the pair
can be represented at all, and whether both tokens are selected.

| ssl1, sep | pair in distinct stride-8 rows | **both tokens in top-900** |
|---|---|---|
| 8 px | 1.00 | **0.75** |
| 12 px | 1.00 | 0.98 |
| 16 px | 1.00 | 1.00 |

At 8 px the peaks always land in different tokens and both are usually selected (clusters: 1.00).
**So `num_queries` is a dead lever** — the second token is not out-ranked.

Level mix of the selected 900 — where proposals actually come from:

| model | levels |
|---|---|
| ssl1 | **stride8 790 (87.8%)**, stride16 106, stride32 2, stride64 2 |
| clusters | stride8 485 (53.9%), stride16 201, stride32 195, stride64 19 |
| 5scale | **stride4 619 (68.8%)**, stride8 51, stride16 165, stride32 20, stride64 45 |

**A "stride-8 grid quantisation" story was floated and REFUTED by this same table.** It predicted the
wall sits where the grid can first separate the pair (16 px = 2 stride-8 tokens). But the 5-scale
model draws 69% of proposals from **stride 4**, where an 8 px pair is 2 tokens apart with a gap
between them — and it still scores 0.000 resolved at 8 px. Recorded because it was a good-looking
hypothesis that predicted the wall's *location*, and it is wrong.

### X.2 — DIRECT confirmation: the proposal box head collapses the pair (`diagnostics/proposal_box_probe.py`, job 2778797)

Monkeypatches `gen_encoder_output_proposals` for the per-token anchors and hooks `enc_out_bbox_embed`
for the delta, then reads the actual proposal box of the two specific tokens.
`box = sigmoid(MLP(memory) + anchor)`.

ssl1, χ-separation of the two predicted boxes:

| sep | d_true | d_anchor | **d_box** | err→peak1 | err→peak2 | both on peak |
|---|---|---|---|---|---|---|
| 4 | 4 | 8.0 | **0.47** | 2.41 | 1.96 | 0.662 |
| 6 | 6 | 8.0 | **0.68** | 3.26 | 2.79 | 0.227 |
| 8 | 8 | 8.0 | **0.77** | 4.08 | 3.67 | **0.000** |
| 12 | 12 | 16.0 | 5.88 | 3.36 | 2.68 | 0.637 |
| 16 | 16 | 16.0 | **15.48** | 0.37 | 0.33 | 0.993 |
| 32 | 32 | 32.0 | 31.98 | 0.16 | 0.17 | 1.000 |

**The head actively pulls the boxes together.** At 8 px the anchors are 8.0 px apart and the boxes
come out **0.77 px** apart — each box is moved ~3.6 px *inward, against its own anchor*, landing both
on the midpoint (per-peak errors 4.08 / 3.67 = half the separation). This excludes passive
inheritance of the grid, which would have given `d_box ≈ d_anchor`. The switch is sharp and sits
exactly at the wall: 0.77 at 8 px → 5.88 at 12 → 15.48 at 16 → exact to 0.2 px thereafter.

5-scale does the same at stride 4 (anchors 4.0/8.0 px, `d_box` 1.36/1.76) — an independent second
refutation of the grid story. clusters likewise (0.39 at 4 px, 1.84 at 8 px), and being *slightly*
less collapsed at 8 px is consistent with phase U/W: it sometimes produced a marginal second proposal
scoring 0.07–0.11 against a 0.10 threshold.

Figure: `diagnostics/proposal_box.png`.

### X.3 — the merged box SPANS both peaks; phase U is exonerated; `min_nms` was a dead knob (2026-08-24)

**Box-size test** (`diagnostics/proposal_box_probe.py`, job 2778844). X.2 measured only the box
*centre*. The height discriminates two very different failures, pre-registered before the run:
`h ≈ 8.5 px` (one peak) → a single-object hypothesis at the midpoint, i.e. unstable Hungarian
assignment averaging the target; `h ≈ the pair span` → the head deliberately describes one merged
elongated object.

ssl1, χ-height of the predicted proposal box:

| sep | predicted h | one peak | pair spans |
|---|---|---|---|
| 2 | 10.90 | 8.5 | 10.5 |
| 4 | 11.47 | 8.5 | 12.5 |
| 6 | 13.85 | 8.5 | 14.5 |
| **8** | **15.45** | 8.5 | **16.5** |
| 12 | 14.58 | 8.5 | 20.5 |
| **16** | **10.05** | **8.5** | 24.5 |
| 32 | 9.33 | 8.5 | 40.5 |

**The box tracks the pair span below the wall and snaps back to single-peak height above it.** All
three models. So the head is not confused and not averaging: it perceives one elongated object and
describes it accurately. **The unstable-matching hypothesis is REFUTED** — that predicted a
single-peak-sized box at the midpoint.

**Simulator χ-gap distribution measured** (CPU, minutes). Base simulator vs phase U vs real organic,
gap to the nearest same-q sibling (`QTOL = 8 px`, the clusters_gate definition):

| | median gap | <5 px | <10 px | <33 px |
|---|---|---|---|---|
| base simulator | 85–93 px | 0.029 | 0.045 | 0.135 |
| **phase U clusters ON** | 40.9 px | **0.177** | 0.250 | 0.442 |
| real organic | **3.9 px** | **0.125** | 0.180 | 0.349 |

**PHASE U IS EXONERATED as a fair test.** It did reach — in fact slightly overshot — the real
close-pair rate. So "the model was never shown close pairs" is NOT why it merges them: clusters1 saw
them at 1.4× the real rate and still collapses (`d_box` 1.84 px at a true 8 px vs ssl1's 0.77 — better
by ~2.4×, nowhere near correct). **Training exposure is necessary but demonstrably not sufficient.**

**A wrong claim of mine, retracted.** I asserted that `filter_nms(min_nms=0.001)` on 2.5σ×3.5σ boxes
is what strips close pairs from the training data, and recommended relaxing it. Measured: relaxing the
threshold **changes nothing**.

| effective IoU threshold | segs/frame | median gap | <5 px |
|---|---|---|---|
| 0.001 (current) | 62 | 85.4 | 0.029 |
| 0.3 | 58 | 85.6 | 0.029 |
| 0.7 | 62 | 84.7 | 0.029 |
| both filters lenient (0.7 / 0.9) | 66 | 86.7 | 0.031 |

`filter_nms` *does* drop 49.5% of peaks, so it is doing heavy work — but not on close pairs. The
simulator places peaks **independently at random** over 512×1024; with ~62 segments, two landing at the
same q within a few px of χ is simply rare. They are never generated, so a lenient filter has nothing
to spare. Real close pairs are the same lattice spacing at different crystallite orientations —
physically correlated at identical q, which the base generator does not model. **`add_peak_clusters`
(phase U) is the structurally correct mechanism, and the only one that produces them at all.**
A "lenient NMS" training run would have been a null.

**BUG FIXED — `min_nms` was a dead config knob.** `simulation.py:358` called

```python
filter_nms(pos, widths, a_pos, a_widths, sc.min_nms)          # five positional args
```

against `def filter_nms(pos, widths, a_pos, a_widths, is_ring, min_nms=0.001)`. `sc.min_nms` landed in
the **`is_ring` slot, which the function body never uses**, and the threshold silently fell back to its
default. Verified empirically: config values 0.001 and 0.9 give byte-identical distributions. Now
passed by keyword, and `is_ring` documented as unused. **Behaviourally a no-op today** (config ==
default == 0.001; pre/post-fix simulator signature identical over 25 seeded frames, 809 boxes,
md5 `c048383e…`), but the knob is live, so a future config change will actually take effect instead of
silently doing nothing.

### Y — linear read-out probe: the offset IS in the feature, the head just doesn't use it (2026-08-24)

`diagnostics/linear_readout_probe.py`, job 2778855, 3 min, existing checkpoints.

X.3 left two possibilities needing opposite responses: (a) the 256-number feature the box head reads
does not contain the offset, so the encoder merged the pair before the head saw it — unfixable from the
head; or (b) the offset is present and the head does not use it — a training problem.

**Method.** The box head is `MLP(output_memory[i])` for a SINGLE token, so the probe uses that exact
vector. Target = the signed χ-offset from the token centre to ITS OWN nearest planted peak, i.e.
precisely what the head is supposed to output. Model = ridge regression, closed form (no sklearn on
this env) — a straight-line fit, no capacity to memorise. Every frame carries exactly two peaks at
every rung, so total flux is constant and the fit cannot cheat on brightness or object count.

**Median |error| in px, ssl1:**

| sep | **trained head** | **ridge (enc out)** | ridge (pre-encoder) | shuffled ctrl | target spread |
|---|---|---|---|---|---|
| 4 | 2.05 | 1.21 | 1.25 | 2.09 | 1.92 |
| 6 | 3.00 | **0.63** | 0.97 | 1.72 | 1.86 |
| **8** | **3.83** | **0.49** | 0.67 | 1.93 | 1.93 |
| 12 | 3.03 | **0.37** | 0.49 | 1.67 | 1.97 |
| 16 | 0.33 | 0.30 | 0.48 | 1.95 | 1.99 |
| 24 | 0.20 | 0.29 | 0.53 | 2.28 | 2.32 |

**Answer: (b).** At 8 px the trained head is off by 3.83 px (it points at the midpoint); a straight line
on the identical vector is off by **0.49 px** — ~8× better. Same result for clusters (3.16 → 0.54) and
5scale (4.47 → 0.41).

**Controls hold.** Shuffled labels give 1.93 against a target spread of 1.93 — exactly chance, so the
fit is not an artefact. Train/validation/test split is by FRAME (60/20/20); splitting by token would
leak, since tokens from one image are correlated. λ chosen on the validation slice only.

**No stage destroys the information.** It is already in the projected backbone features (0.67 px at
8 px) and the encoder *improves* it (0.49 px). So the representation hypothesis from X.2/X.3 is
**REFUTED** — the encoder does not merge the pair. Everything the head needs is in the vector it
receives, and it discards it.

**Caveat on the 4 px row.** There, both peaks often fall in the SAME token (distinct rows only 0.54 of
the time, phase X.1), and identical features with different targets are unlearnable by anything. The
weak 1.21 is a property of the grid, not the probe, and does not affect the 6–12 px rows where the
failure lives.

**Why this matters.** For eleven levers the answer was always "the model cannot". Here it demonstrably
**can** and does not. The ceiling is not perceptual, so a training-side intervention has something real
to aim at — the first time in this sequence that has been true.

**Deliberately NOT claimed:** why training fails to teach the head to use the information. Four
mechanisms have been proposed and refuted in this investigation (proposal selection, stride/grid
quantisation, unstable Hungarian matching, representational merging); a fifth guess is worth little.
Note also that phase U supplied close pairs at 1.4× the real rate and moved the head only ~2.4×, so the
training signal is not simply short of examples.

### Where this leaves the problem

The failure is one component: **`enc_out_bbox_embed`, the encoder's proposal box head.** Upstream is
fine — the image resolves the pair (3.3σ at 8 px), the features resolve it, the grid separates it,
the objectness selects both tokens. Downstream is fine — regression is sub-pixel once the boxes are
distinct. One MLP takes two well-separated anchors and merges them.

**Settled by phase Y:** the encoder does NOT merge the pair. A straight-line fit on the exact vector the
head reads recovers each token's offset to its own peak to 0.49 px where the trained head is off by
3.83 px. The information is present and unused, so the remaining fault is in the TRAINING SIGNAL, not
in perception.

**Narrowed by phase Z:** and not in the head's own training signal either. Trained ALONE on a frozen
trunk, this exact head under this exact loss reaches **0.31 px** — better than ridge, 12× better than
itself in the real run — and it still reaches 0.48 px when close pairs are thinned to the real
organic rate. Loss form, anchor+sigmoid encoding, Hungarian assignment, MLP capacity and close-pair
rarity are each now measured and cleared. **Still open:** the three things phase Z held fixed — the
frozen (and converged) trunk, the absence of every competing loss term, and the ladder's simplified
frames. The failure is created by the CONDITIONS of joint training or by the real training
distribution, not by anything intrinsic to the head or its objective. Five mechanisms have now been
proposed and refuted in this investigation; a sixth is not offered here.

**Method note.** Two claims in this sequence were over-stated before being caught: phase W's "never
proposed", and the stride-8 grid story. Both came from reading a mechanism into a measurement that
did not isolate it. X.2 is a direct read of the model's own output, not an inference chained across
two probes.

### Z — head-only training: which part of the training signal loses the offset? (PRE-REGISTERED)

`diagnostics/head_only_probe.py`, `tmp_diag/run_head_only.sbatch`. ssl1 only, ~25 min, no checkpoint
written.

Phase Y left exactly one question: the head is fed everything it needs and does not use it — why?
Only four things touch `enc_out_bbox_embed`: the **features** (settled, they are fine), the **target
parameterization**, the **loss**, and the **assignment** that decides which query is supervised by
which peak. Rather than propose a fifth mechanism after four have been refuted, this probe measures
which of those three survivors is responsible, by running arms that differ in exactly one of them on
ONE cached feature set.

The trunk is frozen and its features cached once, so only the box MLP moves. This is not a training
run and produces no candidate model.

| arm | what it is | differs from the previous arm in |
|---|---|---|
| **A** | `sigmoid(MLP(mem) + anchor)`, Hungarian-matched, `5·L1 + 2·GIoU` — byte-for-byte the `interm_outputs` loss that trains this head (dino.py:617-633) | — |
| **B** | identical loss and parameterization, but each ladder token is supervised by **its own** peak | the ASSIGNMENT |
| **C** | same MLP and features, predicts the signed χ-offset in **pixels** under plain L1 — no anchor, no sigmoid, no GIoU | the PARAMETERIZATION |
| **D** | phase Y's ridge, recomputed here | CAPACITY (linear vs MLP) |
| **D'** | ridge fitted to the head's OWN target — the logit-space delta over the anchor — read back out in px | isolates the anchor+sigmoid encoding from any optimizer |

Every arm reports the same phase-Y metric on the same scale — median |predicted box χ-centre − its own
peak| in px, on held-out FRAMES, per rung — so the numbers sit directly against the two that matter:
**trained head 3.83 px, ridge 0.49 px, both at sep 8.**

**Pre-registered readings.**
- **A ≈ 0.5** → the objective is capable in isolation; the full run fails for a reason outside the
  loss's design. Lever: reweight / curriculum / data mix, NOT a new loss form.
- **A ≈ 3.8, B ≈ 0.5** → loss and parameterization are fine; the ASSIGNMENT is the fault.
- **A ≈ B ≈ 3.8, C ≈ 0.5** → the anchor+sigmoid+GIoU target itself is what cannot be learned.
- **C ≈ 3.8** → the MLP cannot do what ridge does. Unlikely — an MLP strictly contains a linear map —
  and it would mean an optimization pathology, not a signal one.

**The confound in arm A, and the sweep that removes it.** Arm A trains on the ladder, where rungs
4/6/8 are half the frames — against **12.5%** of adjacent-peak gaps under 5 px in the real organic
labels (and 17.7% in phase U's clusters diet). So a bare "A ≈ 0.5" would conflate *the objective is
capable* with *the diet was generous*, and must not be read as the former alone. Arm A is therefore
repeated at four close-pair diets — close-rung frame fractions **0.50 / 0.25 / 0.12 / 0.06**, which
bracket both the real rate and phase U's — with the training-set SIZE held fixed so only the mix
varies, and the lr inherited from arm A so the sweep stays single-variable.

- **learns at 0.50, stops by 0.12** → rarity is sufficient to explain the full run, and the lever is
  the data mix. Note this would sit awkwardly beside phase U, which raised the rate to 1.4× real in a
  FULL run and moved the head only ~2.4×; the difference would then be the frozen trunk.
- **still learns at 0.06** → rarity is NOT the explanation, and the fault is in training the head
  jointly with everything else.

**What arm C does and does not isolate.** C bundles three changes off B at once — no anchor, no
sigmoid, no GIoU, one coordinate instead of four — so it is a capability check (can the MLP do what
ridge does), not a clean single variable. The anchor+sigmoid encoding is isolated cleanly and without
any optimizer by **D vs D'**, which are the same ridge differing only in target space. C's target is
expressed in normalized units so its gradients share a scale with A and B and the one lr grid is fair
to all three.

**Controls.** Frame-level 60/20/20 split with phase Y's seed, so the rungs are directly comparable.
Learning rate AND epoch are chosen on the VALIDATION slice only, never on test; the lr sweep spans
three decades so a null result cannot be an lr artefact. The head is initialised exactly as the real
one is (`MLP(256, 256, 4, 3)`, zero last layer, dino.py:133-139), so every arm starts from a fresh
detector's state. `LADDER_ISO=1` keeps the stimulus identical to phase Y's.

**Two extra columns, on the record as a check rather than a claim.** (i) The median **height** error
per arm. X.3 showed the merged box SPANS the pair, which is mostly an error in the χ *extent*, and
that coordinate sits far out in the sigmoid tail — anchor `h = 0.05·2^lvl` (utils.py:45) against a
true 8.5/512 = 0.0166, where `dp/dlogit` is about 15× smaller than for the centre near mid-image.
That is arithmetic off `gen_encoder_output_proposals`, not a mechanism; these columns are what would
promote it to one, or kill it. (ii) For arm A, how often a peak's Hungarian match actually lands on
that peak's **own** token, and how far away it lands when it does not — so an A-vs-B gap says *what*
the assignment does wrong rather than only *that* it does.

**RESULT (job 2779790, 16 min, ssl1).** Median |χ-centre error| in px at the ladder tokens, on
held-out frames. Reference: the real trained head is **3.83 px** at sep 8, phase Y's ridge **0.49**.

| arm | sep 4 | sep 6 | **sep 8** | sep 12 | sep 16 | sep 24 |
|---|---|---|---|---|---|---|
| **A** detection loss | 0.54 | 0.35 | **0.31** | 0.25 | 0.19 | 0.23 |
| **B** oracle assignment | 0.74 | 0.41 | **0.31** | 0.28 | 0.21 | 0.26 |
| **C** direct offset | 0.70 | 0.33 | **0.23** | 0.24 | 0.19 | 0.18 |
| **D** ridge px | 1.30 | 0.68 | **0.49** | 0.38 | 0.30 | 0.28 |
| **D'** ridge logit | 1.34 | 0.68 | **0.50** | 0.43 | 0.33 | 0.30 |

**Every pre-registered branch that blamed the head's own training signal is refuted.** Arm A — the
real `interm_outputs` loss, byte-for-byte, with nothing else competing — reaches **0.31 px** where the
same head in the real run is off by 3.83. That is 12× better than the trained head and better than
ridge itself. Reading each contrast:

- **A ≈ B (0.31 / 0.31)** → the **assignment costs nothing**. Handing every token its own peak for
  free changes the result not at all.
- **B ≈ C (0.31 / 0.23)** → the **parameterization costs nothing**. Stripping the anchor, the sigmoid
  and the GIoU term does not help; if anything the box form is slightly the better target.
- **C ≥ D (0.23 / 0.49)** → **capacity is not the issue** in either direction: the MLP does not fail
  to reach the linear fit, it beats it.
- **D ≈ D' (0.49 / 0.50)** → **the anchor+sigmoid encoding costs nothing.** This kills the sigmoid-tail
  arithmetic pre-registered above as a check: `dp/dlogit` really is ~15× smaller for the height
  coordinate, and it makes **no measurable difference** to what a fit can recover. Arm A's height
  error confirms it from the other side — **0.05-0.07 px at every rung**, essentially exact. The
  merged box that SPANS the pair (X.3) is not reproduced here at all.

D landing on 0.49 at sep 8 reproduces phase Y's 0.49 exactly, through an independently written code
path — the two probes agree.

**The diet sweep refutes rarity too.** sep-8 test error against the close-frame fraction:

| f | 0.50 | 0.25 | **0.12** (≈ real) | 0.06 |
|---|---|---|---|---|
| sep-8 px | 0.34 | 0.40 | **0.48** | 0.57 |

Thinning the close-pair diet 8× degrades the head by a factor of 1.7, from 0.34 to 0.57 px. At the
real organic rate it is **0.48 px** — still eight times better than the trained head. Scarcity of
close pairs is **not** why the real head merges them. This agrees with phase U from the opposite
direction: U added close pairs to a full run and gained little; Z removes them from an isolated head
and loses little. "More close pairs" is now dead from both ends.

**Assignment, measured directly.** A peak's Hungarian match lands on that peak's own token 0.68-0.91
of the time (0.77 at sep 8), and when it misses, the matched query's token sits ~1.65 px away. So the
assignment IS imperfect — and arm B shows that imperfection is harmless. A mechanism can be real and
still not be the cause; this is the fifth candidate, and it died before being proposed.

**What this leaves — stated as what was held fixed, not as a guess.** The probe differs from the real
run in exactly three ways, and the fault must live in one of them:

1. **The trunk was frozen, and frozen at convergence.** Arm A reads ssl1's finished features. In the
   real run the head learns against a trunk that starts from SSL init and moves under it.
2. **Only the box terms were optimized.** The class head was frozen, so `loss_ce`, the DN losses, the
   six decoder aux losses and the Co-DINO aux head were all absent. Arm A had no competition.
3. **The frames are ladder frames.** Two-peak pairs, segments only, constant intensity 30, ISO box
   shape — not the heterogeneity of real simulated GIWAXS. The diet sweep varied the close/far MIX
   within that, not the realism of the frames themselves.

(1) and (2) are the joint-training conditions; (3) is a distribution difference. Nothing else remains:
features, loss form, encoding, assignment, capacity and close-pair rarity are now each measured and
cleared. **No mechanism is proposed here** — five have now been refuted in this investigation, and the
next step should separate (1)/(2) from (3) by measurement, not by argument.

**Caveats.** 96.1% of ladder tokens fall inside the top-900 and the remaining 3.9% are excluded, so
the arms describe selected tokens only (which is also what the real head regresses). The sep-4 rung is
weak in every arm, as in phase Y, because both peaks share one token 0.54 of the time. lr and epoch
were chosen on validation only; all three lrs gave the same verdict for A, B and C, so no result here
turns on the sweep.

### Z.1 — lock-in check: the head was already stuck at epoch 279 (2026-08-24)

`diagnostics/lockin_probe.py`, `tmp_diag/run_lockin.sbatch`, job 2779900, ~6 min.

Phase Z left three differences from the real run, two of which are the conditions of joint training.
The joint-training story has a testable signature: if the head learns "one wide box covers both" early,
while the features genuinely cannot separate a close pair, then the information AVAILABLE to it should
keep improving over training while the information USED stays flat. This measures both, at the only two
time points that exist on disk — `checkpoint0279.pth` (epoch 279, just before the lr drop at 280,
organic AP 0.542) and `checkpoint.pth` (epoch 436, the end of the run, organic AP 0.561). One fixed
image set for both.

AVAILABLE = ridge on the exact 256-dim vector the box head reads. USED = the trained head's own error
on the same tokens. Median |error| in px:

| sep | USED @279 | AVAILABLE @279 | USED @436 | AVAILABLE @436 | Δ USED | Δ AVAILABLE |
|---|---|---|---|---|---|---|
| 4 | 2.26 | 1.19 | 2.05 | 1.21 | −0.21 | +0.02 |
| 6 | 3.18 | 0.67 | 3.00 | 0.63 | −0.18 | −0.04 |
| **8** | **3.94** | **0.47** | **3.83** | **0.49** | **−0.11** | **+0.02** |
| 12 | 3.05 | 0.39 | 3.03 | 0.37 | −0.02 | −0.02 |
| 16 | 0.38 | 0.31 | 0.33 | 0.30 | −0.05 | −0.01 |
| 24 | 0.41 | 0.31 | 0.20 | 0.29 | −0.21 | −0.02 |

**This is the pre-registered "both flat" outcome, and it is genuinely ambiguous about timing.** The
information was already fully available at epoch 279 (ridge 0.47, indistinguishable from the final
0.49) and the head was already stuck at 3.94. Whatever locked it in happened BEFORE epoch 279, and
these two checkpoints cannot date it — nothing earlier was ever written (`save_checkpoint_interval =
1000`, run ended at 436). **Do not write this up as dating the lock-in.**

Three things it does establish:

- **"Train longer" is dead as a lever.** The last 157 epochs — a third of the run, and including the
  lr drop at 280, which is normally the single largest step-change in a detection run — moved the
  sep-8 error from 3.94 to 3.83. That is 3%. The head is not slowly converging toward the right
  answer; it has stopped.
- **The head is not weak, it is specifically close-pair-blind.** At sep 24 it reaches 0.20 px and
  BEATS ridge's 0.29; at sep 16 it matches it. So the same head that out-performs a straight-line fit
  on separated peaks is 8× worse than that fit at 6-8 px, with the information equally available in
  both regimes. This is not a regression-quality problem anywhere on the ladder.
- **The features reached final quality by epoch 279 at the latest.** Ridge is flat to ±0.04 px across
  the window at every rung.

**A contrast worth recording:** organic AP rose 0.542 → 0.561 over exactly this window, so the model
as a whole was still improving. Whatever those 157 epochs bought, it was not close-pair resolution.

**Where this leaves the design.** Z.1 has done what two checkpoints can do. The remaining separation —
condition 1 (joint-training trajectory) vs condition 3 (the real training distribution) — needs
experiment 1: phase Z's head-only training with the frames swapped from the ladder to real simulated
ones, **with clusters ON**. That last detail is load-bearing: the base simulator gives `<5 px` gaps at
0.029, below the diet sweep's tested floor of 0.06, so running it with clusters OFF would silently
re-run the rarity test instead of the distribution test.

### Z.2 (experiment 1) — the REAL TRAINING DISTRIBUTION is what teaches the merge (2026-08-24)

`diagnostics/real_frame_head_probe.py`, `tmp_diag/run_realframe.sbatch`, job 2779999, 21 min.

Fills the missing cell of the 2×2. Everything is held at phase Z's settings — frozen converged trunk,
only `enc_out_bbox_embed` moves, the real `interm_outputs` loss, 180 training frames, the same ladder
test slice and the same 60/20/20 seed — and exactly ONE thing changes: **where the training frames come
from.**

Median |χ-centre error| in px on the ladder test slice:

**ISO=1 eval (2.43×8.5 boxes — phase Z's exact eval set):**

| trained on | sep 4 | sep 6 | **sep 8** | sep 12 | sep 16 | sep 24 |
|---|---|---|---|---|---|---|
| real frames, clusters ON | 1.66 | 2.51 | **3.22** | 2.34 | 0.77 | 0.69 |
| real frames, clusters OFF | 1.69 | 2.51 | **3.23** | 2.98 | 1.04 | 0.64 |
| ladder frames (Z baseline) | 0.54 | 0.35 | **0.31** | 0.25 | 0.19 | 0.23 |

**The real trained head is 3.83 px. Training this head on real frames gives 3.22 px. Training the
identical head on ladder frames gives 0.31 px.** The merge is reproduced, from the distribution alone,
with the trunk frozen at convergence and every competing loss term absent. **Condition 3 is the
answer; conditions 1 and 2 — the joint-training trajectory — are not needed to produce the failure.**

**Clusters ON vs OFF is a dead heat: 3.22 vs 3.23.** The two training sets differ by more than 10× in
close-pair rate (measured here: `<5 px` **0.173** vs **0.015**, against real organic 0.125) and the
result does not move. That is the third independent refutation of the rarity story, after phase U and
phase Z's diet sweep. Close-pair scarcity is not a lever and should not be revisited.

**The wide rungs rule out a domain-gap artefact.** A real-trained head tested on ladder stimuli faces a
domain gap regardless — the ladder holds intensity at 30, carries 16 objects and no rings. That gap is
directly visible and directly bounded: at sep 16/24 the real-trained head reaches 0.77/0.69 px against
the ladder-trained head's 0.19/0.23, so the gap is worth about **0.5 px**. The close-pair error is
**3.22 px**. A uniform domain gap would degrade every rung equally; instead the head is nearly fine
when peaks are apart and fails only when they are close — precisely the real model's signature.

**Box-shape control.** A second eval set at ISO=0 (10.6×8.5, real box shape, identical peak positions —
`make_images` seeds per frame) gives real-ON **3.67** and real-OFF **3.75** at sep 8. The real arms are
*in* distribution there and still merge, so the failure is not a box-shape artefact. Note the ladder
baseline reads 2.55 on that eval set: it trained on 2.43-wide boxes and is out of distribution, so the
ISO=0 block is interpretable for the real arms only and NOT a fair line for the baseline.

**Sweep integrity.** For the clusters-ON arm all three learning rates selected the same checkpoint
(1.61 px val at lr 1e-5), i.e. the higher rates never beat the lowest — the failure is not an lr
artefact. The real arms also received MORE supervision than the baseline, not less: 59.3 and 43.6
objects per frame against the ladder's 16, at equal frame count. The result is a failure despite the
advantage.

**What this changes.** For the first time the fault has a location that can be dialled: something about
the real training distribution teaches this head to describe a close pair as one object. Candidates
not yet separated — object density (~60/frame vs 16), class heterogeneity (rings alongside segments),
intensity variation (10-50 vs a fixed 30), box-size variation. Each is now a cheap single-variable
test with this machinery (~7 min per arm), building UP from the ladder case that works rather than
ablating down from the case that fails. **No mechanism is claimed here** — six have been refuted in
this investigation; this identifies where to look, not what is wrong.

### Z.3 — NONE of the four candidate properties explains it; the list was wrong (2026-08-24)

`diagnostics/distribution_ablation_probe.py`, `tmp_diag/run_ablation.sbatch`, job 2780029, 56 min.

Z.2 localised the merge to the real training distribution. This neutralises one property of that
distribution at a time and retrains, everything else held at Z.2's settings and every arm scored on
the same unmodified ladder test slice. Anchors: **real 3.22**, **ladder 0.31**, deployed head 3.83.

| training frames | obj/frame | `<5px` | sep 4 | sep 6 | **sep 8** | sep 12 | sep 16 | sep 24 |
|---|---|---|---|---|---|---|---|---|
| real | 59.3 | 0.173 | 1.66 | 2.51 | **3.22** | 2.34 | 0.77 | 0.69 |
| fixed intensity | 59.3 | 0.173 | 1.74 | 2.50 | **3.20** | 2.19 | 0.82 | 0.63 |
| fixed box size | 59.3 | 0.173 | 4.73 | 6.04 | **6.12** | 3.52 | 3.75 | 3.32 |
| no rings | 53.4 | 0.177 | 1.80 | 2.58 | **3.35** | 1.96 | 0.49 | 0.48 |
| count-matched control | 45.7 | 0.180 | 1.99 | 2.61 | **3.21** | 2.23 | 0.78 | 0.68 |
| density → ~12 | 11.7 | 0.334 | 1.76 | 2.69 | **3.35** | 2.76 | 0.70 | 0.58 |
| **all four at once** | 11.5 | 0.302 | 1.57 | 1.99 | **2.52** | 2.46 | 1.77 | 1.42 |
| ladder | 16.0 | — | 0.54 | 0.35 | **0.31** | 0.25 | 0.19 | 0.23 |

**Verdict: the property list was wrong.** Brightness variation (3.20), rings (3.35), object count
(3.21 count-matched), and crowding (3.35) each move the sep-8 error by less than 0.15 px against a
2.9 px effect. All four at once reaches 2.52 — and that arm's wide rungs degrade in step (1.77/1.42
against the baseline's 0.77/0.69), which is the uniform signature of a domain gap, not a close-pair
fix. Nothing here recovers the ladder's 0.31.

**The nulls are real nulls, not a dead pipeline.** `fixed box size` is an unintended positive control:
it moved the result hard, to 6.12. So the ablation machinery demonstrably bites, and the four flat
arms are flat because the properties do not matter. Note its damage is uniform across rungs
(3.32-6.12), so it is a domain-gap effect from training on one box size and testing on another —
worse generalisation, not more merging.

**A reading error to avoid in this run's raw output:** the ladder row printed `<5px` = 1.000, which
was the dict default rather than a measurement (the ladder set was never passed through the gap
statistic). Fixed in the script; the cell is dashed above rather than quoted.

### What Z.3 actually changes — a reframing that must now be tested, not assumed

Z.2's conclusion was "the real distribution teaches the merge". Z.3 finds no property of it that does.
That makes a second reading live, and it points the opposite way:

**Every ladder frame consists 100% of paired objects at controlled separations — at every diet
setting phase Z ever ran.** The diet sweep varied WHICH separations appeared (0.50 → 0.06 of frames
from the close rungs), never whether an object had a partner at all; the "far" rungs are still pairs,
just wider ones. So the axis that has never been varied is not the rate of close pairs but whether
the training task is ABOUT pairs.

If that is what matters, then it is not that real data contains something poisonous — it is that
close-pair resolution is a skill the head only acquires when it dominates the training signal.
**That would make the lever a loss-side one (weighting, or a dedicated auxiliary objective), not a
data-side one**, and it would mean Z.2's framing needs softening: the ladder teaches un-merging
rather than the real frames teaching merging.

This is stated as the next MEASUREMENT, not as a conclusion. Six mechanisms have been refuted in this
investigation and a seventh has just been refuted four ways over in this section; the record here is
that reasoning ahead of measurement has been wrong every time it has been tried. The test is the
build-UP direction that Z.3 did not run: start from ladder frames and add real objects AROUND the
planted pairs, so the pairs stay but stop being the whole task, and see whether the error climbs to
3.2 as the paired fraction falls.

### Z.4 / Z.5 — dilution REFUTED; re-weighting real but insufficient (2026-08-24)

`diagnostics/pair_focus_probe.py`, `tmp_diag/run_pairfocus.sbatch`, job 2780653, 60 min.
Anchors throughout: **real 3.22**, **ladder 0.31**, deployed head 3.83, all on the same test slice.

**Z.4 — pair dilution. REFUTED.** Ladder frames with real unpaired context added around the planted
pairs; the 16 planted objects are identical across arms, so close-pair supervision is constant in
absolute terms and only the context grows.

| obj/frame | paired-within-24px | sep 4 | sep 6 | **sep 8** | sep 12 | sep 16 | sep 24 |
|---|---|---|---|---|---|---|---|
| 14.8 | 0.767 | 0.24 | 0.18 | **0.19** | 0.18 | 0.56 | 1.19 |
| 26.6 | 0.454 | 0.51 | 0.37 | **0.26** | 0.32 | 0.65 | 1.02 |
| 43.7 | 0.361 | 0.60 | 0.42 | **0.40** | 0.36 | 0.62 | 0.93 |
| 63.2 | 0.363 | 0.58 | 0.34 | **0.41** | 0.32 | 0.36 | 0.81 |

Burying the pairs in four times as much context costs a factor of **2** (0.19 → 0.41), against the
factor of ~10 the hypothesis needed. **"Close-pair resolution is only learned when it dominates the
training signal" is wrong.** Sixteen clean ladder pairs teach the skill regardless of what surrounds
them.

*Two limits, stated rather than buried.* (i) The sweep reached paired-24px **0.36**, not real frames'
**0.137**: the 120-object target realised only 63 objects/frame once context peaks near a planted pair
were dropped and detector gaps applied. So the last stretch to real composition is an extrapolation —
though the curve is flattening (0.19 / 0.26 / 0.40 / 0.41), not accelerating toward 3.22. (ii) The
14.8-object arm is NOT byte-identical to phase Z's ladder arm: it skips `make_images`' detector-mask
pair filtering, so some planted pairs sit in dead regions. It reads better at the close rungs
(0.19 vs 0.31) and worse at the wide ones (1.19 vs 0.23) as a result.

**Z.5 — close-pair loss weight on real frames. A real effect, and not enough.**

| weight | sep 4 | sep 6 | **sep 8** | sep 12 | sep 16 | sep 24 |
|---|---|---|---|---|---|---|
| 1 (baseline) | 1.66 | 2.51 | **3.22** | 2.34 | 0.77 | 0.69 |
| 3 | 1.59 | 2.35 | **3.27** | 2.13 | 0.71 | 0.66 |
| 10 | 1.68 | 2.26 | **2.79** | 1.50 | 0.60 | 0.49 |
| 30 | 1.61 | 2.08 | **2.61** | 1.13 | 0.64 | 0.66 |

Thirtyfold re-weighting buys **19%** at sep 8 (3.22 → 2.61) and **52%** at sep 12 (2.34 → 1.13),
monotone from w=10 up, with the wide rungs flat (0.77 → 0.64) so it is not a domain shift. The loss is
normalised by the SUM OF WEIGHTS, not the box count, so this is genuine gradient re-allocation and not
a disguised learning-rate change. Real, directional, and nowhere near 0.31 — **not enough to justify a
full training run on its own.**

### What survives, and the specific next test

Z.4 rules out share-of-the-frame, so what is left is the **NATURE of the pairs**. Ladder pairs are
identical twins at 4-24 px, spread across the resolution limit. The real simulator's siblings come
from `_sample_chi_gaps` (simulation.py:499-508), and its parameters are worth reading against the
failure:

    cluster_tight_frac   = 0.32          32% of sibling gaps drawn from...
    cluster_tight_gap_px = (1.0, 6.0)    ...1-6 px, i.e. 0.4-2.5 sigma at sigma_chi = 2.43 px
    cluster_broad_lognorm= (4.007, 0.95) the other 68%, median 55 px

Measured on the generated frames this session: `<5 px` **0.173**, `<10 px` **0.247** — so only **0.074**
of gaps land in 5-10 px. The training distribution is concentrated *below* the resolution limit and
*far above* it, with a hole at exactly the separations where resolving is both possible and needed.
The 8 px case the deployed model fails on sits in that hole.

**Pre-registered test, and NOT a claim** — seven mechanisms have been refuted in this investigation and
every one of them looked this reasonable beforehand. Two arms:
1. Ladder pairs re-drawn from the REAL gap distribution (1-6 px tight / 55 px broad) instead of
   4-24 px. If that reproduces ~3.2, the gap distribution is confirmed as the cause.
2. Real frames with `cluster_tight_gap_px` widened to roughly (4, 12). If the merge weakens, this is a
   **data-side lever on a config knob that already exists** — the first actionable one in the sequence.

Arm 2 is the one worth having: unlike everything else tested, it is a change that could go into a real
training run tomorrow.

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
  **SUPERSEDED by phases S/T (2026-08-18):** the "sensitivity ceiling" reading is wrong. Prominence
  does not predict which peaks are missed (AUC 0.489 organic), and 84.5% of misses sit within 8 q-px
  of a peak the model DID detect. It is a χ-SEPARATION ceiling, not a sensitivity one.
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
