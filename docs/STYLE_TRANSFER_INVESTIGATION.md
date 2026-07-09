# Style-Transfer / Synthetic-Input Appearance-Matching — Feasibility & Investigation

Target: close the **synthetic→real appearance gap** in `mlgidDETECT_DINO` (DINO 4-scale, Swin-L 48×6
elongated-window, single-channel `in_chans=1`, 2-class ring/segment, ONNX-exported GIWAXS/GID
diffraction-peak detector on 512×1024 polar images) by making the ~100%-synthetic training images
*look* like real preprocessed images in their intensity/noise statistics.

Scope: attack the dominant remaining failure — **low SENSITIVITY (recall) to FAINT and high-q peaks**
(diagnostic: faint visibility=1 recall ≈ **0.33**, high-q q682–1024 recall ≈ **0.44**). Four
*detector-side* levers (Semi-DETR pseudo-labels, Cross-DINO Boost Loss, Cross-DINO CCTM, Co-DINO aux
heads) have all been null/negative on the organic real-data set (`MODIFICATIONS.md` I/J/K/L). The
working conclusion is that **the bottleneck is the DATA, not the detector**: the model never sees a
faint peak that renders the way a real faint peak renders, so it never learns to fire on one. This doc
scopes a **training-time transform applied only to synthetic inputs** to fix that.

Every claim below is grounded in a `file:line` in this repo, in a committed diagnostic figure, or in
`MODIFICATIONS.md`/`ROADMAP.md`. Line numbers were read on branch `Semi-DETR` at write time — reverify
before editing, they drift.

> **Scope guard (HARD CONSTRAINT #1 — deployed preprocessing MUST NOT change).** Real images at
> inference go raw → `standard_preprocessing` (`util/exp_preprocess.py:247`) → model, kept in lockstep
> with the deployed `mlgidDETECT` package. The transform proposed here is a **TRAINING-TIME transform
> applied ONLY to SYNTHETIC training inputs** (inside `SimulationDataset`, `main.py:52`), whose *target
> distribution* is the output of the unchanged `standard_preprocessing`. The inference path,
> preprocessing, and the exported ONNX graph are **byte-identical to today**. Proof in §6.

---

## 1. Verdict up top

**Verdict: GREENLIGHT a scoped MVP — build (a) deterministic per-image CDF/histogram matching of
synthetic→real intensity, optionally paired with (b) a small real-derived background-noise injection.
DECLINE the learned neural style transfer / CycleGAN route. Gate hard on the faint/high-q recall probe,
not AP alone.**

Why this is worth a run, when four detector-side levers were not:

- It attacks the axis the diagnostics actually point at. `ROADMAP.md:64-91` (the "KEY FINDING") and the
  Diagnostic-C breakdown (`MODIFICATIONS.md:368-378`) both conclude the two failure modes — faint/high-q
  **misses** and confident on-ring **hallucinations** — are a *synthetic→real content gap*, "the model
  trained on synthetic doesn't truly know what real peaks vs real rings/background/diffuse scattering
  look like" (`ROADMAP.md:84-90`). Every detector-side lever tried since leaves the encoder to infer
  real faint-peak appearance from synthetic examples that don't have it. A data-appearance fix is the
  one lever that changes *what the encoder is shown*, not *how it is supervised*.
- The recommended form is **rank-preserving by construction** (a monotone per-image intensity remap),
  so it **cannot delete or move a labeled peak** — directly satisfying HARD CONSTRAINT #2 (faint-peak
  preservation, §2, §5). This is the property a GAN cannot offer.
- It is **cheap, robust, and learned-model-free**: ~1 new function + a one-time offline reference
  histogram; fine-tune from the ssl1 checkpoint; A/B on the existing `--eval` pipeline + `diag_compare.py`.
- The **insertion is structurally training-only** — `SimulationDataset` is instantiated only in the
  training loop (`main.py:235`, `main.py:412`); the eval path (`evaluate_giwaxs_ap`, `main.py:182`) and
  ONNX export (`export_onnx.py`) never import it (§6). Zero deploy risk.

Why the caution / the hard gate (the adversarial read, §8):

- **This is adjacent to Path A, which was TRIED & REVERTED with no AP gain** (`MODIFICATIONS.md:72-121`).
  Path A matched the *masking geometry* and bumped the *quantization level count*; it did **not** do
  full per-image distribution matching to a real reference, nor add a real-derived noise model — so it is
  *distinct*, but it is close enough that the null-result precedent must be taken seriously (§7).
- **Expected magnitude is uncertain and possibly small.** The known pixel-mean was *already* matched
  (means ~0.51–0.56, `MODIFICATIONS.md:83`); what remains is *distribution shape* (spiky vs smooth,
  §3). Whether closing shape moves recall is the open empirical question — hence "gate hard."
- **The label-incompleteness confound can mask a real win** (`MODIFICATIONS.md:211-215`,
  `ROADMAP.md:64-91`): if organic GT misses faint peaks, a genuine faint-recall gain may not register as
  AP. This is why the gate is the **faint/high-q recall probe**, not AP alone (§7, §8).

**Payoff-vs-effort call.** Expected organic-AP payoff is **modest and unproven**, but the MVP is (a)
the most on-thesis *data* lever left, (b) zero inference/ONNX cost, (c) cheap (one function, warm-start
fine-tune), (d) mathematically safe for faint peaks, and (e) cleanly falsifiable on the existing recall
probe. That asymmetry justifies building the deterministic MVP and gating hard. The GAN is **not** worth
its risk (moving/erasing faint peaks, invalidating labels, a second model to train — §4c).

**Effort:** MVP ≈ **low** (a `style_match(image, mask)` function + an offline reference-CDF builder +
one call site in `SimulationDataset.__getitem__` + a config flag; ~80–150 LOC, no model/graph change).
Physics-noise add-on ≈ low-medium. GAN ≈ high — declined.

---

## 2. The two HARD CONSTRAINTS, stated plainly

**HC #1 — the deployed real-image path is frozen.** Real inference: raw reciprocal image →
`standard_preprocessing` (`util/exp_preprocess.py:247`) → `contrast_correction` → model. This is matched
byte-for-byte to the deployed `mlgidDETECT` package (`MODIFICATIONS.md:28-38`, Phase C parity work).
**We change none of it.** The style transform lives entirely on the *synthetic training* side and its
*target* is the (unchanged) output of `standard_preprocessing`. Directionally: we move **synthetic → the
real-preprocessed distribution**, never the reverse, and never at inference. §6 proves the isolation.

**HC #2 — faint-peak preservation is the central risk.** Any transform that *dims* or *denoises* a
synthetic image risks pushing an already-faint synthetic peak below detectability, or erasing it — which
would **worsen the exact metric we are trying to raise**. The mitigation is a design invariant, not a
hope:

- The **recommended MVP is a per-image monotone (rank-preserving) intensity remap** on valid pixels
  (§4a). A monotone map cannot reorder intensities → a pixel that was brighter than its local background
  stays brighter → **a labeled peak that was a local maximum remains a local maximum. Erasure is
  mathematically impossible** for the matching step. This is the core safety argument, and it is why we
  prefer CDF matching over any brightness/contrast/denoise transform.
- The **only** step that *can* erase a peak is additive noise (§4b). It is therefore (i) optional, (ii)
  amplitude-capped below the faintest labeled-peak local contrast, and (iii) guarded by a hard invariant
  check: for every labeled box, the peak-vs-local-background contrast after the transform must stay above
  a floor, else the noise step is skipped for that image (a fail-safe, mirroring the "regenerate
  degenerate images" guard already in the sim, `MODIFICATIONS.md:99-100`).
- We **never** adopt a transform that can remove a labeled peak (rules out CycleGAN/neural ST, §4c).

---

## 3. What is ALREADY known about the pixel gap (build on it, don't rediscover)

Two independent prior efforts already characterized the synth-vs-real pixel distribution. **Reuse their
conclusions:**

### 3.1 Path H audit (`MODIFICATIONS.md:72-121`) — the mean is matched; masking and quantization are not
- **Contrast/intensity center already well-matched**: means ~0.51–0.56 for both synthetic and real
  (`MODIFICATIONS.md:83`) — the Phase-C preprocessing-parity work holds. So the gap is **not** a
  brightness offset; a global gain/bias would be pointless.
- **Dominant *spatial* gap = masking**: synthetic masked ~3.5% of pixels vs real ~30%
  (`MODIFICATIONS.md:84-85`). Path A fixed this to ~0.30. **No AP gain; reverted** (`:114-121`).
- **Secondary gap = quantization**: `digitalize_img` quantized each synthetic image to **16–64
  levels** (`simulation.py:933-935`, `channels = randint(16, 64)`) vs real ~250 (`MODIFICATIONS.md:90`).
  Path A bumped it to 128–256. **No AP gain; reverted.**

### 3.2 The committed histogram diagnostic (`diagnostics/synth_vs_real_hist.png`) — SPIKY vs SMOOTH
I re-read the figure. It plots synthetic (n=25) vs real-preprocessed (n=23) nonzero-pixel densities on
[0,1]:
- **Real preprocessed (blue): smooth, near-uniform** across [0,1] with a mild *excess at LOW intensity*
  (density ~1.3 near 0.0–0.15) and a flat plateau (~1.0) through the mid/high range. This is the expected
  signature of the real pipeline's `cv2.equalizeHist` (`util/exp_preprocess.py:92-98`), which flattens
  the histogram, followed by the masked-region zeroing (`:101`).
- **Synthetic (orange): SPIKY and bright-skewed** — sharp density spikes (reaching 3–4×) at discrete
  levels (~0.47, ~0.62, ~0.7, ~0.77, ~0.9–1.0) and *more mass at high intensity* than real. The spikes
  are the **`digitalize_img` quantization** (`simulation.py:933-935`) imprinting discrete grey levels;
  the bright skew is the synthetic contrast chain (`simulation.py:292-300`).

**The residual gap is therefore in the distribution *shape*, not its mean**: synthetic pixels cluster at
a few quantized, brighter levels; real pixels form a smooth, slightly low-skewed continuum. Companion
figures `synth_vs_real_corrected.png`, `synth_after_fix.png`, `synth_vs_real_final.png`
(`diagnostics/README.md` "Other figures") document Path A's before/after of the *masking* fix — i.e. they
show the *spatial* correction, not a distribution-shape correction. **No prior work matched the
per-image intensity *distribution shape* to a real reference.** That is the gap this MVP addresses, and
it is genuinely un-tried.

### 3.3 Why shape (not mean) could matter for faint recall — the hypothesis
Real faint peaks live in the *smooth low-intensity continuum*; synthetic faint peaks are rendered at
*quantized, brighter* levels. A detector trained only on the synthetic rendering learns a faint-peak
appearance (sharp, mid/high grey, few levels) that a real faint peak (smooth, low grey, continuous)
doesn't match → it under-fires on real faint peaks. CDF-matching the synthetic image onto the real
continuum makes synthetic faint peaks render at the same smooth low-contrast levels real ones do — so
the model trains on realistic faint appearance. **This is the mechanism the MVP bets on. It is
plausible, and it is unproven — §8.**

---

## 4. Ranked menu of approaches

| # | Approach | Learned model? | Can move/erase a peak? | ONNX/deploy risk | Effort | Verdict |
|---|----------|----------------|------------------------|------------------|--------|---------|
| **(a)** | **Deterministic statistical matching** (per-image CDF/histogram match synth→real + optional noise model) | **No** | **No** (matching is monotone) / noise capped+guarded | **None** (training-only) | **Low** | **RECOMMENDED MVP** |
| (b) | **Physics-based augmentation** (inject real-like detector noise / background / smearing into synthetic) | No | Only via noise (capped+guarded) | None (training-only) | Low-med | **Recommended add-on / Phase 1** |
| (c) | **Learned neural style transfer / CycleGAN** synth→real | **Yes** (a whole GAN) | **YES — hallucinates/moves content** | None at inference, but... | High | **DECLINE** |

### 4a. Deterministic statistical matching — RECOMMENDED MVP

**What.** For each synthetic training image, apply a **per-image monotone intensity remap** so its
nonzero-pixel intensity CDF matches a **fixed reference CDF pooled offline from real preprocessed
images**. Standard exact-histogram/CDF matching: sort the target reference CDF once; for each synthetic
image, map each valid pixel's rank/quantile through `inverse_reference_CDF(synthetic_CDF(pixel))`. This
replaces the *shape* of the synthetic histogram (spiky, bright) with the *shape* of the real one (smooth,
low-skewed) **while preserving pixel ordering**.

**Why it is the MVP.**
- **Directly closes the §3.2 gap** (spiky/bright → smooth/real) — the one distribution-shape gap no
  prior work touched.
- **Rank-preserving ⇒ HC #2 satisfied by construction** (§2). No peak can be dimmed below its local
  background, because the map is monotone; a faint synthetic peak is *re-rendered at the real faint
  grey-level*, not removed.
- **No learned model, no hallucination, fully deterministic and inspectable.** Cheapest robust option.

**The reference (target) distribution — what to pool, and the leakage note.**
- The target is the output of the **unchanged** `standard_preprocessing` (`util/exp_preprocess.py:247`) →
  `_contrast_correction` (`:62`), i.e. exactly what the deployed model consumes: log10 (`:89`) →
  normalize on the nonzero mask (`:90`) → `equalizeHist` (`:92-98`) → masked pixels = 0 (`:101`).
- **Build the pooled reference CDF from REAL frames that are NOT the eval set.** The curated real
  corpus `backbone_ssl_corpus.h5` (**12,991 frames**, `datasets/real_unlabeled.py:30`, established leak-
  clean vs eval in the SSL work, MEMORY "Backbone SSL dataset") is the natural source — but it must be
  passed through `standard_preprocessing` (or confirmed equivalent) so the reference is *exactly* the
  deployed preprocessing output, not the curation-time `/255` (`backbone_transform.to_model_input`,
  `datasets/real_unlabeled.py:85`). **Do NOT build the reference from organic/41 eval frames.** Using
  eval-frame pixel statistics is the same gray area the SSL work flagged (`ROADMAP.md:60-62`); a pooled
  aggregate histogram leaks little, but the corpus route avoids the question entirely.

**Insertion point (training-only) — see §5 for the exact code site.**

**Faint-peak guardrail.** Matching alone needs none (monotone). If §4b noise is layered on top, apply the
§2 per-box contrast-floor check.

### 4b. Physics-based augmentation — recommended add-on (Phase 1)

**What.** Inject *real-derived* detector noise / background / smearing into synthetic, calibrated to real
statistics: estimate the residual (high-frequency) noise std and background texture in real *background*
(non-peak, valid) regions, and add comparable low-amplitude noise to synthetic; optionally a mild
q-direction intensity ramp / point-spread smearing to mimic detector response.

**Why it complements (a).** CDF matching fixes the *marginal* intensity histogram; it does not fix the
*spatial noise texture*. Real background has a characteristic grain that synthetic (Poisson +
salt-pepper, `simulation.py:257,287`) may not match. Matching the noise texture makes real background
"look normal" to the encoder → fewer confident on-ring hallucinations (the precision failure,
`ROADMAP.md:76-90`) and a more realistic faint-peak SNR context.

**Building blocks already exist in the repo** — reuse them rather than reinvent: the SSL/semi
augmentations `augment_v2` (`backbone_curation/backbone_transform.py:57`) and `photometric_strong`
(`datasets/real_unlabeled.py:33`) already implement polar-legal gamma, exposure scale, q-ramp, and
additive-noise ops that keep no-data = 0. These are *generic* augmentations, though; the physics-add-on's
novelty is **calibrating the noise amplitude to a real-measured background std** rather than a hand-set
constant (`rng.normal(0, 0.03)`, `:54`).

**Faint-peak guardrail.** Additive noise is the one erasure risk — cap amplitude below the faintest
labeled-peak local contrast and apply the §2 per-box contrast-floor fail-safe. **Never** a blur/denoise
that removes peak signal.

### 4c. Learned neural style transfer / CycleGAN synth→real — DECLINE

**Assess, then decline.** A CycleGAN/neural-ST would learn a synth→real appearance mapping. It is
rejected here for reasons specific to *this* problem, not generic GAN skepticism:

1. **It can move or erase faint peaks — violating HC #2 at its core.** A GAN's mapping is *not*
   rank-preserving and *not* label-aware. It can hallucinate texture, shift a faint arc, or smooth it
   away — and then the **peak labels (`boxes`, `is_ring`, `MODIFICATIONS.md:49-50`) no longer describe
   the transformed image.** The very peaks we care about (faint, low-SNR) are exactly the ones a GAN is
   most likely to alter, because they carry the least signal to preserve. This is a *label-integrity*
   failure, not just a quality one.
2. **A second model to train, tune, and trust**, with mode-collapse and cycle-consistency failure modes,
   on a domain (single-channel 512×1024 polar diffraction) with no pretrained ST model and a *small* real
   set. High effort, high variance.
3. **Harder to keep the A/B honest.** With a deterministic remap you can *inspect* every transformed
   image and *prove* peaks are preserved. With a GAN you cannot, so a recall change is confounded by
   "did the GAN keep the peaks?".
4. **No upside over (a)+(b) for our stated goal.** Our gap is *statistical* (histogram shape + noise
   texture, §3), which (a)+(b) match deterministically. A GAN's extra power (learning *structural*
   appearance) is precisely the power that endangers the labels. Wrong tool.

**If ever revisited**, the only label-safe GAN variant would be one constrained to a *residual, monotone,
peak-preserving* intensity map (i.e. a learned version of (a)) — at which point (a) already gives the
safe part for free. Declined.

---

## 5. Exact insertion point (`file:line`) + value range at that point

### 5.1 Where the synthetic tensor is produced and fed to the model

`SimulationDataset.__getitem__` (`main.py:61-86`) is the single point where a synthetic image tensor is
produced and handed to the training loop:

- `main.py:65` — `image, boxes, mask, is_ring = self.simulation.simulate_img()`. `simulate_img`
  (`simulation.py:209`) returns `clahe_img` after the contrast chain and a final `normalize`
  (`simulation.py:300`, `normalize` = min-max to **[0,1]**, `simulation.py:38-39`). So at `main.py:65`
  **`image` is a single-channel `H×W` cuda float tensor in [0,1]**, and **`mask`** is the validity mask
  (True = valid pixel; `simulation.py:523,533,541` return `~<dark>`).
- `main.py:69` — `image = image.repeat(self.args.num_channels, 1, 1)` broadcasts the single logical
  channel to `num_channels` (=1, `config/DINO/DINO_4scale_swin.py:40`).
- `main.py:74-86` — boxes/labels/targets are built; `(image, target)` is returned to `collate_fn`
  (`main.py:92`) → the model.

**Insertion: between `main.py:65` and `main.py:69`** — immediately after `simulate_img()` returns and
**before the channel repeat**, on the single-channel [0,1] tensor, using `mask` to define valid pixels:

```python
# main.py, SimulationDataset.__getitem__, immediately after line 65 (image from simulate_img)
if getattr(self.args, 'use_style_match', False):
    image = style_match(image, mask, self._ref_cdf)   # NEW: monotone CDF match on valid pixels
# ... existing line 69: image = image.repeat(self.args.num_channels, 1, 1)
```

- `style_match` lives in a **new** small module (e.g. `util/style_match.py`) or beside the sim; `_ref_cdf`
  is the pooled real reference CDF, loaded once in `SimulationDataset.__init__` (`main.py:54`) when the
  flag is set (default-off → byte-identical to today).
- **Why here, not inside `simulate_img`:** `SimulationDataset.__getitem__` is the exact boundary between
  "image generation" and "fed to model," it already has both `image` and `mask` in scope, and it is
  *structurally* training-only (§6). Keeping the transform out of `simulation.py` also keeps the sim core
  untouched and the change trivially revertable (mirrors the discipline of the reverted Path A, which
  edited `simulation.py` and had to be `git checkout`-ed, `MODIFICATIONS.md:75-77`).
- **Flag gating** follows the repo's established `getattr(args, 'flag', False)` pattern (as `use_cctm`,
  `use_boost_loss`, `use_co_heads`, `use_semi` all do) — absent the flag, the training path is
  byte-identical to today.

### 5.2 The target distribution (what real preprocessing produces)

`standard_preprocessing` (`util/exp_preprocess.py:247`) → `contrast_correction` (`:105`) →
`_contrast_correction` (`:62`):
- optional percentile clip (`:81-86`), **log10** (`:89`), **normalize on nonzero mask** to [0,1]
  (`:90`, `:34-35`), **`cv2.equalizeHist`** (`:92-98`, the flatten-to-uniform step), **masked pixels set
  to exactly 0** (`:101`). Output: `float32` in [0,1], near-uniform histogram, masked = 0.
- Note a residual sim/real convention gap the MVP should respect: current (reverted-Path-A) synthetic
  dark regions are set to a *gray* `level ∈ [-0.1, 0.5]` (`simulation.py:509,522,532`), whereas real dark
  regions are exactly 0 (`:101`). Build the reference CDF from **valid (nonzero)** real pixels and apply
  the match to **valid** synthetic pixels only (using `mask`), so this convention difference does not
  contaminate the mapping.

### 5.3 The faint/high-q recall probe (the gate)

`/mnt/lustre/work/schreiber/szb389/tmp_diag/diag_compare.py` — loads a checkpoint, runs the deployed
postprocessing (`onnx_to_xyxy` + `filter_boxes`) on `organic_labeled.h5`, and prints **recall by
visibility (3/2/1), by type (ring/segment), and by q-third (0-341/341-682/682-1024)**, plus FP/img and
precision. This is the exact probe that produced the 0.33 (vis=1) / 0.44 (q682-1024) baselines. **It is
the gate**, run on the MVP checkpoint vs the ssl1 baseline.

---

## 6. ONNX / deploy-safety — the isolation is structural and airtight

Two independent facts guarantee the inference path, preprocessing, and exported `.onnx` are byte-for-byte
unchanged:

1. **`SimulationDataset` is training-only.** It is imported and instantiated **only** in the training
   entry points — `main.py:235` (`main()` setup) and `main.py:412` (per-epoch rebuild inside the training
   loop). The evaluation path `evaluate_giwaxs_ap` (`main.py:182-226`) builds `PyGIDDataset` /
   `H5GIWAXSDataset` with `standard_preprocessing` (`main.py:206,209`) — it **never** touches
   `SimulationDataset` or `simulation.py`. So the style transform is on a code path that inference and
   `--eval` cannot reach.
2. **The exporter never imports the sim.** `export_onnx.py` imports only `torch`, the model registry, and
   the model (`export_onnx.py:1-5`); `DINOOnnxWrapper.forward` returns only
   `out["pred_logits"], out["pred_boxes"]` (`export_onnx.py:26-27`) and runs `model.eval()` on CPU
   (`:30-37`). The transform changes only the *training input distribution* → it changes only the
   *weights* in `checkpoint.pth`, exactly like Semi-DETR / Boost / CCTM-off / Co-DINO
   (`MODIFICATIONS.md:129`, `docs/CO_DINO_INVESTIGATION.md:371`).

**Nothing in `standard_preprocessing`, the ONNX graph, or the deployed forward changes.** The transform
is *upstream of the synthetic image only*, at training time only.

---

## 7. Reconciling with Path A (TRIED & REVERTED) — how this differs and what it must learn

Path A (`MODIFICATIONS.md:72-121`) is the closest precedent and its null result is the single most
important prior. Read it carefully:

**What Path A changed:** it edited `simulation.py` to (i) fix the `calculate_angle_limits_mask`
WIDTH-regression bug so the synthetic **masked fraction** matched real (3.5% → ~0.30), (ii) set masked
regions to **0** (was gray) to match real, and (iii) raise `digitalize_img` from 16–64 to **128–256**
levels (`MODIFICATIONS.md:92-108`).

**Why it gave no gain:** retrain (`ringseg_2class_pathA_...`) tied organic AP (~0.52) and slightly *hurt*
41 (`MODIFICATIONS.md:114-119`). The stated conclusion: *"the masking distribution gap was real but not
performance-limiting — DETR already ignores zero regions; the heavier masking removes some high-q peaks
from training labels, marginally hurting 41."*

**How the proposed MVP differs — and the honest risk that it doesn't differ enough:**
- **Different lever.** Path A matched a *spatial* property (masked fraction) + a *coarse* quantization
  count. It did **not** match the per-image *intensity distribution shape* to a real reference, and added
  **no real-derived noise model**. The MVP is a *distribution/appearance-matching* transform (§3.2 shows
  the shape gap Path A left untouched). So it is genuinely a different experiment.
- **But the digitalize bump is a warning.** Raising `digitalize_img` to 128–256 levels *already partially
  smoothed the spikes* (§3.2's spikes come from that op) — and it did nothing. CDF matching smooths the
  spikes *further and exactly*, plus fixes the bright-skew, so it is a stronger version of the same idea.
  **If the residual spikiness/skew were the bottleneck, Path A's digitalize bump should have shown at
  least a flicker. It didn't.** That is real evidence *against* the MVP's mechanism, and it must be stated
  up front: **the MVP is at material risk of re-treading Path A.** The counter-argument is that Path A
  matched the *count* of levels but not the *shape* (still uniform-quantized, still bright-skewed), and
  never matched a real *continuum* — so the bet is that *shape*, not *level count*, is what the encoder
  reads. This is a thin distinction, and §8's gate exists precisely to kill it fast if it is illusory.
- **What the MVP must learn from Path A's failure analysis:** Path A's heavier masking "removes some
  high-q peaks from training labels, marginally hurting 41" (`MODIFICATIONS.md:118-119`). The MVP must
  **not** repeat this — the monotone remap does not remove peaks (§2), but the §4b noise step could, which
  is why it is capped and guarded. And the MVP must be watched on **41** (the easier control) for the same
  regression, not just organic.

**Bearing of `ROADMAP.md`'s KEY FINDING on the A/B interpretation** (`ROADMAP.md:64-91`,
`MODIFICATIONS.md:211-215`): the organic eval is *plausibly label-limited* — expert review confirmed many
confident FPs are genuine hallucinations *on real rings*, but the faint/high-q **misses** may include
peaks the GT itself under-labels. Consequence for the A/B: **a genuine faint-recall improvement may not
show up in organic AP** (a recovered-but-unlabeled faint peak scores as an FP, not a TP). This is *the*
reason the gate is the **recall probe stratified by visibility/q** (§5.3), read alongside AP — and why an
"AP flat, faint/high-q recall UP" outcome should be treated as a *win-with-caveat*, not a null (the
inverse of the CCTM "AP-up, recall-flat = decline" rule, `MODIFICATIONS.md:352`).

---

## 8. Adversarial: will this actually help, and how could it backfire?

This section decides the project. Be honest.

**The mechanism that could work (why we build it).** §3.3: real faint peaks live in a smooth
low-intensity continuum; synthetic faint peaks render at quantized/brighter levels. The encoder, trained
only on synthetic, learns the wrong faint-peak appearance and under-fires on real faint peaks (the 0.33
vis=1 recall hole). A monotone CDF match re-renders synthetic faint peaks at the *real* faint grey-levels
without moving them → the encoder trains on realistic faint appearance → higher real faint recall. Of the
levers tried, this is the only one that changes *what a faint peak looks like in training*, which is the
most direct attack on a "the data doesn't show real faint peaks" ceiling.

**Why it might NOT (the failure modes to watch):**
1. **The Path-A precedent (§7).** The mean was already matched and the digitalize-level bump (a partial
   version of this) did nothing. If *shape* is not what the encoder reads, this is a null. **Primary
   risk.**
2. **Sensitivity may be a representation ceiling the backbone can't cross regardless of input styling.**
   `ROADMAP.md:26-31` calls the faint/high-q miss a *representation/sensitivity ceiling*; the CCTM
   diagnostic showed faint(vis=1) recall pinned at 0.330→0.330 even when easy-peak recall rose
   (`MODIFICATIONS.md:311-314`). If the backbone simply cannot encode a low-SNR arc, styling the input to
   look real won't manufacture the missing sensitivity — same as CCTM couldn't (`MODIFICATIONS.md:315-318`).
   Appearance-matching *helps the encoder learn the right target*; it does not add capacity. If the
   ceiling is capacity, this fails.
3. **Faint-peak erasure (HC #2) via the noise step.** Guarded (§2), but if the cap is set too loose the
   §4b noise buries faint synthetic peaks → *worsens* faint recall. Keep matching-only for the MVP;
   add noise only in Phase 1 with the per-box floor active.
4. **41 regression.** Path A hurt 41 by dropping high-q training peaks (`:118`). Watch 41.
5. **Label-incompleteness masks a real win (§7).** A genuine faint-recall gain may not lift organic AP.
   Mitigated by gating on the recall probe, not AP.
6. **Reference-CDF representativeness.** If the pooled real reference over-represents one geometry/detector,
   the match biases synthetic toward it. Pool broadly across the 12,991-frame corpus; diversity matters
   more than count (`ROADMAP.md:49-51`).

**How we know early (cheap falsification):**
- **Gate on the recall probe, stratified.** Run `diag_compare.py` on the MVP checkpoint vs ssl1. Success
  = **faint(vis=1) and high-q(q682-1024) recall UP**. If those two are flat while AP nudges, it is the
  CCTM easy-peak-refinement null shape (`MODIFICATIONS.md:352`) → decline.
- **Inspect transformed images (free, unique to the deterministic route).** Overlay boxes on a batch of
  style-matched synthetic images and confirm every labeled peak is still a visible local maximum (the
  `viz_fp.py` overlay pattern, `diagnostics/viz_fp.py`). This *proves* HC #2 holds — impossible with a GAN.
- **Watch 41** for the Path-A-style high-q regression.

---

## 9. How to A/B (warm-start vs from-scratch) + the gate

**Baselines.** ssl1 = SSL-pretrained backbone (`backbone_dir`, `config/DINO/DINO_4scale_swin_ssl.py:12`)
+ detector trained under the ssl1 recipe → **organic 0.586 / 41 0.762** (`MODIFICATIONS.md:361-364`,
`config/DINO/DINO_4scale_swin_codino_scratch.py:9`). This is the strongest single model and the A/B
reference.

**Warm-start vs from-scratch — the reasoning (and where this differs from architectural changes).**
The repo's established discipline: *architectural / training-scheme* changes that reshape **encoder
feature learning** (Co-DINO aux heads) demand a **from-scratch** co-train, because a warm-start from a
converged encoder "can't fully reorganize" and under-tests the change
(`config/DINO/DINO_4scale_swin_codino_scratch.py:3-13`). CCTM likewise needed a 10× LR graft trick to
gain traction from warm-start (`MODIFICATIONS.md:296-304`).

**A data-distribution change is a *different category* and is arguably fair to test from warm-start:**
- It adds **no architecture** and changes **no supervision** — it only shifts the *input distribution*.
  The model just needs to re-adapt its features to new input statistics, which is a lighter lift than
  reorganizing around a new loss/head. A warm-start from ssl1 (whose backbone already saw *real* texture
  via SSL) is a legitimate, cheap first-look.
- **Recommended plan:**
  1. **Phase 0 (screen) = warm-start fine-tune from the ssl1 checkpoint** with `use_style_match=True`,
     everything else identical. Cheapest possible A/B; eval every 2 epochs is already wired
     (`config/DINO/DINO_4scale_swin.py:5-10`). Gate at ~20-40 epochs on the recall probe.
  2. **Phase 2 (confirm, only if Phase 0 is non-negative) = from-scratch** on the ssl1 recipe
     (`_base_ = ['DINO_4scale_swin_ssl.py']`) with the transform on from epoch 0. This is the *clean*
     apples-to-apples vs ssl1 (0.586/0.762): the **only** difference is the synthetic input transform,
     and the encoder co-adapts to the real-matched appearance from init. The honest caveat: a warm-start
     screen can *under*-show the benefit (the backbone was trained on old-appearance synthetic), so a
     *negative* Phase-0 is not fully conclusive — but a *positive* Phase-0 is a strong go-signal, and
     from-scratch is a 500-epoch cost only spent once the cheap screen passes.

**The gate (both must be read):**
- **Organic AP** (`config/DINO/DINO_4scale_swin.py:7`) — the primary AP number, with the §7
  label-limitation caveat.
- **The faint/high-q recall probe** (`diag_compare.py`, §5.3) — the *decisive* readout, because a real
  faint-recall gain may not surface in AP (§7, §8).
- **41 AP** — the easy control; must not regress Path-A-style.

Decision shapes (mirroring `MODIFICATIONS.md:352`):
- faint/high-q recall **UP** (AP up or flat) → **win** (flat-AP-but-recall-up = win-with-caveat, §7).
- faint/high-q recall **flat**, AP up → easy-peak refinement (CCTM null shape) → **decline**.
- faint/high-q recall flat, AP flat/down → **decline** (Path A re-tread confirmed).

---

## 10. Phased plan

**Phase 0 — deterministic CDF matching, matching-only (the real test).**
- **Build:** offline reference-CDF builder (pool nonzero pixels from N real corpus frames passed through
  `standard_preprocessing`, save a quantile table); `style_match(image, mask, ref_cdf)` (monotone
  per-image CDF match on valid pixels); the `main.py:65→69` call site; the `use_style_match` flag +
  reference path in the config; load `_ref_cdf` in `SimulationDataset.__init__`.
- **Verify before training (free):** overlay boxes on style-matched synthetic images (§8) → every labeled
  peak still a visible local max (HC #2 proof). Confirm masked pixels handled per §5.2.
- **Run:** warm-start fine-tune from ssl1 (§9 Phase 0); A/B vs ssl1 on organic + 41 every 2 ep.
- **Read-out (the whole experiment):** `diag_compare.py` recall probe (faint vis=1 / high-q q682-1024)
  vs ssl1, alongside organic AP. Apply §9 decision shapes. **This is the gate.**

**Phase 1 — physics noise add-on (only if Phase 0 is non-negative).**
- Add real-calibrated background-noise injection (§4b) on top of the matched image, amplitude capped +
  per-box contrast-floor guard active. Reuse the `augment_v2`/`photometric_strong` op inventory
  (`backbone_curation/backbone_transform.py:57`, `datasets/real_unlabeled.py:33`) but calibrate noise std
  to a real-measured background std. Re-gate on the recall probe + 41 (watch for erasure regressions).

**Phase 2 — from-scratch confirmation (only if Phase 0/1 clearly help).**
- Retrain from scratch on the ssl1 recipe with the transform on from epoch 0 (§9 Phase 2). Clean A/B vs
  ssl1 (0.586/0.762). This is the number that would justify deployment.

**Phase 3 — combine / tune (optional).**
- Sweep the reference-CDF source (corpus subset vs broad), the noise cap, and whether to match *only*
  low-intensity (faint-focused partial match) vs full-range. Re-gate.

**Stop after Phase 0 if the gate fails (§11).**

---

## 11. Stop rule

1. **Phase 0 is the gate.** If matching-only, fine-tuned to convergence (~20-40 ep warm-start) from ssl1,
   does **not** raise faint(vis=1) **or** high-q(q682-1024) recall vs ssl1 on `diag_compare.py` →
   **decline style-transfer.** Do not proceed to physics noise or from-scratch. Rationale: matching is
   the cleanest, safest, most direct version of the appearance-match hypothesis; if the *exact*
   distribution match moves nothing, the residual-shape hypothesis (§3.3) is falsified and the Path-A
   verdict (`MODIFICATIONS.md:114-121`) stands — the bottleneck is representation capacity
   (`ROADMAP.md:26-31`), not input appearance.
2. **AP-up-but-recall-flat = decline, not proceed** (CCTM null shape, `MODIFICATIONS.md:352`). Our
   ceiling is recall; refining bright peaks is not progress on it.
3. **Recall-up-but-AP-flat = win-with-caveat, not decline** (§7 label-limitation) — proceed to Phase 2 to
   confirm, and flag the eval-label question for expert review (`ROADMAP.md:78`, the standing `viz_fp`
   review path).
4. **41 regression escape hatch.** If 41 drops Path-A-style, check whether the transform (esp. §4b noise)
   is erasing high-q training peaks before concluding anything — fix the guard once before declaring a
   result.

---

## 12. Risks / unknowns (explicit)

1. **Path-A re-tread (primary).** The digitalize-level bump already partially smoothed the histogram with
   no gain (§7). The MVP bets that *distribution shape*, not *level count*, is the readable difference —
   thin, hence the hard gate.
2. **Representation-capacity ceiling.** If faint peaks are un-encodable by the current backbone regardless
   of input styling (CCTM saw exactly this, `MODIFICATIONS.md:311-318`), appearance-matching cannot help.
   Appearance-matching complements, and does not replace, the SSL-backbone lever (`ROADMAP.md:32-62`).
3. **Faint-peak erasure via noise (HC #2).** Matching is safe (monotone); the §4b noise step is the only
   erasure path — capped + per-box floor + inspect-before-train. Keep MVP matching-only.
4. **Label-incompleteness confound.** A real faint-recall win may not surface in organic AP
   (`MODIFICATIONS.md:211-215`). Gate on the recall probe.
5. **Reference-CDF representativeness / leakage.** Build from the real corpus (`datasets/real_unlabeled.py:30`),
   NOT eval frames; pool broadly; pass through `standard_preprocessing` so the target = deployed output.
6. **Warm-start under-test.** A negative Phase-0 warm-start is not fully conclusive (the backbone was
   trained on old-appearance synthetic); a positive one is a strong go. From-scratch confirmation is the
   deployable number (§9).
7. **Interaction with in-flight flags.** `use_style_match` is an orthogonal, default-off training-only
   flag (as `use_cctm`/`use_boost_loss`/`use_co_heads`). Keep others off for the clean Phase-0 A/B.

---

## 13. References + repo anchor points

### Prior work reconciled
- **Path A (TRIED & REVERTED, no AP gain):** `MODIFICATIONS.md:72-121` — masking-fraction + digitalize
  fix; the closest precedent (§7).
- **Diagnostic C / KEY FINDING (synth→real content gap, label-limitation):** `ROADMAP.md:14-91`,
  `MODIFICATIONS.md:368-378`.
- **Detector-side levers that were null/negative on organic:** Semi-DETR (`MODIFICATIONS.md:125-221`),
  Cross-DINO Boost (`:223-262`), CCTM (`:264-324`), Co-DINO (`:326-353`); sibling docs
  `docs/CROSS_DINO_INVESTIGATION.md`, `docs/CO_DINO_INVESTIGATION.md`, `docs/SEMI_DETR_INTEGRATION.md`.
- **The pixel-distribution diagnostics:** `diagnostics/synth_vs_real_hist.png` (spiky/bright vs
  smooth/real, §3.2), `synth_vs_real_corrected.png`, `synth_after_fix.png`, `synth_vs_real_final.png`
  (Path A masking before/after); `diagnostics/README.md` "Other figures".

### Repo anchor points (for the implementer)
- **Synthetic-input insertion (training-only):** `main.py:61-86` (`SimulationDataset.__getitem__`);
  insert between `main.py:65` (`simulate_img()` return, image in [0,1], `mask` = validity) and `main.py:69`
  (channel repeat). `SimulationDataset` built only at `main.py:235` and `main.py:412`.
- **Synthetic image production + range:** `simulation.py:209` (`simulate_img`), `:292-300` (contrast
  chain), `:300` + `:38-39` (final `normalize` → [0,1]), `:933-935` (`digitalize_img`, 16-64 levels =
  the histogram spikes), `:505-543` (dark-area mask, returns validity mask; dark = gray `level`).
- **Real preprocessing (target distribution):** `util/exp_preprocess.py:247` (`standard_preprocessing`),
  `:62-103` (`_contrast_correction`: log10 `:89`, normalize-on-mask `:90`+`:34-35`, `equalizeHist`
  `:92-98`, masked=0 `:101`).
- **Real reference corpus (leak-clean, non-eval):** `datasets/real_unlabeled.py:30` (`backbone_ssl_corpus.h5`,
  12,991 frames), MEMORY "Backbone SSL dataset".
- **Physics-noise op inventory (reuse for §4b):** `backbone_curation/backbone_transform.py:57`
  (`augment_v2`), `datasets/real_unlabeled.py:33` (`photometric_strong`).
- **The gate — faint/high-q recall probe:** `/mnt/lustre/work/schreiber/szb389/tmp_diag/diag_compare.py`
  (recall by visibility/type/q-third on `organic_labeled.h5`).
- **ONNX / deploy isolation:** `export_onnx.py:1-5` (imports — no sim), `:26-27` (output whitelist),
  `:30-37` (eval-mode CPU export); eval path `main.py:182-226` (`standard_preprocessing`, no sim).
- **A/B baseline + config:** ssl1 (`config/DINO/DINO_4scale_swin_ssl.py:12` `backbone_dir`; organic
  0.586 / 41 0.762, `MODIFICATIONS.md:361-364`); eval sets + interval + LR/epochs/lr_drop
  `config/DINO/DINO_4scale_swin.py:5-10,15,24-25`; from-scratch recipe pattern
  `config/DINO/DINO_4scale_swin_codino_scratch.py:14`.
- **Flag-gating precedent:** `getattr(args, 'flag', False)` — as `use_cctm`, `use_boost_loss`,
  `use_co_heads`, `use_semi`.
