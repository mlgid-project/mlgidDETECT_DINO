# SSL round-2 recipe (simmim2)

> **STATUS: IMPLEMENTED (2026-06-12).** `backbone_transform.py::augment_v2` (v1 untouched),
> `--aug v1|v2` flag in `train_simmim.py`/`ssl_dataset.py`, launch via `run_simmim2.sbatch`
> (→ `ssl_runs/simmim2`, 150 epochs — round-1 converged by ~ep40-77, so 150 with full cosine
> anneal fits one 72h session). simmim1 was NOT resubmitted past its wall (flatlined).

Parked plan for a second SimMIM pretraining run, to try **once the current simmim1 (ep77)
detector run lands its AP plateau** and we know how much round-1 SSL actually helped.

Round-1 status it builds on: simmim1 val L1 converged flat at ~0.1096 from ~ep40 (near the
irreducible detector-noise floor — lower recon ≠ better features). The ep77 backbone already
matches/beats the from-scratch baseline on the detector (organic +~0.03 at matched epochs), so
**more epochs is not the lever** — the levers are *harder pretext + stronger aug + corpus diversity*.

---

## Changes vs round-1

Three coordinated changes: harder mask, richer (still polar-legal) aug, fresh out_dir so simmim1
is untouched. Corpus diversity is the 4th lever (data task, see bottom).

### 1) `run_simmim.sbatch` — harder mask, fresh dir
```diff
-OUT_DIR=.../ssl_runs/simmim1
+OUT_DIR=.../ssl_runs/simmim2          # don't overwrite the v1 backbone/curves

 srun $PY -u train_simmim.py \
     --epochs 200 \
     --batch 16 --accum 2 \
     --lr 1.5e-4 --warmup_epochs 10 --wd 0.05 \
-    --mask_ratio 0.6 --mask_patch 32 --drop_path 0.2 \
+    --mask_ratio 0.70 --mask_patch 32 --drop_path 0.25 \
     --out_dir "$OUT_DIR" $RESUME
```
0.6 → 0.70 makes reconstruction harder → forces more global/structural features. Val L1 will read
**higher** than simmim1 — expected, NOT a regression. `drop_path` 0.2 → 0.25 to match the stronger reg.

### 2) `backbone_transform.py::augment` — richer, polar-legal aug
```diff
 def augment(x, rng):
     m = x > 0
     # random vertical (chi) flip
     if rng.random() < 0.5:
         x = x[::-1].copy(); m = m[::-1].copy()
-    # intensity gamma on valid pixels
-    g = rng.uniform(0.8, 1.25)
+    # wider intensity gamma on valid pixels
+    g = rng.uniform(0.7, 1.40)
     x = np.where(m, np.clip(x, 1e-6, 1.0) ** g, 0.0).astype(np.float32)
-    # mild additive noise on valid pixels
-    if rng.random() < 0.5:
-        x = np.where(m, np.clip(x + rng.normal(0, 0.02, x.shape), 0, 1), 0.0).astype(np.float32)
+    # global exposure scale (detector dose / acquisition-time variation)
+    if rng.random() < 0.7:
+        x = np.where(m, np.clip(x * rng.uniform(0.8, 1.2), 0, 1), 0.0).astype(np.float32)
+    # smooth q-direction intensity ramp (sample absorption / footprint along q)
+    if rng.random() < 0.5:
+        ramp = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.85, 1.0), x.shape[1])[None, :]
+        x = np.where(m, np.clip(x * ramp, 0, 1), 0.0).astype(np.float32)
+    # additive noise on valid pixels
+    if rng.random() < 0.7:
+        x = np.where(m, np.clip(x + rng.normal(0, 0.03, x.shape), 0, 1), 0.0).astype(np.float32)
     return x
```
Why each is physics-legal: **vflip** = chi convention; **gamma/exposure** = dose & contrast
variation; **q-ramp** = absorption/footprint falloff along q (a real GIWAXS effect); **noise** =
counting statistics. This aug also doubles as the mitigation for the **moneta-vs-mlgidDETECT histeq
mismatch** (both are log+histeq polar [0,1], but with slightly different contrast curves — the
gamma/exposure/ramp make the backbone invariant to that shift).

**Deliberately excluded:** arbitrary rotation (breaks q-radial geometry); azimuthal (chi) roll
(only valid for full-360° cakes — our frames include wedges/segments, so a roll would wrap rings
incorrectly).

> Note: round-1 (simmim1) shares this same `augment`. If you edit it in place, simmim1 can't be
> reproduced exactly — either branch/copy the file or gate the new behavior behind a flag if that
> reproducibility matters.

### 3) Corpus diversity — highest-value lever (data task, out of code scope)
Add more **distinct scans**, not more frames of the scans we already have — diversity is the
limiter, not frame count. When more real shards arrive: re-run `finalize_manifest.py` with a higher
`N_TARGET`, rebuild with `corpus_builder.py`. Even +200 new scans would matter more than any knob above.

---

## How to judge simmim2 (do NOT compare val losses)

Higher mask ratio means simmim2's val L1 reads **worse** than simmim1 by construction — comparing
recon losses across recipes is meaningless. The only valid A/B is **downstream**:

1. let simmim2 finish (or take a mid checkpoint), `export_backbone.py` → `backbone_export/`
2. point a detector run at it (new `DET_RUN`, e.g. `detector_runs/dino_ssl2`)
3. compare its **AP plateau on organic / 41** against the current simmim1 detector run
   (`compare_ap.py`), same as the baseline comparison

Win = higher plateau (esp. organic) and/or faster early climb than the simmim1-backbone detector.

## Context / pointers
- baseline to beat (from-scratch): 41 ~0.768 · organic ~0.554
- simmim1-backbone detector (the round-1 result to beat): see `dino_ssl-*.out` / `plot_state.py`
- env `DINO_GIWAXS`; A100 / 72h / idempotent auto-resume (same as round-1)
