# SSL round-3 recipe / idea pool (simmim3)

Candidates to implement **if round 2 (simmim2) is not fully satisfactory**. Judged, as always,
ONLY by downstream detector AP (organic / 41), never by SSL val loss.

## How to read round-2 results first (decision guide)

Compare the simmim2-backbone detector vs the simmim1-backbone detector (`compare_ap.py`):

| outcome | what it means | round-3 move |
|---|---|---|
| v2 > v1 | harder mask + richer aug helped | stack idea 1 (mask shape) ON TOP of v2 recipe |
| v2 ≈ v1 | recipe knobs saturated on this corpus | idea 1 as last cheap knob; otherwise idea 4 (corpus) is the only real lever |
| v2 < v1 | a knob hurt (likely aug too strong or 0.70 too hard) | A/B the knobs separately (mask 0.70 + v1 aug, and v2 aug + mask 0.6) before anything new |

---

## Idea 1 — Anisotropic (tall-narrow) masking  [primary candidate, ~10 lines]

Match mask anisotropy to the data's redundancy structure (and the 48x6 windows):
along chi the image is redundant (rings continue smoothly -> square blocks are filled by
cheap vertical copying); along q it is informative (neighbors say nothing about masked q).
Tall-narrow blocks kill the vertical-copy shortcut and force "does this feature continue
or end?" reasoning = ring-vs-segment knowledge the detector needs.

**Hard limit:** never erase all of chi at some q — peak existence there becomes
unrecoverable (q-neighbors can't supply it) and the model trains to hallucinate.
At 70% ratio, P(full block-column masked): 32x32 cells ~0.3% (fine), 64-tall ~6%
(borderline), 128-tall ~24% (bad).

**Spec:**
- cells **64x16 px** (grid 8x64 = 512 cells — same count & area as now, A/B isolates pure shape)
- **column-balanced sampling**: mask exactly k=round(ratio*8) of the 8 cells per block-column
  -> every q always keeps visible chi-evidence; removes the failure mode entirely
- implementation: `MaskGenerator(mask_patch_h, mask_patch_w)` + per-axis `tok/pix_factor`
  in `simmim_model.py` (mask already upsampled with separate per-axis repeat_interleave)

Expectation: second-order knob vs ratio/cell-size — small gain at best.

## Idea 2 — Scan-aware contrastive auxiliary loss  [medium effort]

Frames of the SAME scan are natural positive pairs (same sample, slightly evolved) — the
one contrastive variant that survives our corpus redundancy (~75% near-dups make standard
in-batch negatives false). Add a small InfoNCE/BYOL-style term on pooled encoder features:
positives = two frames of one scan, negatives = other scans only.
Risk: loss balancing (recon vs contrastive); keep weight small (e.g. 0.1).

## Idea 3 — Hybrid content-driven corpus sampling  [cheap, uses existing tools]

Re-select the ~13k with `select_dissimilar.py` logic: T>2 distinct core (4,592 frames,
non-redundant by construction) + even-spaced per-scan fills up to budget. Same data volume,
more information-dense epochs. Build via a `manifest_dissim_*`-style manifest +
`corpus_builder.py` pointed at it (separate h5 — do NOT touch backbone_ssl_corpus.h5).

## Idea 4 — Corpus diversity  [highest expected value, data acquisition]

The standing #1 lever: more DISTINCT scans (new beamtimes/materials), not more frames.
Measured: current 68k contains only ~4.6k genuinely distinct frames; recipe knobs cannot
substitute for new samples. Then: finalize_manifest with higher N_TARGET -> rebuild corpus.

## Idea 5 — Detector-side fine-tuning schedule  [no new SSL run needed]

If SSL features are good but gains fade late in detector training, tune the transfer:
- lower `lr_backbone` (preserve pretrained features longer) or freeze stages 1-2 early on
- longer detector schedule with later lr_drop (baseline peaked at ep338-360)
Cheap to test against the EXISTING simmim1/simmim2 exports — try before any new SSL run.

## Idea 5b — Two-phase / annealed backbone freeze  [no new SSL run needed]

Rationale: the detector trains on SIMULATED data only — with all stages trainable at
lr_backbone == lr (current setup, nothing frozen), the long tail of epochs slowly overwrites
the SSL real-data features (wash-out). But freezing from epoch 0 forfeits the useful early
adaptation (SimMIM features aren't detection-optimal; the steep early AP climb IS that
adaptation). So: **adapt early, freeze late** -> a backbone that is "adapted to detection
but still SSL-flavored".

- **Phase A** (ep 0..N): all stages trainable (as now).
- **Phase B** (ep N..end): freeze the backbone — or the soft variant: decay lr_backbone
  (e.g. cosine -> 0 on the backbone param group only) instead of a hard cut; same effect,
  no optimizer discontinuity.
- **Choosing N (new hyperparameter):** data-driven — freeze when the SSL-vs-baseline AP
  delta peaks / starts shrinking, or where the steep climb ends (current curve: ~ep100-150).
  Per-epoch eval (eval_interval=2) gives the trigger directly.
- **Implementation:** stop run at ep N -> switch config (backbone_freeze_keywords or
  lr_backbone=0) -> resume from checkpoint. **GOTCHA to verify first:** optimizer
  load_state_dict restores the SAVED lrs, so check that main.py re-applies config lrs after
  resume (or patch it to). 
- **Snapshots:** to A/B several freeze points, keep periodic checkpoint copies (e.g.
  ep100/150/200) while the run passes them — checkpoint.pth alone is overwritten.
- **Decision signal:** only worth running if the current dino_ssl curve shows the wash-out
  signature (delta vs baseline shrinking toward 0 in late epochs). If the delta holds to
  plateau, skip — the init bias persists on its own.

## Idea 6 — DINO-5scale: feed the stride-4 level to the detector  [expensive, config-only]

(Origin: colleague's suggestion to pass levels 1,2,4,5 instead of 2,3,4,5.)
Currently DINO-4scale: detector consumes backbone stages 2-4 (strides 8/16/32) + an extra
stride-64 map; the SHARPEST level (stage 1, stride 4) is computed and thrown away.
Narrow features (peaks 10-30 px in q ~ 1.5-4 tokens at stride 8) gain q-localization from
stride 4 — and the eval's q-matcher rewards exactly that.

Refinements vs the original suggestion:
- dropping level 3 saves only ~5% (level 1 alone = ~75% of tokens: 32,768 of 43,648) and
  non-contiguous selection isn't supported (assert in backbone.py) -> just run FULL 5-scale
- deformable attention soft-mixes all levels per query anyway — no need to hand-assign
  object sizes to levels

**Spec:** `return_interm_indices=[0,1,2,3]`, `num_feature_levels=5` (pure config switch).
Cost: ~4x encoder tokens -> ~2-3x train time/memory; likely batch 1 + grad accum on A100 40GB.

**Hardware / feasibility (galvani, verified via scontrol 2026-06-12):**
Scale anchor: 4-scale runs 1.13 s/it, 9.5 min/ep, batch 2, 1x A100 40GB. Our 5-scale token
count (~44k @ 512x1024) is BELOW the DINO paper's 5-scale swin-L on COCO -> known-feasible
territory. Plan, simplest first:
1. `--amp` (exists in main.py, currently UNUSED) — ~halves activation memory + faster;
   good chance 5-scale @ batch 2 then fits one 40GB A100 as-is (use_checkpoint already on).
   Caveat: fp16 + deformable attention occasionally needs loss-scale babysitting.
2. still OOM -> batch_size 1 (no grad-accum in main.py; maybe nudge lr down).
3. too slow -> multi-GPU DDP on ONE node: a100-galvani nodes have 8-9 GPUs and multi-GPU
   single-node jobs ARE allowed (web guide's "min 2 nodes" is wrong; scontrol: MaxNodes=32).
   ~linear speedup; VERIFY first that SimulationDataset gets a per-rank sampler/seed
   (stock DDP path distributes coco, not our sim loader). Fairshare scales with GPUs blocked.
4. capacity trick: a100-preemptable-galvani (PreemptMode=REQUEUE) — auto-resume loses <=1
   epoch per preemption; good for long experimental runs. a100-fat = big HOST RAM, same
   40GB GPUs -> not the enabler.
5. comfort option: ferranti cluster (15 nodes x 8 H100 80GB) — 2x VRAM, ~2-3x speed; 5-scale
   would run like current 4-scale. Needs: access check (separate cluster) + env rebuild.
First step when running it: 1-epoch smoke test (5-scale, --amp, batch 2, one A100 —
preemptable is fine) and read real memory + s/it from the log; that picks option 1/2/3.
Worst case (batch 1, single GPU): ~25-30 min/ep -> 360 ep over 2-3 auto-resumed 72h sessions.

**SSL tie-in (why this belongs here):** stage 1 is already SimMIM-pretrained (and the pixel
reconstruction objective forces fine detail through it) — it's the most real-data-flavored
layer (noise statistics, true peak profiles = what simulation reproduces worst). Feeding it
to the detector plugs into where real-data SSL has the most to offer; expect the SSL-vs-
baseline gap to widen under 5-scale. Caveat: out_indices=(0,1,2,3) adds a new `norm0` not in
the SSL export -> randomly initialized via strict=False (harmless). For a clean A/B also run
a from-scratch 5-scale baseline (5-scale helps non-SSL models too).

## Explicitly rejected (with reasons — don't revisit without new evidence)
- more SSL epochs: val flat from ~ep40; ep77 ≈ ep200 (round-1 evidence)
- heavier decoder: lowers recon loss, worsens features (SimMIM ablation)
- lower mask ratio: easier task -> weaker features
- standard contrastive/DINO pretraining: aug-menu too limited by polar physics +
  false negatives from corpus redundancy
- chi-roll / rotation aug: physically invalid for wedge frames / q-radial geometry
