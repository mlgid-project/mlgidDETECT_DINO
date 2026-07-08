# Co-DETR / Co-DINO Adoption — Feasibility & Investigation

Target: adopting **Co-DETR** ("DETRs with Collaborative Hybrid Assignments Training", arXiv:2211.12860,
ICCV 2023) onto `mlgidDETECT_DINO` (DINO 4-scale, Swin-L 48×6 elongated-window, single-channel `in_chans=1`,
2-class ring/segment, ONNX-exported GIWAXS/GID diffraction-peak detector on 512×1024 polar images).

Scope: assess whether Co-DETR's collaborative auxiliary heads raise our dominant remaining failure —
**low SENSITIVITY (recall) to faint and high-q peaks** (diagnostic: faint vis=1 recall ≈ 0.33, high-q
q682–1024 recall ≈ 0.44), which caps organic-set AP. This is a *recall ceiling*, not a precision or
detail-routing problem. Every claim below is grounded in either the fetched paper or a `file:line` in this repo.
Line numbers were read on branch `Semi-DETR` at write time — reverify before editing, they drift.

> **Scope guard:** the Co-DETR mechanism we recommend is **TRAINING-ONLY** — auxiliary heads are attached to
> encoder features during training and *discarded at inference* (the paper's central selling point: "in
> inference, these auxiliary heads are discarded and thus our method introduces no additional parameters and
> computational cost"). Unlike CCTM (which sits on the traced decoder path, `deformable_transformer.py:370-372`),
> the aux heads never touch the exported forward. See §5.

---

## 1. Verdict up top

**Verdict: ADOPT-IN-PART — build a scoped MVP, then gate hard. Greenlight ONE encoder auxiliary head with dense
one-to-many supervision (encoder-supervision only), and DECLINE the full K-head zoo and the "customized positive
queries" branch unless the MVP moves faint/high-q recall specifically.**

Why this is the best-aligned of the three DETR add-ons we have investigated:

- Co-DETR's **dominant driver is the auxiliary head itself, not the positive queries.** On the paper's own
  ablation (Deformable-DETR R50, 12 ep): baseline 37.1 AP → **aux-heads-only 41.6 (+4.5)** → positive-queries-only
  40.5 (+3.4) → both 42.3 (+5.2). The bigger single lever (+4.5) is precisely **dense one-to-many supervision
  injected directly into the encoder/backbone features** — which is the exact mechanism we hypothesised for
  faint-peak sensitivity. This is a *better* thesis-match than Cross-DINO's CCTM (feature *refinement*, decl.
  in `docs/CROSS_DINO_INVESTIGATION.md`) or Boost Loss (loss *re-weighting*).
- It is **strictly training-only and ONNX-perfectly-safe** — safer than CCTM. The aux heads are gated out at
  eval/export by two independent guards (`self.training` + the export wrapper reading only
  `pred_logits`/`pred_boxes`, `export_onnx.py:26-27`). The deployed graph is **byte-identical** to today's. §5.

Why "in part" and why the hard gate (the adversarial read, §7):

- **On a DINO base the marginal gain collapses.** Co-DETR adds **+4.5 AP on plain Deformable-DETR** but only
  **+1.0–1.8 AP on DINO** (Table 3: DINO-Def-DETR R50 12ep 49.4→51.2 with K=2; Swin-L 36ep 58.5→59.5). The
  reason is structural: DINO *already* supplies dense-ish encoder supervision that plain Deformable-DETR lacks —
  two-stage query-selection loss (`interm_outputs`, `dino.py:296-301`), 100 denoising queries (`use_dn=True`,
  `config:113-114`), and 6 auxiliary decoder losses (`aux_loss=True`, `config:90`). **Our config has all three
  ON.** Co-DETR partly re-sells headroom DINO already claimed.
- **"Small object" ≠ "faint object."** The reported small-object win (AP_S 41.5→45.1, +3.6 on Co-DETR-DINO++
  Swin-L) is about objects with *few pixels*, not *low SNR*. A faint high-q arc can be large. Co-DETR was never
  evaluated on a low-contrast recall problem; the mapping to our ceiling is *plausible but unproven* (§7).
- **Elongated-box fragility.** ATSS assigns positives by anchor-IoU + center distance; our peaks are extreme
  aspect-ratio arcs. Anchor IoU can be near-zero even at the correct center for a thin segment → few/no positives
  assigned → the dense-supervision benefit evaporates for exactly the elongated faint segments we care about
  (same failure family as the `sqrt(area)` CS-label caveat in `docs/CROSS_DINO_INVESTIGATION.md:122`). This is a
  strong reason to prefer a **center-sampling FCOS-style assigner over ATSS** for our MVP (§4, §6).

**Payoff-vs-effort call.** Expected payoff on organic AP is **modest** (paper predicts +1–1.8 AP on a DINO base,
and the faint-specific benefit is speculative). But the MVP is (a) the most on-thesis lever left, (b) **zero
inference/ONNX cost**, (c) trainable as a warm-start fine-tune, and (d) cheaply falsifiable via the existing
faint/high-q recall diagnostic. That asymmetry justifies building the single-head MVP and gating hard. The full
K-head + positive-query machinery (the invasive, decoder-surgery half) is **not** worth it up front.

**Effort:** MVP ≈ medium (one new head module + one assigner + one criterion + build/config wiring; ~250–350 LOC;
no decoder or DN surgery). Full method ≈ high (adds customized-positive-query injection into the decoder with a
blocking attention mask — touches the same query-assembly code as DN, `deformable_transformer.py:390-416`).

---

## 2. Co-DETR method breakdown (grounded in the fetched paper)

Sources: abstract (`arxiv.org/abs/2211.12860`) + HTML body (`ar5iv.labs.arxiv.org/html/2211.12860`), two
independent fetches. Component names, placements, ablation numbers, and hyper-parameters were consistent across
fetches. **Caveat:** the loss equations below are transcribed from the HTML render — verify against the PDF
before implementing the exact weighting. Unlike Cross-DINO, **official code exists** (Sense-X/Co-DETR, mmdet-based)
— useful as an assigner reference, but *not* directly liftable into this DETR fork (§8).

### The idea in one paragraph

A DETR decoder uses **one-to-one** Hungarian matching: each GT box supervises exactly **one** query. That makes
the *encoder* feature learning sparse — most spatial locations receive gradient only indirectly (via the tokens
that query-selection happens to pick, `deformable_transformer.py:397-398`). Co-DETR bolts **K conventional
detection heads** (ATSS, Faster-R-CNN/RetinaNet-style), each with its own **one-to-many** label assignment, onto
the encoder's multi-scale feature pyramid. Now *every* location near a GT box gets a direct classification +
regression gradient. The heads are **auxiliary and training-only** — thrown away at inference.

### Components and where they sit

| # | Component | Pipeline location | Nature | Keep for our MVP? |
|---|-----------|-------------------|--------|-------------------|
| 1 | **K collaborative aux heads** (ATSS, Faster-RCNN) | Attached to the **encoder output**, reshaped to a 2-D feature pyramid `{F₁…F_J}` | Conventional dense detector heads, each with a **one-to-many** assigner | **YES — this is the whole point.** MVP = **K=1**. |
| 2 | **Customized positive queries** | Aux heads' positive coordinates `B_i^pos` → **extra decoder queries**, supervised one-to-one *without* Hungarian matching | Query injection + a blocking attention mask in the decoder | **NO for MVP** (invasive; targets decoder matching stability, not encoder faint-peak representation) |
| 3 | **Collaborative loss** | Sums the DETR one-to-one loss + K aux-head losses + K encoder-supervision losses | Loss assembly | Adapted (§4c) |

### Feature-pyramid construction (item 1)

The encoder emits one token sequence `F ∈ (bs, ΣHW, C)`; Co-DETR reshapes it back to multi-scale 2-D maps
`{F₁…F_J}` (bilinear up/downsample + 3×3 conv to hit the standard strides). **In our repo this reshape is
trivial — the per-level slices are already in scope** (§3.1). Each `A_i` then runs its assigner on `F_i`:

```
P_i^pos, B_i^pos, P_i^neg = A_i( head_i(F), G )        # A_i = ATSS / Faster-RCNN assignment; G = GT boxes
L_i^enc = classification+regression loss on head_i's dense predictions vs A_i's assignment
```

### One-to-many assignment (ATSS, the paper's primary aux head)

ATSS per GT box: (1) generate anchors per pyramid level; (2) pick the top-k anchors by center distance per level
as candidates; (3) set an adaptive IoU threshold = mean+std of candidate IoUs; (4) positives = candidates with
IoU ≥ threshold **and** center inside the box. Faster-RCNN aux head uses a fixed IoU>0.5 rule. Both assign
**many** anchors per GT (vs DETR's one query per GT).

### Total loss (transcribed — verify against PDF)

```
L^global = Σ_l ( L̃_l^dec  +  λ₁ Σ_{i=1..K} L_{i,l}^dec  +  λ₂ L^enc ),      L^enc = Σ_{i=1..K} L_i^enc
```
with **λ₁ = 1.0, λ₂ = 2.0**. `L̃^dec` = the standard DETR one-to-one loss (our current `SetCriterion`);
`L_{i}^dec` = the customized-positive-query decoder loss (item 2, dropped in our MVP); `L_i^enc` = the aux head's
dense one-to-many loss on the encoder pyramid (item 1, **the piece we keep**).

### Hyper-parameters and cost

- **K = 1 or 2** (paper default; ATSS for K=1, ATSS+Faster-RCNN for K=2). **DINO variants use K=2.**
- **K-ablation (Table 7):** K=0 → 47.1; **K=1 → 48.7 (+1.6)**; K=2 → 49.5 (+2.4); K=3 → 49.5 (saturates);
  K=6 → 48.9 (degrades, head conflicts). **K=1 captures ~2/3 of the K=2 gain** — the justification for a single-head MVP.
- **Training overhead (Table 7, Deformable-DETR++):** K=2 costs +~1,600 MB and 70→120 GPU-h (~**1.7×** wall-clock);
  K=1 is roughly half the *added* cost (~1.35×).
- **Schedule:** 12/36/50 ep depending on backbone.
- **Inference:** aux heads discarded → **zero** added params/FLOPs on the deployed model.

### Which component drives the gain (critical for us)

The **auxiliary head (dense encoder supervision)** is the larger single lever (+4.5 alone vs +3.4 for positive
queries on Deformable-DETR). The paper attributes the aux head to "encoder discrimination via dense spatial
supervision" and the positive queries to "decoder attention stability / reduced Hungarian-matching instability."
**Our recall ceiling is an encoder-representation problem (faint peaks aren't discriminatively encoded), not a
decoder-matching-instability problem** → we keep the aux head and drop the positive queries. See §7.

---

## 3. Exact code insertion map (`file:line`)

### 3.1 Encoder output → 2-D feature pyramids (the primary insertion point)

The encoder returns `memory` and everything needed to un-flatten it is already in scope:

- `models/dino/deformable_transformer.py:351-360` — **encoder call** → `memory` shape `(bs, ΣHW, 256)`. This is `F`.
- `models/dino/deformable_transformer.py:341` — `spatial_shapes` `(nlevel, 2)` (the `(H_l, W_l)` per level).
- `models/dino/deformable_transformer.py:342` — `level_start_index` (cumulative token offsets per level).
- `models/dino/deformable_transformer.py:323-338` — the **forward flatten** that we invert:
  `src.flatten(2).transpose(1,2)` at `:328` maps `(bs,C,H,W)→(bs,HW,C)`; the inverse is a `transpose+reshape`.
- `models/dino/deformable_transformer.py:370-372` — **CCTM block** (default-off, `use_cctm`). If CCTM is on it
  rewrites `memory` *before* our aux heads see it — compose **after** line 372 so the aux head reads the
  post-CCTM memory (correct: aux supervision should see whatever feeds the decoder).
- `models/dino/deformable_transformer.py:374-416` — two-stage query selection (`gen_encoder_output_proposals` →
  top-k → decoder `tgt`/`refpoint`). Our aux-head branch sits **parallel** to this, reading the same `memory`.

**Prior art for the reshape (cite this — it is the exact loop we mirror):** `util/utils.py:30-49`
(`gen_encoder_output_proposals`) already walks `spatial_shapes` with a running `_cur` cursor and does
`memory_padding_mask[:, _cur:_cur+H_*W_].view(N_, H_, W_, 1)`. The aux-head split is the identical loop, kept as
`(bs, C, H, W)`:

```python
# NEW: models/dino/co_heads.py — reshape memory back to a 2-D pyramid for the aux head(s)
def memory_to_pyramid(memory, spatial_shapes, level_start_index):
    # memory: (bs, ΣHW, C); returns List[(bs, C, H_l, W_l)] — inverse of deformable_transformer.py:328
    bs, _, C = memory.shape
    feats = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        start = int(level_start_index[lvl]); n = int(H) * int(W)
        f = memory[:, start:start+n].transpose(1, 2).reshape(bs, C, int(H), int(W))
        feats.append(f)
    return feats
```

For a 512×1024 input our four levels are strides 8/16/32/64 → `(64,128),(32,64),(16,32),(8,16)`,
ΣHW = 8192+2048+512+128 = **10,880 tokens** — matching `num_feature_levels=4` (`config:70`). The mask needed to
zero no-data regions per level is `mask_flatten` (`deformable_transformer.py:339`), sliced the same way.

**Where the aux head is called.** Insert a training-only block in the transformer forward right after CCTM:

```python
# models/dino/deformable_transformer.py, immediately after line 372
self.co_head_pyramid = None
if self.co_heads is not None and self.training:          # training-only guard (see §5)
    self.co_head_pyramid = memory_to_pyramid(memory, spatial_shapes, level_start_index)
```

and surface it up to `DINO.forward` so the loss can be assembled there (cleaner than computing loss inside the
transformer). Two clean routes: (a) stash `self.co_head_pyramid` on the transformer and read it in `DINO.forward`
(mirrors how `hs_enc`/`init_box_proposal` are returned, `deformable_transformer.py` decoder return); or
(b) extend the transformer's return tuple. Route (a) is least invasive.

### 3.2 Model forward → aux-head predictions in the output dict

- `models/dino/dino.py:270` — `self.transformer(...)` call; add the pyramid to its outputs (Route (a) above).
- `models/dino/dino.py:290-320` — the `out` dict is assembled here (`pred_logits`, `pred_boxes`, `aux_outputs`,
  `interm_outputs`, `enc_outputs`, `dn_meta`). Add, **only when training and enabled**:

```python
# models/dino/dino.py, near the enc_outputs block (~:316)
if self.training and getattr(self, 'co_heads', None) is not None and self.transformer.co_head_pyramid is not None:
    out['co_head_outputs'] = self.co_heads(self.transformer.co_head_pyramid)  # per-level dense cls/box preds
```

`co_head_outputs` is invisible to inference because (i) it is `self.training`-gated and (ii) the export wrapper
reads only `pred_logits`/`pred_boxes` (`export_onnx.py:26-27`). Exactly how `aux_outputs`/`interm_outputs`/
`enc_outputs` are already dead-code at export.

### 3.3 Loss assembly + flag gating (mirror `use_boost_loss`)

The aux-head loss does **not** fit `SetCriterion` (that class is query-based / Hungarian; the aux head is dense /
anchor-based). Add a **separate small criterion** and add its scalars into the existing weighted sum.

- `models/dino/dino.py:713-854` — `build_dino`. The `weight_dict` is assembled at `:786-825`
  (`loss_ce/bbox/giou`, then `_dn` `:793-795`, `_{aux}` `:803-807`, `_interm` `:809-825`). Add co-head keys and a
  coefficient here so the engine's weighted sum picks them up unchanged:

```python
# models/dino/dino.py, after weight_dict is built (~:826)
if getattr(args, 'use_co_heads', False):
    co_coef = getattr(args, 'co_loss_coef', 1.0)        # ~ paper λ₂ = 2.0 for encoder supervision
    weight_dict['loss_co_cls']  = co_coef * args.cls_loss_coef
    weight_dict['loss_co_bbox'] = co_coef * args.bbox_loss_coef
    weight_dict['loss_co_giou'] = co_coef * args.giou_loss_coef
```

- `models/dino/dino.py:830-846` — `SetCriterion` is built and training-only flags are attached via the
  **`getattr(args, 'flag', default)` pattern** (`use_boost_loss` at `:842`, `matcher_o2m` at `:836`). Attach the
  co-head module + its criterion the same way:

```python
# models/dino/dino.py, alongside the use_boost_loss block (~:842)
if getattr(args, 'use_co_heads', False):
    from .co_heads import CoHeads, CoCriterion
    model.co_heads = CoHeads(d_model=args.hidden_dim, num_classes=num_classes,
                             num_levels=args.num_feature_levels, k=getattr(args, 'co_num_heads', 1),
                             assigner=getattr(args, 'co_assigner', 'fcos'))
    co_criterion = CoCriterion(num_classes, assigner=getattr(args, 'co_assigner', 'fcos'))
else:
    model.co_heads = None
    co_criterion = None
```

- **Where the co-head loss is summed.** `engine.py:52-55` computes `criterion(outputs, targets)` then
  `sum(loss[k]*weight_dict[k])`. Add a parallel call `co_criterion(out['co_head_outputs'], targets)` (only when
  `out` contains the key) returning `{'loss_co_cls','loss_co_bbox','loss_co_giou'}`, and fold those into the same
  weighted sum using the `weight_dict` keys added above. This keeps normalisation/AMP/backward identical to today.

Default-off gating everywhere is `getattr(args, 'use_co_heads', False)` — identical to the `use_cctm`
(`deformable_transformer.py:1122`) and `use_boost_loss` (`dino.py:842`) precedents. Absent the flag, the build is
byte-identical to the current model.

### 3.4 Optimizer / param groups

- `util/get_param_dicts.py:23-51` — the `default` branch already special-cases a warm-started grafted module: it
  splits out `cctm.*` into its own group at `args.lr * lr_cctm_mult` (`:31-42`) so a fresh module can learn faster
  than the gently-fine-tuned body. **The co-heads are the same situation** (random-init heads grafted onto a
  converged detector) and want the same treatment. Extend the existing split with a `co_heads.*` group:

```python
# util/get_param_dicts.py, generalize the lr_cctm_mult split (~:31-42)
{"params": [p for n,p in model.named_parameters()
            if "co_heads" in n and p.requires_grad], "lr": args.lr * getattr(args, 'lr_cohead_mult', 10.0)},
```

Rationale mirrors the CCTM note at `get_param_dicts.py:24-30`: at the body's 1e-5 fine-tune rate (`config:15`) a
fresh head "can't gain traction and a null result would be a false negative." A ~10× head LR lets the aux head
earn its keep while backbone/encoder stay anchored. Absent `lr_cohead_mult` → falls back to the main non-backbone
group (still correct).

### 3.5 Build / config wiring

- `main.py:174-178` (`build_model_main`) → `build_dino` (`dino.py:713`). Arbitrary config keys reach `args` via
  `SLConfig.fromfile` + `cfg.merge_from_dict(args.options)` (`main.py:239-241`), consumed with
  `getattr(args, 'flag', default)`. No parser changes needed — this is exactly how `use_cctm`, `use_boost_loss`,
  `lr_cctm_mult`, and the semi flags (`main.py:395`) all reach the model.
- **New config `config/DINO/DINO_4scale_swin_codino.py`** — `_base_ = ['DINO_4scale_swin.py']`, add:

```python
_base_ = ['DINO_4scale_swin.py']
use_co_heads   = True
co_num_heads   = 1          # K=1 MVP (paper K=1 gives ~2/3 of K=2; ATSS is head_0)
co_assigner    = 'fcos'     # 'fcos' (center-sampling, robust to elongated boxes) | 'atss' (paper-faithful)
co_loss_coef   = 2.0        # paper λ₂ for encoder supervision
lr_cohead_mult = 10.0       # fresh heads on a converged warm-start (mirror lr_cctm_mult)
# use_cctm / use_boost_loss stay at their base defaults (off) — orthogonal, can be combined later
```

Launch mirrors the existing sbatch pattern (auto-resume from `checkpoint.pth`, warm-start via
`--pretrain_model_path`) exactly as `docs/SEMI_DETR_INTEGRATION.md` §7 and the SSL scripts show.

---

## 4. MVP scope and phased plan

**MVP = ONE encoder auxiliary head, FCOS-style center-sampling assigner, encoder-supervision only, NO customized
positive queries.** This is the smallest *faithful* version: it keeps the component that carries the larger gain
and maps to our thesis (dense encoder supervision), and it avoids all decoder/DN surgery. Gating metric is
**organic-set AP**; `41` is the easy control; eval runs every 2 epochs (`config:5-10`). All phases fine-tune from
the current best checkpoint (warm start), not from scratch.

### Assigner choice for the MVP (important)

Recommend **FCOS-style center sampling over ATSS** for Phase 0:
- FCOS positive rule = "location whose center falls inside a GT box (within a center-sampling radius)". No anchor
  generation, no per-level anchor-IoU, no adaptive threshold → ~half the code and **no aspect-ratio fragility**.
- Our peaks are extreme-aspect arcs; ATSS anchor-IoU can be ~0 at the correct center for a thin segment, starving
  exactly the elongated faint boxes of positives (§1, §7). FCOS center-sampling assigns positives regardless of
  aspect ratio → more robust dense supervision for our geometry.
- ATSS remains the **paper-faithful** upgrade (Phase 2) if FCOS is promising but we want to match the paper.

### Phase 0 — single FCOS aux head, encoder-only (the real test)
- **Build:** `models/dino/co_heads.py` (`memory_to_pyramid` §3.1 + a shared 3×3-conv `CoHeads` cls/box head over
  the 4 pyramid levels + `CoCriterion` with an FCOS center-sampling assigner + focal-cls / GIoU-box loss);
  the `deformable_transformer.py:372` reshape hook; the `dino.py:290-320` output-dict entry; the
  `dino.py:826/842` weight_dict + build wiring; the `get_param_dicts.py:42` LR group; the new config.
- **Run:** fine-tune from current best; A/B vs the matched baseline on `41` + `organic` every 2 ep.
- **Read-out (this is the whole experiment):** don't gate on AP alone — re-run the **faint(vis=1) and
  high-q(q682–1024) recall diagnostic** that produced the 0.33 / 0.44 baseline numbers. Success = organic AP up
  **and** faint/high-q recall up. AP up but faint/high-q recall flat = "refining easy peaks" (the CCTM-null
  pattern) → decline (§6, §7).
- **ONNX check:** export the Phase-0 checkpoint via `backbone_curation/export_onnx_ensemble.py` and diff the graph
  against a baseline export — must be identical (aux heads are `self.training`-gated, §5). This closes the deploy
  question on day one and is essentially free.

### Phase 1 — assigner / weight tuning (only if Phase 0 is non-negative)
- Sweep `co_loss_coef ∈ {1,2,4}`, center-sampling radius, `lr_cohead_mult ∈ {5,10,20}`, and the pyramid levels
  the head attaches to (all 4 vs finest-2 — faint high-q peaks live on the finest levels). Re-gate on organic AP +
  faint/high-q recall.

### Phase 2 — K=2 and/or ATSS (only if Phase 0/1 clearly help)
- Add a second aux head (`co_num_heads=2`, ATSS as head_1) — paper's DINO config. Expect a *small* increment
  (K=2 vs K=1 ≈ +0.8 AP on Deformable-DETR; less on a DINO base). Swap FCOS→ATSS if we want paper fidelity and the
  elongated-box concern turns out not to bite.

### Phase 3 (optional, full method) — customized positive queries
- Only if encoder-supervision alone helps and we want the last increment. Inject aux-head positive coordinates as
  extra decoder queries with a blocking attention mask, supervised one-to-one — touching the same query-assembly
  region as DN (`deformable_transformer.py:390-416`) and the decoder attention mask. Highest effort, least aligned
  with our (encoder-representation) ceiling. Default expectation: **skip.**

**MVP = Phase 0.** **Full Co-DETR = through Phase 3.**

---

## 5. ONNX / deploy-safety

This is Co-DETR's structural advantage over CCTM, and it is airtight here.

| Component | Computed at inference/export? | Traced into ONNX? | Verdict |
|---|---|---|---|
| Encoder aux head(s) (MVP) | **No** — guarded by `self.training` (§3.1, §3.2) | **No** | **Zero risk** (strictly training-only) |
| `co_head_outputs` dict key | Not created at eval | **No** — wrapper reads only `pred_logits`/`pred_boxes` | **Zero risk** |
| Customized positive queries (Phase 3) | Extra queries only added when training | **No** | Zero risk (but code-invasive) |

Two independent guarantees the exported graph is unchanged:

1. **`self.training` gate.** The aux-head reshape (`deformable_transformer.py:372` hook) and the head call
   (`dino.py:~316`) are both inside `if self.training and ...`. Export runs `model.eval()`
   (`export_onnx.py:31`, `wrapper.eval()` `:37`) → `self.training == False` → the aux-head code is **never
   entered** during the trace. Nothing to fold, nothing to strip.
2. **Output whitelist.** `DINOOnnxWrapper.forward` returns only `out["pred_logits"], out["pred_boxes"]`
   (`export_onnx.py:26-27`). Even if `co_head_outputs` were produced, it is not in the wrapper's output tuple, so
   the tracer cannot reach it — identical to how `aux_outputs`/`interm_outputs`/`enc_outputs` (`dino.py:291-318`)
   are already invisible at export.

The MSDeformAttn CPU/ONNX shim (`backbone_curation/export_onnx_ensemble.py:23-35`, rebinding to the pure-PyTorch
`grid_sample` core) and the swin gradient-checkpoint gate (`swin_transformer.py:401`,
`not torch.onnx.is_in_onnx_export()`) are untouched — the aux heads never interact with either. **The deployed
`DINO.forward` inference path, pre/post-processing, and the exported `.onnx` are all byte-for-byte unchanged.**
Co-DINO changes only the weights inside `checkpoint.pth`, exactly like Semi-DETR (`docs/SEMI_DETR_INTEGRATION.md`).

---

## 6. Adversarial: will denser one-to-many supervision actually raise FAINT/high-q recall?

This is the section that decides the project. Be honest.

### The mechanism that *could* work (why we're building it)
In DINO, the encoder receives direct gradient mainly through the two-stage query-selection loss: it scores every
token, keeps the **top-900** (`deformable_transformer.py:397-398`, `num_queries=900` `config:57`), and supervises
those via `interm_outputs` (`dino.py:296-301`). A **faint peak whose token never makes the top-900 cut gets almost
no direct encoder gradient** — the encoder is never told "there is a peak here," so it never learns to represent
it discriminatively, so it keeps missing it: a self-reinforcing recall hole. An FCOS/ATSS aux head assigns **every
location inside every GT box** as positive, **independent of query selection**. A faint high-q arc that
query-selection ignores still gets a direct classification+regression gradient on its encoder tokens. That is a
*specific, plausible* lever for the exact 0.33/0.44 recall holes — arguably the most direct one of the three DETR
add-ons we've studied.

### Why it might NOT (the failure modes to watch)
1. **DINO already partly does this.** Two-stage interm loss + 100 DN queries + 6 aux decoder losses (all ON in our
   config) already densify supervision far beyond plain Deformable-DETR — which is exactly why Co-DETR's gain
   shrinks from +4.5 to +1.0–1.8 on a DINO base (§2). The headroom the aux head exploits is partly pre-claimed.
2. **The assigner is not faint-aware.** FCOS/ATSS assign positives by *geometry* (center-in-box / anchor-IoU),
   treating a faint peak and a bright peak with the same box identically. This is *better* than one-to-one (where a
   faint peak competes for a single query and loses), but it is **not** a faint-specific booster — it is
   uniform-density supervision. If faint peaks are missed because their GT boxes are *also* missing/mislabeled
   (the label-completeness issue noted in `docs/SEMI_DETR_INTEGRATION.md:36`), denser supervision on the labels we
   *have* won't create positives for the ones we *don't*.
3. **The CCTM precedent.** CCTM was declined because it *refined easy peaks* rather than surfacing faint ones. The
   aux head can fail the same way: overall AP creeps up (bright peaks get crisper dense supervision) while
   faint/high-q recall stays pinned. The mechanism is more on-thesis than CCTM's, but the null-result *shape* is
   identical and we must test for it, not assume past it.
4. **Elongated-box starvation (ATSS only).** As in §1 — thin arcs can get zero anchor-IoU positives, so ATSS
   supervises the *fat* peaks and starves the *thin faint* ones. FCOS center-sampling avoids this; it is why the
   MVP uses FCOS.

### How we know early (cheap falsification)
- **Gate on the recall diagnostic, not AP.** Re-run the faint(vis=1) / high-q(q682–1024) recall probe that gave
  0.33 / 0.44. If, after the aux head warms up (~the first few eval intervals), those two numbers are flat while
  AP nudges up, it's a CCTM-style easy-peak refinement → **decline**.
- **Watch encoder proposal recall directly.** Log the two-stage `interm_outputs` recall on faint GT
  (`dino.py:296-301`): the aux head should *raise the rate at which faint peaks enter the top-900 proposals* if the
  mechanism is working. That is the leading indicator, upstream of AP.
- **Per-class split.** Segment=0 is the low-recall class; track its recall separately (as
  `docs/SEMI_DETR_INTEGRATION.md:326` does for pseudo-labels).

---

## 7. Stop rule

1. **Phase 0 is the gate.** If a single FCOS aux head, fine-tuned to convergence (~30–40 ep) from the current best
   checkpoint, does **not** raise organic AP **and** does **not** raise faint(vis=1)/high-q recall vs the matched
   baseline → **decline Co-DINO.** Do **not** proceed to K=2 or the positive-query branch. The paper's own
   K-ablation (K=1 already ≈ 2/3 of K=2) means if K=1 is null, K=2 will not rescue it; and the positive-query
   branch targets decoder matching stability, *not* the encoder faint-peak representation that is our ceiling — it
   is even less aligned.
2. **AP-up-but-recall-flat = decline, not proceed.** Treat an AP bump with flat faint/high-q recall as the
   CCTM-null pattern (easy-peak refinement), not a partial win. Our ceiling is recall; refinement of bright peaks
   is not progress on it.
3. **Elongated-box escape hatch.** If FCOS Phase 0 is flat, do **not** conclude "Co-DETR fails" without checking
   whether the aux head assigned meaningful positives on the thin/faint segment boxes (log positive counts per GT
   aspect-ratio bin). A dense head that assigned ~no positives to elongated faint boxes was never actually tested —
   fix the assigner (radius / level selection) once before declaring the mechanism dead.

---

## 8. Risks / unknowns (explicit)

1. **Reconstruction uncertainty (lower than Cross-DINO, not zero).** Official code exists (Sense-X/Co-DETR), so
   the *assigner logic* (ATSS/FCOS) is standard and well-specified — a real advantage over Cross-DINO's
   no-public-code situation. But it is **mmdet-framework** code, not liftable into this DETR fork: we reimplement
   the head + assigner + loss (~250–350 LOC) against the equations, using mmdet's `AtssAssigner`/FCOS as a
   reference. The exact λ₁/λ₂ weighting and the customized-positive-query detail should be checked against the PDF
   before Phase 3.
2. **Modest expected magnitude on a DINO base.** +1.0–1.8 AP is the paper's DINO number, and the faint-specific
   benefit is unproven (§6). Realistic upside is a *small* organic-AP gain; the bet is that a *small AP gain
   concentrated in faint/high-q recall* is disproportionately valuable to us because that is precisely our ceiling.
3. **Assigner complexity vs simplicity trade.** ATSS = anchor generation + per-level IoU + center-distance top-k +
   adaptive threshold (~150 LOC, aspect-ratio-fragile for our arcs). FCOS center-sampling = ~half that and robust.
   We deliberately start with the *simpler* assigner (FCOS) for both effort and geometry reasons; this is a
   controlled deviation from the paper, disclosed.
4. **Training-cost multiplier.** The aux head adds a full-pyramid forward+backward + the (CPU-ish) assignment each
   step. Paper K=2 was ~1.7× GPU-h; K=1 ≈ ~1.35×, +1–2 GB. On our single-GPU 512×1024 Swin-L (`use_checkpoint=True`
   `config:41`, `--amp`) that is manageable but not free — budget ~+35% wall-clock for Phase 0. Attaching the head
   to only the finest 2 levels (where faint high-q peaks live) both cuts cost and focuses supervision.
5. **Label-completeness ceiling (shared with Semi-DETR).** If faint peaks are missed because their GT is *absent*
   (not merely under-supervised), a denser head over the *present* labels cannot invent the missing positives.
   Co-DINO and Semi-DETR are complementary here: Semi-DETR *adds* labels (real pseudo-labels,
   `docs/SEMI_DETR_INTEGRATION.md`); Co-DINO *supervises the labels we have more densely*. Neither alone fixes
   missing GT.
6. **Interaction with in-flight branches.** CCTM (`use_cctm`) and Boost Loss (`use_boost_loss`) are orthogonal
   training-only flags (default-off); Co-heads compose with both (the aux head reads post-CCTM memory, §3.1). Keep
   them off for the clean Phase-0 A/B, then combine only if independently positive — same discipline as
   `docs/CROSS_DINO_INVESTIGATION.md` §7.

---

## 9. References

- **Co-DETR paper** — Zong, Song, Liu, "DETRs with Collaborative Hybrid Assignments Training," ICCV 2023,
  arXiv:2211.12860 — https://arxiv.org/abs/2211.12860 (HTML: https://ar5iv.labs.arxiv.org/html/2211.12860;
  PDF: https://arxiv.org/pdf/2211.12860 — verify λ₁/λ₂ and the positive-query loss here before Phase 3).
  Verified: K=1/2 aux heads (ATSS, Faster-RCNN); one-to-many encoder supervision; customized positive queries;
  aux heads discarded at inference (zero cost); λ₁=1.0, λ₂=2.0; Deformable-DETR R50 12ep ablation
  37.1→41.6(aux)/40.5(pos)/42.3(both); DINO R50 49.4→51.2 / Swin-L 58.5→59.5 (K=2); K-ablation K1+1.6/K2+2.4/K3
  sat/K6 degrades; AP_S 41.5→45.1; K=2 overhead +~1.6GB, 70→120 GPU-h.
- **Official code (assigner reference, not liftable):** Sense-X/Co-DETR (mmdet-based) — https://github.com/Sense-X/Co-DETR
- **ATSS** — Zhang et al., "Bridging the Gap Between Anchor-based and Anchor-free Detection via ATSS," CVPR 2020,
  arXiv:1912.02424.
- **FCOS** (center-sampling assigner recommended for the MVP) — Tian et al., ICCV 2019, arXiv:1904.01355.
- **DINO** (base detector) — Zhang et al., ICLR 2023, arXiv:2203.03605.
- **Deformable DETR** (MSDeformAttn) — Zhu et al., arXiv:2010.04159.
- Prior sibling investigations (style/rigor + shared constraints): `docs/CROSS_DINO_INVESTIGATION.md`,
  `docs/SEMI_DETR_INTEGRATION.md`.

### Repo anchor points (for the implementer)
- **Primary insertion — memory→pyramid + aux-head hook:** `models/dino/deformable_transformer.py:351-360` (memory),
  `:341-342` (spatial_shapes / level_start_index), `:370-372` (compose after CCTM); reshape prior-art
  `util/utils.py:30-49`.
- **Output-dict entry (training-only):** `models/dino/dino.py:290-320` (near the `enc_outputs` block `:316`).
- **Build / weight_dict / flag gating:** `models/dino/dino.py:786-825` (weight_dict), `:830-846` (criterion build +
  `getattr` flag pattern, `use_boost_loss` precedent `:842`).
- **Loss summation:** `engine.py:52-55` (add the co-criterion call to the existing weighted sum).
- **Param group:** `util/get_param_dicts.py:31-42` (generalize the `lr_cctm_mult` split to `co_heads.*`).
- **Config merge / build:** `main.py:174-178` (`build_model_main`), `main.py:239-241` (SLConfig merge → arbitrary
  keys reach `args`).
- **ONNX safety:** `export_onnx.py:18-27` (output whitelist), `:31-37` (eval-mode export);
  `backbone_curation/export_onnx_ensemble.py:23-35, 52` (MSDeformAttn shim + `export=True`);
  `models/dino/swin_transformer.py:401` (checkpoint gate).
- **Eval A/B sets:** `config/DINO/DINO_4scale_swin.py:5-10` (`41` + `organic`, every 2 ep).
- **New files:** `models/dino/co_heads.py` (heads + assigner + `CoCriterion`),
  `config/DINO/DINO_4scale_swin_codino.py` (`_base_` on `DINO_4scale_swin.py`).
