# Cross-DINO Adoption — Feasibility & Investigation

Target: adopting **Cross-DINO** (arXiv:2505.21868, May 2025) into `mlgidDETECT_DINO`
(DINO + Swin-L 48×6 elongated-window, 4-scale, single-channel, ONNX-exported diffraction-peak detector).

Scope: assess whether Cross-DINO helps our dominant failure mode (**faint / small / low-contrast / high-q / segment-peak recall**), which is exactly the small-object regime Cross-DINO targets. Every claim below is grounded in either the fetched paper or a `file:line` in this repo.

---

## 1. Executive summary + verdict

**Verdict: adopt in part, not in full. Port the two ONNX-safe, cheap pieces (Boost Loss + Category-Size soft label first, then the CCTM feature-enrichment block). Do NOT adopt the backbone. Do NOT import the Cross-DINO repo wholesale.**

Why:

- Cross-DINO's headline "+4.4 AP_small over DINO" (36.4 vs 32.0 on COCO) is **mostly the backbone swap**, not the novel detector modules. Per the paper's own ablations, replacing ResNet-50 with their CLAP-Strip-MLP backbone gives **~+3.0 AP_small** (32.0 → 35.0), while the two portable modules — **CCTM + Boost Loss — together give only ~+1.4 AP_small on that strong backbone, and only ~+0.6 AP_small on ResNet-50** (32.0 → 32.6). So the piece that carries the gain is precisely the piece we **cannot** take: it would mean throwing away our 48×6 elongated-window Swin-L, the single-channel patch-embed, and the whole ONNX-export path that is already validated.
- The two portable modules are, however, **cheap and low-risk**:
  - **Boost Loss + CS soft label** is a *training-only* change to `SetCriterion.loss_labels` (`models/dino/dino.py:355`). **Zero ONNX/inference impact** (loss is never on the exported forward path). It directly targets classification confidence on small/faint positives — arguably a better match to our "faint peak recall" failure than to COCO's generic small objects. **Highest ROI, lowest risk — try first.**
  - **CCTM** is an elementwise gating fusion inserted at the encoder→decoder boundary in `models/dino/deformable_transformer.py:308`. It is pure multiply/add/sigmoid → **ONNX-traceable**, does **not** touch the MSDeformAttn custom op, and does **not** change the feature-level count. Moderate effort, low risk.
- The obvious "just add a high-resolution P2 scale" shortcut is **already exhausted**: the team's prior "5-scale" experiment (`config/DINO/DINO_5scale_swin_ssl.py:5-6`) *was* the stride-4 P2 level, and it was UNHELPFUL. So the small-object win, if any, must come from the modules, not from more scales.

**Payoff-vs-effort call:** Expected payoff is **modest** (paper shows +0.6 AP_small for these modules without the special backbone), but the **cost and risk of the first experiment (Boost Loss + CS label) are near-zero** — one file, no ONNX change, fine-tune from the current checkpoint, A/B on the existing `--eval` pipeline. That asymmetry makes it worth a run. CCTM is a reasonable second experiment. The full method (backbone included) is **not** worth it here.

---

## 2. Cross-DINO paper breakdown

Sources: abstract (`arxiv.org/abs/2505.21868`) + HTML body (`arxiv.org/html/2505.21868v1`), cross-checked in two independent fetches. No public code repository was found (web search returned only arXiv mirrors). **Caveat:** exact loss equations below were reconstructed from the HTML render by a summarizer; treat the *formulas* as "verify against the PDF before implementing," while the *component names, placements, and ablation numbers* were consistent across both fetches.

### Components and where they sit in the DETR pipeline

| # | Component | Pipeline location | Nature |
|---|-----------|-------------------|--------|
| 1 | **CLAP-Strip-MLP** ("Deep MLP Network") | **Replaces the CNN backbone entirely** | New architecture; carries most of the small-object gain |
| 2 | **CCTM** (Cross Coding Twice Module) | **Between Transformer encoder and decoder** — fuses backbone feature `B` with encoder feature `E` → `E_cf` → decoder | Elementwise multiplicative gating (no attention, no concat) |
| 3 | **Category-Size (CS) soft label** | **Loss / label-assignment target** (classification head) | `cs_i = sqrt((h_i/H)·(w_i/W)) · y_i` — object size folded into the class target |
| 4 | **Boost Loss** | **Classification loss** (augments/replaces focal CE) | Up-weights confidence of small/faint positives using the CS soft label |

Notes on each:

- **CLAP-Strip-MLP** replaces the ResNet/Swin backbone with a Strip-MLP variant ("obtain more comprehensive feature representations… allowing it to serve as a general vision backbone"). It still emits **4-scale** feature maps — Cross-DINO keeps the standard 4-scale pyramid; **no P2/high-res level is added**.
- **CCTM** ("cross code the backbone feature `B` and the encoder feature `E` to get more fine-grained Cross Feature `E_cf`"). Reconstructed operations: a two-step gated mix, roughly `E'_cross1 = E + B·(1 − E')` then `E_cf = 2·E·B'·E' + B·(1 − B'·E')`, where `E'`, `B'` are gating maps (sigmoids). The key structural fact — verified in both fetches — is **it sits after the encoder, before the decoder, and reinjects the raw backbone detail into the encoder memory**.
- **CS soft label**: `cs_i = sqrt((h_i/H)(w_i/W)) · y_i` (Eq. 4). Small boxes → small positive target; folds size into the target used by the classification loss.
- **Boost Loss** (Eq. 5, augments the decoder classification loss; α=0.25, β=1.0, γ=2.0):
  `L_Boost = -(1/N) Σ [ α(1−cs_i^β)^γ · cs_i^β · log(p_i) + (1−α) p_i^γ (1−y_i) log(1−p_i) ]`.

### Baseline, params, schedule

- **Assumed baseline backbone: ResNet-50** (12-epoch DINO-4scale). The paper does **not** use Swin as its baseline.
- **45M params, ~277 GFLOPs** for the full Cross-DINO (CLAP-Strip-T) — *fewer* params/FLOPs than DINO-R50 (47M / 279G). The module-only add-on (CCTM+Boost on R50) costs **+1M params / +9 GFLOPs**.
- **12-epoch** training schedule (COCO).

### Ablations — which piece carries the small-object gain

COCO, 12 epochs (numbers as rendered from the paper's Table VI / backbone table; AP_S = AP_small):

| Configuration | AP | AP_S | Δ AP_S |
|---|---|---|---|
| DINO, ResNet-50 (baseline) | 49.0 | 32.0 | — |
| + CCTM only | 49.8 | 32.4 | +0.4 |
| + Boost Loss only | 49.6 | 32.2 | +0.2 |
| + CCTM + Boost Loss | 50.0 | 32.6 | **+0.6** |
| DINO, CLAP-Strip-T backbone | 51.7 | 35.0 | **+3.0** (backbone swap) |
| + CCTM (on Strip) | 52.4 | 35.7 | +0.7 |
| + Boost Loss (on Strip) | 52.6 | 36.2 | |
| **Full Cross-DINO** (Strip + CCTM + Boost) | 52.6 | **36.4** | **+4.4 total** |

**Read this carefully:** of the +4.4 AP_small, **~+3.0 is the backbone** (untransferable to our project), and **~+1.4 is CCTM+Boost — but only +0.6 of that is realized without the special backbone.** The portable part is real but small.

---

## 3. Current-architecture map (file:line)

### Backbone — Swin-L, elongated 48×6 window, single-channel
- `config/DINO/DINO_4scale_swin.py:35-40` — `backbone='swin_L_384_22k'`, `window_size_h=48`, `window_size_w=6`, `patch_size_h/w=4`, **`num_channels=1`** (single-channel input).
- `config/DINO/DINO_4scale_swin.py:47` — `return_interm_indices=[1,2,3]` → backbone returns **stages 1,2,3** (strides **8/16/32**). Stage 0 (stride 4) is discarded.
- `models/dino/swin_transformer.py:737-744` — `swin_L_384_22k` entry sets `embed_dim=192`, `depths=[2,2,18,2]`, `num_heads=[6,12,24,48]`, **`window_size_h=48, window_size_w=6, in_chans=1`** (these lines are present/committed — contrary to the older team note that they were commented out).
- `models/dino/swin_transformer.py:548,573` — `num_features = [192, 384, 768, 1536]` per stage.
- `models/dino/backbone.py:172-209` — `build_swin_transformer(...)` wired with `in_chans=args.num_channels`; `bb_num_channels = backbone.num_features[4-len(return_interm_indices):]` → **`[384, 768, 1536]`** for our config (`backbone.py:209`).
- `models/dino/swin_transformer.py:433` — patch-embed is a single `Conv2d(in_chans, embed_dim, kernel=(4,4), stride=(4,4))` — the *only* place the single-channel assumption lives.
- ONNX-safety patches already in the backbone: `swin_transformer.py:366-394` (mask built without in-place indexing), `swin_transformer.py:398-407` (gradient checkpointing disabled under `torch.onnx.is_in_onnx_export()` / tracing).

### Input projection / 4-scale construction
- `models/dino/dino.py:96-119` — `self.input_proj`: for the 3 backbone levels, `Conv2d(C,256,1)+GroupNorm`; for the extra level(s), `Conv2d(·,256,3,stride=2,pad=1)+GroupNorm`.
- `models/dino/dino.py:242-259` — `DINO.forward`: per-level `input_proj[l](src)` builds `srcs`; **the 4th level (stride 64) is synthesized by a stride-2 conv on the last backbone feature** (`dino.py:247-259`), with its mask/pos-encoding derived on the fly.
- Net result: **current 4 scales = strides 8, 16, 32, 64** (finest = stride 8, i.e. 64×128 for a 512×1024 input). **There is no stride-4 P2 level.**
- `config/DINO/DINO_4scale_swin.py:70-72` — `num_feature_levels=4`, `enc_n_points=dec_n_points=4`.

### Deformable encoder entry & encoder→decoder boundary
- `models/dino/deformable_transformer.py:256-291` — `forward(srcs, masks, …)`: flattens/concats the 4 levels into `src_flatten` (bs, Σhw, 256), builds `spatial_shapes`, `level_start_index`, `valid_ratios`, adds `level_embed`.
- `models/dino/deformable_transformer.py:299-308` — **encoder call** → `memory` (bs, Σhw, 256). **This is `E`; `src_flatten` is `B`.**
- `models/dino/deformable_transformer.py:318-360` — two-stage query selection (`two_stage_type='standard'`): `gen_encoder_output_proposals(memory,…)` → `enc_out_class/bbox_embed` → top-k → decoder targets. **CCTM would fuse `memory` with `src_flatten` immediately after line 308, before line 318.**

### Class / box heads
- `models/dino/dino.py:132-152` — `_class_embed = nn.Linear(256, num_classes)`, `_bbox_embed = MLP(256,256,4,3)`, shared across all 6 decoder layers; attached to `transformer.decoder`.
- `models/dino/dino.py:284-290` — per-layer class/box outputs → `out['pred_logits']`, `out['pred_boxes']`.

### Loss / criterion (Boost Loss + CS label target)
- `models/dino/dino.py:355-379` — `SetCriterion.loss_labels`: builds a **one-hot** `target_classes_onehot` (`dino.py:368-372`) and applies **`sigmoid_focal_loss(..., alpha=0.25, gamma=2)`** (`dino.py:373`). **This is the exact site the CS soft label + Boost Loss replace.**
- `models/dino/dino.py:395-421` — `loss_boxes` (has matched boxes → box sizes available for CS).
- `config/DINO/DINO_4scale_swin.py:91-102` — loss coefficients (`cls_loss_coef=1.0`, `focal_alpha=0.25`, etc.).

### ONNX export path (the hard constraint)
- `export_onnx.py:17-27` — `DINOOnnxWrapper` exposes **only** `out["pred_logits"], out["pred_boxes"]` to the trace. (So anything on the loss/aux/interm branches is invisible to export.)
- `export_onnx.py:30-59` — export runs on **CPU**, opset 16, `do_constant_folding=True`.
- `models/dino/ops/modules/ms_deform_attn.py:123` and `models/dino/ops/functions/ms_deform_attn_func.py:21` — MSDeformAttn uses a **custom CUDA op** with **no ONNX symbolic** and **no CPU kernel**.
- `backbone_curation/export_onnx_ensemble.py:23-35` — **the working export shim**: it rebinds `MSDeformAttnFunction` to the **pure-PyTorch `ms_deform_attn_core_pytorch`** (`ms_deform_attn_func.py:41`, a `grid_sample`-based implementation) for export only. This is ONNX-traceable and level-count-agnostic. **Key consequence for us:** the exported deformable-attention graph is plain `grid_sample`, so *anything built from standard tensor ops around it also exports fine.*

---

## 4. Component-by-component port plan (this codebase)

### 4a. Boost Loss + CS soft label — RECOMMENDED, try first
- **Insertion point:** `models/dino/dino.py:355-379` (`SetCriterion.loss_labels`), plus mirror it in the DN loss path if DN uses the same routine (check `dn_components` / weight_dict at `dino.py:770-773`).
- **Changes:**
  1. Replace the hard one-hot target (`dino.py:368-372`) with the CS soft target for matched positives: `cs = sqrt((w·h)) · onehot` using the matched target box `w,h` (already normalized to image size in cxcywh; available via `targets[...]["boxes"]` and `indices`). Negatives stay 0.
  2. Replace `sigmoid_focal_loss(...)` at `dino.py:373` with the Boost-Loss formula (add a new function beside `sigmoid_focal_loss` in `models/dino/utils.py`). Gate behind a config flag (e.g. `use_boost_loss=True`) so it's an A/B toggle.
  3. Add `α, β, γ` to `config/DINO/DINO_4scale_swin.py` (defaults 0.25 / 1.0 / 2.0).
- **Composes with Swin-L?** Yes — completely backbone-agnostic. It only touches targets/loss.
- **ONNX impact:** **none** — the loss is never traced (`export_onnx.py:26` only forwards `pred_logits`/`pred_boxes`).
- **Effort:** ~1 file (`dino.py`) + a helper in `utils.py` + config flags. Fine-tune from current checkpoint.
- **Domain caveat:** CS uses `sqrt(area)`. Our objects are diffraction arcs/rings — very elongated boxes — so `sqrt(w·h)` can behave oddly (a long thin segment has large `w` but tiny `h`). Consider tuning β or using min(w,h) / a domain-appropriate size proxy. This is the main thing to validate empirically.

### 4b. CCTM — RECOMMENDED as second experiment
- **Insertion point:** `models/dino/deformable_transformer.py`, right after the encoder returns `memory` (`deformable_transformer.py:308`) and before `gen_encoder_output_proposals` (`:323`). Add `memory = self.cctm(memory, src_flatten)`.
  - `memory` (`E`) and `src_flatten` (`B`) are both `(bs, Σhw, 256)` and **token-aligned** (same flatten order) — so CCTM is a straight elementwise gated fusion. No re-flattening, no level bookkeeping.
- **Module:** a small `nn.Module` (two `Linear`/`Conv1d` gate projections + sigmoids + the multiply/add of Eq. 2–3). Instantiate in `DeformableTransformer.__init__` (near `deformable_transformer.py:157-181`) and wire a `use_cctm` flag through `build_deformable_transformer`.
- **Note on `B`:** the paper's `B` is the *raw backbone* feature; our `src_flatten` is the *input-projected* backbone feature (already at `d_model=256`, `dino.py:244`). This is actually more convenient (dims already match `E`), and is faithful enough to the CCTM intent (reinject pre-encoder detail). If desired, one could instead pass a separate projection of the finest raw level, but that adds complexity for little reason.
- **Composes with Swin-L?** Yes — it operates on encoder tokens, downstream of the backbone; backbone identity is irrelevant.
- **ONNX impact:** **safe** — only `Linear`/matmul/`sigmoid`/`mul`/`add`, all opset-16 core ops; does not touch MSDeformAttn. It *is* on the traced path (feeds the decoder → `pred_logits`/`pred_boxes`), so it will be exported — but every op it uses is standard.
- **Effort:** ~2 files (`deformable_transformer.py` + new module, or inline) + build/config wiring. Retrain (fine-tune) from checkpoint. New params are small.

### 4c. CLAP-Strip-MLP backbone — NOT recommended
- **Would require** replacing `build_swin_transformer` in `models/dino/backbone.py:172-209` with a Strip-MLP backbone, re-deriving `bb_num_channels`, and — critically — **discarding**: the 48×6 elongated window (`swin_transformer.py:741-743`), the single-channel patch-embed (`swin_transformer.py:433`), the ONNX-safe mask/checkpoint patches (`swin_transformer.py:366-407`), and every downstream assumption tuned for Swin-L features.
- **Composes with our constraints?** Poorly. Strip-MLP is a fixed-token-grid MLP mixer; its ONNX-exportability, its behavior on 512×1024 non-square single-channel inputs, and the availability of a domain-appropriate pretrained checkpoint are all **unknowns**. There is **no public Cross-DINO code** to lift from.
- This is the piece with the biggest reported gain (~+3.0 AP_small) but the worst effort/risk fit here. **Skip.**

---

## 5. Compatibility analysis / hard constraints

### ONNX-exportability (the deciding constraint)
| Component | Traced by export? | ONNX verdict |
|---|---|---|
| Boost Loss + CS label | **No** (loss branch not in `DINOOnnxWrapper`, `export_onnx.py:26`) | **Zero risk** |
| CCTM | Yes (feeds decoder) | **Safe** — only `Linear`/mul/add/sigmoid, opset-16 core |
| CLAP-Strip-MLP backbone | Yes | **Unknown / high risk** — untested ops, no reference impl |

The MSDeformAttn CPU/ONNX gotcha (`ms_deform_attn.py:123` custom CUDA op, no symbolic) is already solved by the pure-PyTorch `grid_sample` rebind (`backbone_curation/export_onnx_ensemble.py:23-35`). Neither portable module interacts with it.

### 48×6 elongated window + single-channel input
- Cross-DINO assumes **square-window backbones and RGB** (ResNet-50 / Strip-MLP). **Only the backbone component (4c) is sensitive to this** — and we're not taking it.
- **CCTM and Boost Loss are window-shape- and channel-agnostic**: CCTM works on flattened encoder tokens; Boost Loss works on targets/logits. Our 48×6 window and `in_chans=1` (`swin_transformer.py:741-743`, `config:40`) are irrelevant to both. This is the reason the portable subset is attractive: it sidesteps every one of our customizations.

### 4-scale + MSDeformAttn interaction if levels change
- Neither portable module changes the feature-level count → **no MSDeformAttn / `n_levels` interaction, no re-init of `sampling_offsets`/`level_embed`.**
- (Only relevant if one tried the "add a scale" route — see §6. The pure-PyTorch export core is level-count-agnostic, and MSDeformAttn's `n_levels` is set from `num_feature_levels`, so scale changes are *mechanically* fine; they were just found unhelpful.)

### Port modules onto current DINO vs import Cross-DINO wholesale
**Recommendation: port the two modules onto the current DINO. Do NOT import the Cross-DINO repo wholesale.**
- There is **no public Cross-DINO code** to import anyway (search found none).
- Even if there were, importing it would drag in the Strip-MLP backbone and a different DINO fork, forcing us to **re-earn** the 48×6 window, single-channel patch-embed, the ONNX mask/checkpoint patches (`swin_transformer.py:366-407`), and the CPU-export MSDeformAttn shim (`export_onnx_ensemble.py`). CCTM (~30 lines) and Boost Loss (~30 lines) are far cheaper to reimplement from the equations than to re-integrate all of our deployment plumbing into their tree.

---

## 6. Cheaper "minimum viable subset"

**The MVS is: Boost Loss + CS soft label (loss only), optionally then CCTM.** That is essentially "Cross-DINO minus the backbone," and it is exactly the portable, ONNX-safe part.

Reconciling with the two obvious cheaper ideas:

1. **"Just add a higher-res P2/P3 scale."** **Already tried and unhelpful.** `config/DINO/DINO_5scale_swin_ssl.py:5-6,19-20` shows the prior "5-scale" run *was* the stride-4 P2 level (`return_interm_indices=[0,1,2,3]` → strides 4/8/16/32 + a stride-64 level). Per team memory it did not help, at ~4× encoder tokens / 2–3× train cost (single-GPU only). So more scales is **not** the lever — consistent with Cross-DINO itself, which also stays at 4 scales and gets its small-object gain from the backbone + modules, not from resolution.
2. **"Only the loss/label-assignment component."** This is the recommended first step precisely because it is the cheapest possible change (one loss function), has **zero ONNX cost**, and — unlike generic scale-adding — directly attacks *classification confidence on faint/small positives*, which is our stated failure mode. The paper measures its isolated effect at +0.2–0.6 AP_small on COCO R50; our narrow physics-synthetic domain with a specific faint-peak failure could respond differently (better or worse — unknown, hence experiment).
3. **CCTM as a "lightweight feature-enrichment head."** This is the middle option: more than a loss tweak, less than a backbone swap; ONNX-safe; reinjects fine backbone detail into the encoder memory. Reasonable second experiment.

**MVS mapping to code:** §4a (`dino.py:355-379` + `utils.py`) for the loss; §4b (`deformable_transformer.py:308`) for CCTM. Nothing else needs to move.

---

## 7. Effort & risk estimate + experiment plan

### Effort / files touched
| Change | Files | Retrain | ONNX risk | Params/FLOPs | Effort |
|---|---|---|---|---|---|
| **Boost Loss + CS label** | `models/dino/dino.py` (loss), `models/dino/utils.py` (helper), config flags | fine-tune from checkpoint | **none** | 0 / 0 | **low** |
| **CCTM** | `models/dino/deformable_transformer.py` (+ small module), build + config wiring | fine-tune from checkpoint | low | small | medium |
| Strip-MLP backbone | `backbone.py` + new backbone + full ONNX re-validation | from scratch | high | changes everything | **high — not recommended** |

### Retrain cost
Both portable changes can **fine-tune from the current best checkpoint** rather than train from scratch (the schedule is `epochs=500`, `lr=1e-5`, `config:15,24`). Budget a short fine-tune (tens of epochs) per variant; eval every 2 epochs is already wired (`config:9-10`).

### Key unknowns / risks
- **Magnitude of payoff:** paper says +0.6 AP_small for these modules on R50. Could be smaller or larger on our domain — genuinely unknown.
- **CS `sqrt(area)` on elongated arc/ring boxes** may mis-weight (see §4a). Mitigate by tuning β or a size proxy.
- **Equation fidelity:** the Boost/CS formulas here were reconstructed from the HTML render. **Before implementing, verify Eq. 4–5 against the paper PDF** (the PDF fetch was unreadable in this session; get a clean copy).
- **CCTM exact gating form** (Eq. 2–3) is likewise reconstructed — verify against the figure/equations. The *placement* (post-encoder, pre-decoder) is well-confirmed.

### Recommended A/B plan (uses the repo's `--eval` AP pipeline)
Baseline = current DINO checkpoint. Eval sets are already configured: `config/DINO/DINO_4scale_swin.py:5-8` → **`41`** and **`organic`** (`.h5`), AP logged per set every 2 epochs (`config:9-10`). Gate decisions on the **`organic`** set (the harder, real-data set that reflects the faint/low-contrast failure regime); use `41` as the easier control.

1. **Exp A — Boost Loss + CS label only.** Fine-tune from checkpoint; A/B AP on `41` + `organic`. Cheapest, no ONNX change. If organic AP (esp. segment/faint recall) improves → keep.
2. **Exp B — CCTM only.** Fine-tune from checkpoint; same A/B. Export a test ONNX via `backbone_curation/export_onnx_ensemble.py` to confirm the CCTM ops trace cleanly and outputs match the torch model (parity check) — this closes the deployment-risk question early.
3. **Exp C — CCTM + Boost (only if both A and B are individually non-negative).**
4. **Stop rule:** if Exp A and Exp B both fail to move `organic` AP, do **not** pursue the Strip-MLP backbone — the paper's own ablation predicts the modules alone are worth little without it, and the backbone is off-limits here. Treat Cross-DINO as investigated-and-declined at that point.

---

## 8. References

- **Cross-DINO paper (abstract):** https://arxiv.org/abs/2505.21868 — "Cross-DINO: Cross the Deep MLP and Transformer for Small Object Detection," May 2025.
- **Cross-DINO paper (HTML body, used for component placement + ablations):** https://arxiv.org/html/2505.21868v1
- **Cross-DINO PDF:** https://arxiv.org/pdf/2505.21868 (verify Eq. 2–5 and Table VI here before implementing; this session's PDF text extraction failed).
- **DINO (baseline detector):** Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes," arXiv:2203.03605.
- **Deformable DETR (MSDeformAttn):** Zhu et al., arXiv:2010.04159.
- **Swin Transformer (backbone):** Liu et al., arXiv:2103.14030.

### Repo anchor points (for the implementer)
- Loss / Boost insertion: `models/dino/dino.py:355-379`; helper alongside `models/dino/utils.py` (`sigmoid_focal_loss`).
- CCTM insertion: `models/dino/deformable_transformer.py:308` (after encoder, before `:318` query selection); module init near `:157-181`.
- 4-scale construction / extra level: `models/dino/dino.py:96-119, 242-259`.
- Backbone (48×6, single-channel): `models/dino/swin_transformer.py:433, 737-744`; `models/dino/backbone.py:172-209`; `config/DINO/DINO_4scale_swin.py:35-47`.
- ONNX export path + MSDeformAttn CPU shim: `export_onnx.py:17-59`; `backbone_curation/export_onnx_ensemble.py:23-35`; `models/dino/ops/functions/ms_deform_attn_func.py:41`.
- Prior (unhelpful) 5-scale / P2 experiment: `config/DINO/DINO_5scale_swin_ssl.py:5-6,19-20`.
- Eval A/B sets: `config/DINO/DINO_4scale_swin.py:5-10`.
