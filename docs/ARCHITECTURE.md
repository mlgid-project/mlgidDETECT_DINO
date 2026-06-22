# mlgidDETECT_DINO — architecture walkthrough

A plain-language, code-grounded tour of the whole model: from a polar GIWAXS image in,
to a list of detected peak boxes out. Every step lists its **input → output tensor shape**,
a **rough formula**, the **file:line** it lives at, and where the **🔧 current modifications**
and **🚀 candidate improvements** (DINO-5, stage-freezing) plug in.

Shapes use **B** = batch size. The detector input is `(B, 1, 512, 1024)` — one grayscale
channel, **512 = χ (chi, vertical)**, **1024 = q (horizontal)**. Hyperparameters below are the
real values from `config/DINO/DINO_4scale_swin.py`.

---

## Big picture (one breath)

```
image ─▶ BACKBONE (Swin-L, 4 stages)  ─▶ NECK (project to 256ch, +1 level)
      ─▶ ENCODER (6× deformable attn)  ─▶ pick 900 queries
      ─▶ DECODER (6× deformable attn)  ─▶ heads ─▶ 900 (class, box) ─▶ NMS ─▶ AP
```

Three jobs: the **backbone** turns the image into feature maps at several zoom levels; the
**neck** standardizes them; the **DINO transformer** turns features into a fixed list of object
guesses and refines them. Only the **backbone** is ever initialized from pretrained weights
(ImageNet *or*, here, our SSL backbone) — everything after it trains from scratch every run.

---

## Shape flow at a glance

| stage | tensor | stride |
|---|---|---|
| input image | `(B, 1, 512, 1024)` | 1 |
| patch embed | `(B, 32768, 192)` | 4 |
| Swin stage 1 | grid 128×256, C=192 | 4 |
| Swin stage 2 → **C2** | `(B, 384, 64, 128)` | 8 |
| Swin stage 3 → **C3** | `(B, 768, 32, 64)` | 16 |
| Swin stage 4 → **C4** | `(B, 1536, 16, 32)` | 32 |
| neck → P2,P3,P4,P5 | each `(B, 256, …)` | 8/16/32/64 |
| flatten+concat | `(B, 10880, 256)` | — |
| encoder ×6 | `(B, 10880, 256)` | — |
| queries | `(B, 900, 256)` + boxes `(B, 900, 4)` | — |
| decoder ×6 | `(B, 900, 256)` | — |
| **output** | classes `(B,900,2)` + boxes `(B,900,4)` | — |

---

# PART A — BACKBONE: Swin-L
`models/dino/swin_transformer.py` · built at `models/dino/backbone.py:172`

Turns the image into 3 feature maps of increasing abstraction / decreasing resolution.

### A0 · Patch embed — `PatchEmbed` (swin_transformer.py:416)
- `(B, 1, 512, 1024)` → `(B, 32768, 192)`
- A `Conv2d(1→192, k=4, s=4)` slices the image into 4×4 patches, then flatten + LayerNorm.
  `128=512/4`, `256=1024/4`, so `L = 128·256 = 32768` tokens of dim 192.
- Math: `x = LayerNorm(flatten(W_patch * I))`
- 🔧 **in_chans=1** (grayscale, not RGB) — `config:` backbone built with `in_chans=args.num_channels`.

### A1 · One Swin block — `SwinTransformerBlock` (swin_transformer.py:150)
The repeated unit. Tokens are viewed as an (H,W) grid; one block does:
1. **Window partition** → tile into **48×6** windows = 288 tokens each.
2. **Windowed self-attention** — `WindowAttention` (swin_transformer.py:69), attention *inside*
   each window only:
   `Attn(Q,K,V) = softmax(QKᵀ/√d + B_rel) · V`
   where `B_rel` is the learned **relative-position bias** (its table shape depends on the
   window size — why a window-12 ImageNet checkpoint can't fully load here).
3. **MLP** (`Mlp`, swin_transformer.py:18): `Linear(C→4C)→GELU→Linear(4C→C)`, with residuals + LN.
- Every 2nd block **shifts** the windows by (24,3) first (SW-MSA) so neighboring windows talk.
- Shape unchanged by a block.
- 🔧 **window 48×6** (tall-narrow) — `config:36-37 window_size_h=48, window_size_w=6`.

### A2 · Stages + downsampling — `BasicLayer` (swin_transformer.py:292), `PatchMerging` (:251)
Each stage stacks blocks at one resolution; **PatchMerging** between stages concatenates 2×2
neighbors and halves H,W / doubles C: `Linear(4C→2C)`. Depths = `[2,2,18,2]`.

| stage | blocks | grid | C | stride | returned? |
|---|---|---|---|---|---|
| 1 | 2 | 128×256 | 192 | 4 | no — 🚀 **DINO-5 would add this level** |
| 2 | 2 | 64×128 | 384 | 8 | ✅ C2 |
| 3 | 18 | 32×64 | 768 | 16 | ✅ C3 |
| 4 | 2 | 16×32 | 1536 | 32 | ✅ C4 |

> By stage 3 the 48-tall window already spans the **whole χ axis** — one attention sees a ring's
> full vertical extent. That is why the elongated window suits tall ring features.

**Which levels leave the backbone:** `out_indices=(1,2,3)` → stages 2,3,4 (backbone.py:209
reads their channel counts `[384,768,1536]`).

### 🔧 How the backbone gets its weights — `backbone.py:190-208`
The module is **built first** with the exact arch, *then* values are poured in:
```
backbone.load_state_dict(_tmp_st, strict=False)     # backbone.py:207
```
- Works because the checkpoint keys/shapes match the rebuilt module 1:1 (our SSL export was made
  with the identical `build_swin_transformer` config → `missing=0 unexpected=0`).
- `strict=False` + a key filter (`:200`) drop `head.*` and tolerate any non-matching key.
- Only fires `if "backbone_dir" in args` — the from-scratch baseline has none → **random init**.
- 🔧 **SSL init**: `config/DINO/DINO_4scale_swin_ssl.py` sets `backbone_dir` → our SimMIM weights.
- 🚀 **Stage freezing (Idea 5b)**: `backbone.py:184` sets `requires_grad=False` on params whose
  name matches `backbone_freeze_keywords` (currently `None`). Same shapes/flow, weights frozen.

---

# PART B — NECK: input projections
`models/dino/dino.py:96-112`

Standardize every backbone level to **256** channels and synthesize one extra coarse level.
The projection convs **size themselves to the backbone's channel counts** (read from
`backbone.num_channels`, dino.py:101) — this is what makes DINO backbone-agnostic.

| level | op (file:line) | out shape | stride |
|---|---|---|---|
| P2 | `Conv1×1(384→256)`+GroupNorm — dino.py:102 | `(B,256,64,128)` | 8 |
| P3 | `Conv1×1(768→256)` | `(B,256,32,64)` | 16 |
| P4 | `Conv1×1(1536→256)` | `(B,256,16,32)` | 32 |
| P5 | `Conv3×3 s2(1536→256)` — dino.py:107 | `(B,256,8,16)` | 64 |

**Flatten + concatenate** each level to `(B, H·W, 256)` and stack:
`8192+2048+512+128 = 10880` tokens → `(B, 10880, 256)`, plus a **sine positional embedding** and a
learned **level embedding** so each token knows (where, which-scale).
- 🚀 **DINO-5 (Idea 6)**: add P1 (stage-1, stride 4) → 5 levels; sequence grows
  `10880 → ~43k` tokens (≈4× encoder cost). `config:70 num_feature_levels`,
  `config:47 return_interm_indices`.

---

# PART C — ENCODER: deformable self-attention ×6
`DeformableTransformerEncoderLayer` (deformable_transformer.py:765), stacked by
`TransformerEncoder` (:434). `config:49 enc_layers=6`.

Refines the 10880-token sequence. Each layer = deformable self-attention + FFN; shape in = out.

- **Deformable attention** — the efficiency trick: a token attends not to all 10880 others but to
  `enc_n_points=4` learned sample points **per level × 4 levels = 16** locations:
  ```
  DeformAttn(z_q, p_q) = Σ_m W_m Σ_{l,k} A_{mlqk} · W'_m · x_l( φ_l(p_q) + Δp_{mlqk} )
  ```
  `m`=8 heads, `Δp`=learned offsets, `A`=softmax attention weights over the 16, `x_l(·)`=bilinear
  sample. `config:56 nheads=8`, `config:71 enc_n_points=4`.
- **FFN**: `Linear(256→2048)→ReLU→Linear(2048→256)` + residual/LN. `config:53 dim_feedforward=2048`.
- Output = **memory**, `(B, 10880, 256)`.

---

# PART D — QUERIES + DECODER + HEADS
`DeformableTransformerDecoderLayer` (deformable_transformer.py:822), `TransformerDecoder` (:579).

### D1 · Query selection (two-stage) — `config:78 two_stage_type='standard'`
A scoring head ranks all 10880 memory tokens; keep the **top `num_queries=900`** → their positions
become initial **reference boxes**. Out: queries `(B,900,256)` + boxes `(B,900,4)`.
`config:57 num_queries=900`.
> Training only — **denoising**: noised ground-truth boxes are added as extra queries to stabilize
> bipartite matching (the "DN" in DINO); removed at inference.

### D2 · Decoder ×6 — `config:50 dec_layers=6`
Each layer, on the 900 queries:
1. **Self-attention** among queries (standard MHA, 8 heads) — `softmax(QKᵀ/√d)V` → deduplicate.
2. **Deformable cross-attention** — each query samples `dec_n_points=4`×4 levels around its
   reference box in *memory*. `config:72 dec_n_points=4`.
3. **FFN** (256→2048→256).
4. **Box refinement** — update the reference box from this layer's output (iterative).
- Emits `(B,900,256)` per layer (the 6 outputs feed auxiliary losses).

### D3 · Prediction heads (per decoder layer) — dino.py:132-133
- **Class head**: `Linear(256 → 2)` → logits `(B,900,2)`. 🔧 **num_classes=2** (ring / segment, was
  91 COCO).
- **Box head**: `MLP(256→256→256→4)` → `(B,900,4)`, added to the reference → `(cx,cy,w,h)∈[0,1]`.

### Final transformer output
**classes `(B, 900, 2)`** + **boxes `(B, 900, 4)`**.

---

# After the model (inference)
1. Sigmoid class logits → confidence; take top scores.
2. Scale boxes to `[512, 1024]`.
3. **NMS** — `perform_nms` (util/nms.py:4), ring/segment-aware.
4. **AP** — q-space matcher vs ground-truth peaks on labeled `.h5` (41 + organic), per-epoch.

---

# Where each lever plugs in (summary)

| lever | part | mechanism | status |
|---|---|---|---|
| 🔧 SSL backbone init | A | `backbone_dir` → SimMIM weights via load_state_dict (backbone.py:207) | live (winning on organic) |
| 🔧 window 48×6, in_chans=1 | A | swin build args (backbone.py:172) | live |
| 🔧 num_classes=2 (ring/segment) | D3 | `Linear(256→2)` (dino.py:132) | live |
| 🔧 q-space + ring/segment NMS, pygid eval | post | util/nms.py, util/pygidloader.py | live |
| 🚀 stage freezing (5b) | A | `requires_grad=False` (backbone.py:184) | candidate, no new SSL run |
| 🚀 DINO-5scale (6) | B | +P1 level, `num_feature_levels=5` | candidate, ~4× encoder cost |

**Through-line:** DINO-5 and freezing both extract *more* from the SSL backbone we already have —
one by **using** its discarded sharp stage-1 features, the other by **preserving** its real-data
features during the sim-only detector training — and neither changes the backbone weights.

> Config values cited from `config/DINO/DINO_4scale_swin.py`. If the model is reconfigured, re-check
> the line references — they were accurate as of this writing (branch `backbone-ssl`).
