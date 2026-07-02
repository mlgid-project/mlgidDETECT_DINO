# Semi-DETR-style Semi-Supervised Pseudo-Labeling for mlgidDETECT_DINO

Integration design document. Target: add teacher-student, mean-teacher pseudo-labeling
(Semi-DETR, Zhang et al. CVPR 2023, arXiv:2307.08095) to the existing DINO detector so we
can exploit the curated ~13k-frame **real, unlabeled** GIWAXS corpus for sim→real domain
adaptation.

All file:line references below were read directly from the repo at the time of writing
(branch `main`). Verify again before editing — line numbers drift.

> **Scope guard (important):** everything here is **TRAINING-ONLY**. Semi-supervised
> training changes weights, not the model graph, the pre-processing, or the post-processing.
> The exported ONNX inference graph consumed by the separate `mlgidDETECT` package is
> **unaffected**; `models/dino/dino.py::DINO.forward` and `PostProcess` are untouched, and
> the deployed pre/post path (`util/exp_preprocess.py`, `util/postprocessing.py`) stays in
> lockstep. A Semi-DETR run just produces a better `checkpoint.pth` to export from.

---

## 1. Executive summary + recommended phased plan

### Why this lever
Training is currently **100% physics-synthetic** (`main.py::SimulationDataset.__getitem__`
→ `simulation.py::FastSimulation.simulate_img`, main.py:61-86). Eval is on tiny real labeled
sets (~41 frames, ~8 organic). The team's own diagnostics rank sim→real domain adaptation as
the #1 untried lever, and the dominant failure mode is **faint / low-visibility / high-q /
segment-peak recall**. Two facts make Semi-DETR unusually well-matched here:

1. A mean-teacher (EMA) already exists and is wired through training + checkpointing
   (`util/utils.py::ModelEma` at utils.py:373-396; built at main.py:296-299; EMA step at
   engine.py:92-94; save/load at main.py:425-428 / 334-339). We are *adding a consumer* of an
   existing teacher, not building teacher machinery from scratch.
2. Many current false positives look like **real peaks the ground truth missed**
   (label-completeness problem). Teacher pseudo-labels on real frames can supply exactly the
   positives synthetic GT and sparse human GT both omit — this is upside, not just noise.

### What is essential vs nice-to-have for THIS project
Semi-DETR has four mechanisms (see §3). Ranked by expected value here:

| Mechanism | Verdict for this project |
|---|---|
| Mean-teacher + pseudo-labeling on real unlabeled | **Essential** (the whole point) |
| Stage-wise **hybrid matching** (one-to-many → one-to-one) | **High value** — directly attacks the "one noisy pseudo-box can't supervise enough queries" problem that hurts faint-peak recall. Adopt in phase 3. |
| Cost-based **Pseudo-Label Mining (PLM)** | **Medium** — a cheaper per-class confidence threshold captures most of the benefit first; add GMM mining only if fixed thresholds plateau. |
| **Cross-view query consistency** | **Skip for MVP.** Highest engineering cost (needs RoIAlign on query features + a second aligned decoder pass), least clearly matched to the faint-peak recall problem. Reconsider only if phases 1-3 plateau. |

### Recommended phased plan

- **Phase 0 — Warm start / burn-in.** No new code. Take the current best ensemble-member
  checkpoint (or train synthetic-only for N epochs) as the student init and seed the EMA
  teacher from it (`ema_m.set(model)`, utils.py:395-396). Pseudo-labeling on a random-init
  teacher produces garbage; a burn-in is mandatory.
- **Phase 1 — MVP mean-teacher (no hybrid matching).** Add a `RealUnlabeledDataset`, a dual
  loader, teacher pseudo-label generation with a per-class score threshold, and
  `total = L_sup(synthetic) + λ·L_unsup(real pseudo)` in a new `train_one_epoch_semi`. Reuse
  `SetCriterion` unchanged for the unsupervised branch (DN disabled on pseudo-targets — see
  §4d). This is the smallest change that tests the hypothesis (§7).
- **Phase 2 — Threshold / recall tuning.** Sweep per-class thresholds (lower for segment=0 to
  attack faint recall), λ, burn-in length, EMA decay. Watch for pseudo-label drift (§6).
- **Phase 3 — Stage-wise hybrid matching (full Semi-DETR core).** Add one-to-many assignment
  for the unsupervised branch for the first `T1` steps of the semi phase, then switch to
  one-to-one, reusing `SimpleMinsumMatcher` as the one-to-many building block (§5). Optionally
  add GMM PLM.
- **Phase 4 (optional) — cross-view query consistency**, only if warranted.

**MVP = Phase 0 + Phase 1.** **Full Semi-DETR = through Phase 3 (+4).**

Semi-DETR reference hyperparameters (COCO, from the paper — arXiv:2307.08095): EMA momentum
**0.999**; pseudo-label confidence threshold **0.4**; unsupervised loss weight **wu = 4**
(COCO-Partial) / **2.0** (COCO-Full); one-to-many stage length **T1 = 60k** iters
(Partial); matching-cost weights α=1, β=6; consistency weight wc=1. Treat these as starting
points, not gospel — our images, class count (2), and query count (900) differ.

---

## 2. Current training-path map (file:line)

| Concern | Location | Notes |
|---|---|---|
| Epoch loop | `main.py:394-458` | rebuilds dataset + loader **every epoch** |
| Synthetic dataset | `main.py:52-90` (`SimulationDataset`) | `__getitem__` 61-86; `__len__`=1000 (main.py:88-90) |
| Target dict format | `main.py:72-84` | `boxes` cxcywh norm by (w,h); `area`; `labels`=`is_ring` int64; `image_id`; `iscrowd`; `orig_size`; `size`. **All tensors created on `device='cuda'`** (main.py:57, 75, 80-84) |
| On-GPU constraint | `main.py:54-59`, DataLoader at `main.py:395-402` | `num_workers=0` **mandatory** because `__getitem__` returns cuda tensors and `FastSimulation` runs on cuda |
| collate | `main.py:92-114` | stacks images → tensor; returns `(samples, list_of_target_dicts)` |
| train call | `main.py:404-406` | passes `ema_m=ema_m` |
| Engine train loop | `engine.py:20-119` | |
| DN gate | `engine.py:27` | `need_tgt_for_training = args.use_dn` |
| Model forward | `engine.py:46-50` | `model(samples, targets)` if use_dn else `model(samples)` |
| Loss call | `engine.py:52-55` | `criterion(outputs, targets)`; weighted sum via `criterion.weight_dict` |
| **EMA step** | `engine.py:92-94` | `if args.use_ema and epoch >= args.ema_epoch: ema_m.update(model)` |
| EMA object | `util/utils.py:373-396` | `ModelEma(model, decay)`; `.module` is teacher (eval mode, utils.py:378); `.update(model)` EMAs **all** state_dict values incl. buffers (utils.py:385-393); `.set(model)` hard-copies (utils.py:395-396) |
| EMA build | `main.py:296-299` | only if `args.use_ema` |
| EMA ckpt save/load | `main.py:425-428` / `main.py:334-339` | teacher persisted as `ema_model`; on resume without it, re-inits from student |
| Model forward body | `models/dino/dino.py:221-322` | returns `pred_logits, pred_boxes, aux_outputs, interm_outputs, enc_outputs, dn_meta` |
| DN build | `dino.py:261-268` → `dn_components.py::prepare_for_cdn` (dn_components.py:20-137) | in **inference** (`training=False`) returns all-None (dn_components.py:130-135) — so `teacher.module.eval()` forward gives clean outputs, no DN |
| Criterion | `dino.py:474-628` (`SetCriterion.forward`) | one-to-one matcher at dino.py:486; label (sigmoid-focal) + box (L1+giou) + cardinality; DN losses dino.py:505-541; aux/interm/enc each re-run matcher (dino.py:549, 591, 609) |
| Matcher | `models/dino/matcher.py` | `HungarianMatcher` fwd matcher.py:47-95 (scipy `linear_sum_assignment`, one-to-one); `SimpleMinsumMatcher` fwd matcher.py:120-175 (per-target min-cost query, matcher.py:169-173); `build_matcher` matcher.py:178-191 |
| Weight dict | `dino.py:764-810` | includes `_dn`, `_{aux}`, `_interm`, `_enc` variants |
| Postprocess (deployed) | `util/postprocessing.py` | `onnx_to_xyxy` (postprocessing.py:32-51, top-225 + cxcywh→xyxy) + `filter_boxes` (postprocessing.py:54-87, class-aware NMS + score thr) |
| Postprocess (in-graph) | `dino.py:639-688` (`PostProcess`) | topk `num_select` (=150, config:86) + optional NMS |
| Real-image inference recipe | `main.py:212-226` | shows exact "forward real image → raw outputs → onnx_to_xyxy → filter_boxes → boxes+scores" pattern to copy for pseudo-labels |
| Eval (unchanged) | `main.py::evaluate_giwaxs_ap` (main.py:182-231), called main.py:449-458 | labeled AP on 41 + organic; **do not touch** |
| Detector input format | `main.py:69`, `main.py:214` | single-channel [0,1] `repeat(num_channels,1,1)`; `num_channels=1` (config:40) |

---

## 3. Semi-DETR mechanisms — adopt vs skip

Semi-DETR is the first DETR-native SSOD method (built on and evaluated with DINO). Four
pieces, with our decision:

1. **Mean-teacher pseudo-labeling.** Teacher = EMA of student (momentum 0.999). Teacher runs
   on **weakly**-augmented unlabeled images to produce pseudo-labels; student trains on
   **strongly**-augmented versions of the same images. **ADOPT (essential).** We already have
   `ModelEma`.

2. **Stage-wise hybrid matching.** DETR's one-to-one bipartite assignment is brittle under
   noisy pseudo-labels: a single pseudo-GT is matched to exactly one query, so few queries get
   a positive gradient and training is unstable/slow. Semi-DETR trains the unlabeled branch
   with **one-to-many** assignment (each pseudo-GT supervises the top-M lowest-cost queries)
   for the first `T1` iterations, then switches to standard **one-to-one**. This both
   stabilizes early learning and yields higher-quality pseudo-labels for the later one-to-one
   stage. **ADOPT in phase 3.** Directly relevant: with faint peaks, one query per pseudo-box
   is too weak a signal, and our label-completeness upside benefits from denser positives.

3. **Cross-view query consistency.** Enforces that object-query representations are consistent
   across the teacher (weak) and student (strong) views, *without* solving explicit query
   correspondence. Concretely (paper): RoIAlign features at predicted boxes → a few MLPs →
   `L_consistency = MSE(ô_student, detach(ô_teacher))`, weight wc=1. **SKIP for MVP.** Highest
   implementation cost (RoIAlign over query boxes, aligned second decoder pass, extra head),
   and its benefit is representation regularization rather than the recall problem we most need
   to fix. Revisit only after phases 1-3.

4. **Cost-based Pseudo-Label Mining (PLM).** Beyond a fixed score threshold, fit a 2-component
   GMM to the *matching cost* between pseudo-GT and student proposals per image, and treat the
   low-cost ("reliable") cluster as additional positives (threshold = cost of the reliable
   cluster center). **DEFER to phase 3 (optional).** Start with a per-class score threshold; add
   GMM mining only if thresholds plateau. Rationale: PLM's payoff is precisely recovering
   borderline boxes (faint peaks / missed GT), so it is the natural phase-3 upgrade, but it is
   not needed to test the core hypothesis.

**Net recommendation:** mean-teacher + per-class thresholds (Phase 1) → hybrid matching
(Phase 3) → optional PLM. Cross-view consistency is the one Semi-DETR component we consciously
drop unless everything else plateaus.

---

## 4. Concrete integration design

### 4a. Data — `RealUnlabeledDataset` + dual loader

**New file:** `datasets/real_unlabeled.py` (or fold into `main.py` beside `SimulationDataset`).
Reuse the corpus reader + transform already proven for SSL:

- Corpus: `/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5`,
  dataset `images (12991,512,1024) uint8`, 0 = no-data.
- Reader pattern: `backbone_curation/ssl/ssl_dataset.py::SimMIMDataset` (ssl_dataset.py:47-81)
  — per-worker lazy h5 open (`_h()`, ssl_dataset.py:59-63), `to_model_input` at ssl_dataset.py:76.
- Transform: `backbone_curation/backbone_transform.py::to_model_input` (backbone_transform.py:27-32,
  `/255 → [0,1]`, keeps no-data exactly 0). Physics-legal augment primitives already exist:
  `augment` (backbone_transform.py:41-54: vflip + gamma + noise) and `augment_v2`
  (backbone_transform.py:57-77: + exposure scale + q-ramp + noise).

**Weak vs strong augmentation for POLAR GIWAXS (q-radial × χ):**

| | WEAK (teacher / pseudo-label view) | STRONG (student view) |
|---|---|---|
| Geometry | identity, OR at most a **vertical (χ) flip** shared with the strong view | **same** vertical flip as weak (share the flip decision so pseudo-boxes remain valid), no other geometry |
| Photometric | none (or tiny) | gamma (0.7-1.4), exposure scale (0.8-1.2), q-direction intensity ramp, additive Gaussian noise, mild blur, contrast jitter |
| No-data | preserved (0 stays 0) | **preserved — reapply `x[valid==0]=0` after every op** |

Physics caveats (respect them; wrong here corrupts labels):
- **Vertical (χ) flip is OK** and is already the team's sanctioned SSL geometry op
  (backbone_transform.py:46-47 and its docstring). If used, apply the **same** flip to both
  views and flip pseudo-box `cy → 1 - cy` accordingly (weak-view boxes must map into the
  student frame). Simplest MVP: **no geometry at all** (identity weak, photometric-only
  strong) — then pseudo-boxes need no coordinate transform.
- **Horizontal (q) flip: forbidden.** q is a physical radial axis (peak position ↔ d-spacing);
  mirroring q is unphysical and would teach wrong geometry. Do **not** use it.
- **No arbitrary rotation / no aspect-changing resize** — polar geometry is not
  rotation-invariant (backbone_transform.py docstring). Peaks are elongated along χ; the
  backbone even uses an elongated 48×6 window for this reason.
- **No-data zero regions must stay zero** after strong aug (mask with `valid = x>0`, exactly as
  `augment` does at backbone_transform.py:50,53). Blur especially will bleed nonzero values into
  no-data; re-zero afterward.

**On-GPU / num_workers constraint.** The synthetic path forces `num_workers=0` because
`FastSimulation` and `SimulationDataset.__getitem__` build cuda tensors (main.py:54-86,
DataLoader at main.py:395-402). Two clean options for the real loader:

- *Option A (simplest, recommended for MVP):* keep the real dataset **CPU/numpy** (h5 read +
  numpy augment, like `SimMIMDataset`) and move to cuda inside the training step. This lets the
  real loader use `num_workers>0`. Do **not** put cuda tensors in a multi-worker dataset.
- *Option B:* mirror `SimulationDataset` and build cuda tensors in `__getitem__` with
  `num_workers=0`. Consistent with the synthetic path but no async prefetch.

Recommended: **Option A**. Augmentation is cheap numpy; the h5 is the only I/O.

```python
# datasets/real_unlabeled.py   (CPU dataset; move to cuda in the training step)
import numpy as np, torch, h5py
from backbone_curation import backbone_transform as BT   # to_model_input / augment(_v2)

class RealUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_channels=1, strong="v2", geom_flip=False):
        self.h5_path, self.num_channels = h5_path, num_channels
        self.strong, self.geom_flip, self._h5 = strong, geom_flip, None
        with h5py.File(h5_path, "r") as h:
            self.n = h["images"].shape[0]
    def _h(self):
        if self._h5 is None: self._h5 = h5py.File(self.h5_path, "r")  # per-worker open
        return self._h5
    def __len__(self): return self.n
    def __getitem__(self, i):
        u8 = self._h()["images"][i]                      # (512,1024) uint8, 0=no-data
        rng = np.random.default_rng()
        x = BT.to_model_input(u8)                        # float32 [0,1], 0=no-data
        do_flip = self.geom_flip and (rng.random() < 0.5)
        if do_flip:
            x = x[::-1].copy()                           # shared χ-flip for BOTH views
        weak   = x.copy()                                # teacher view: (near-)identity
        strong = BT.augment_v2(x, rng) if self.strong == "v2" else BT.augment(x, rng)
        # re-zero no-data after strong aug (belt-and-braces; augment_v2 already masks)
        nod = (x == 0)
        weak[nod] = 0.0; strong[nod] = 0.0
        # (1,H,W) each; channel-repeat happens after .cuda() in the step to save host mem
        return (torch.from_numpy(weak)[None],
                torch.from_numpy(strong)[None],
                bool(do_flip))

def collate_unlabeled(batch):
    weak   = torch.stack([b[0] for b in batch])          # (B,1,H,W)
    strong = torch.stack([b[1] for b in batch])
    flips  = [b[2] for b in batch]
    return weak, strong, flips
```

**Dual loader.** Keep the synthetic loader exactly as main.py:395-402. Add a second loader for
real frames and iterate them **together**, one real batch per synthetic batch:

```python
real_ds = RealUnlabeledDataset(args.unlabeled_h5, num_channels=args.num_channels,
                               strong=args.strong_aug, geom_flip=args.semi_geom_flip)
real_loader = DataLoader(real_ds, batch_size=args.unlabeled_batch_size, shuffle=True,
                         num_workers=args.unlabeled_workers, collate_fn=collate_unlabeled,
                         drop_last=True, persistent_workers=args.unlabeled_workers > 0)
# pair with the synthetic loader; real corpus (12991) >> synthetic __len__ (1000/epoch),
# so cycle the synthetic loader (or vice versa) — see train_one_epoch_semi (§4e).
```

### 4b. Teacher — reuse `ModelEma`

No new teacher code. Turn on the existing machinery via config (§4f):
`use_ema=True`, `ema_decay=0.999` (Semi-DETR value; current default `0.9997` at config:125 is
also fine), `ema_epoch = burn_in_epoch`. Semantics already correct:

- Built at main.py:296-299 when `use_ema`.
- Stepped at engine.py:92-94 **only** `if epoch >= args.ema_epoch` — reuse this to gate the
  teacher update to after burn-in.
- `ema_m.module` is the teacher, already `.eval()` (utils.py:378). For pseudo-label generation
  call it in eval mode with no targets → DN auto-disabled (dn_components.py:130-135).
- **Burn-in seeding:** at the moment pseudo-labeling starts (epoch == burn_in), hard-copy
  student→teacher once with `ema_m.set(model)` (utils.py:395-396) so the teacher isn't a stale
  average of early noise. Do this once (guard with a flag), otherwise `.update()` (engine.py:94)
  keeps EMA-tracking as normal.

**Define the burn-in.** `args.semi_start_epoch` = first epoch that adds the unsupervised loss.
Before it: pure synthetic training (current behavior), EMA either off or just tracking. Set
`ema_epoch = semi_start_epoch` (or a bit earlier so the teacher is a smoothed student by the
time it is first queried).

### 4c. Pseudo-label generation — teacher forward → target dicts

Copy the **exact** deployed inference recipe from `evaluate_giwaxs_ap` (main.py:212-226): raw
outputs → `onnx_to_xyxy` → `filter_boxes`. That keeps pseudo-labels consistent with what the
model is scored/deployed on. Then convert xyxy pixel boxes back to the **normalized cxcywh
int64-cuda target-dict format** `SimulationDataset` emits (main.py:72-84).

```python
@torch.no_grad()
def make_pseudo_targets(teacher, weak_imgs, args, cfg,
                        thr_ring=0.4, thr_seg=0.3):     # per-class score thresholds
    """weak_imgs: (B,1,H,W) cuda in [0,1]. Returns list[B] of DINO target dicts on cuda."""
    teacher.eval()
    x = weak_imgs.repeat(1, args.num_channels, 1, 1)    # match detector input (main.py:214)
    outputs = teacher(x)                                # eval + no targets => no DN (dn_components.py:130-135)
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']   # (B,Q,2),(B,Q,4) cxcywh norm
    H, W = args.data_h, args.data_w                     # 512,1024
    targets = []
    for b in range(x.shape[0]):
        prob = logits[b].sigmoid()                      # (Q,2)
        score, cls = prob.max(-1)                        # best class per query
        # per-class threshold: LOWER for segment=0 to chase faint recall (§6)
        thr = torch.where(cls == 1, torch.as_tensor(thr_ring, device=x.device),
                                    torch.as_tensor(thr_seg,  device=x.device))
        keep = score > thr
        # (optional) class-aware NMS in normalized xyxy to dedupe, mirroring filter_boxes
        bx = boxes[b][keep]                              # cxcywh norm, cuda
        lb = cls[keep].to(torch.int64)                   # segment=0 / ring=1
        # ---- build the EXACT target-dict schema (main.py:72-84) ----
        xyxy = box_cxcywh_to_xyxy(bx)                    # util.box_ops
        area = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1]) * (W*H)
        n = bx.shape[0]
        targets.append({
            "boxes": bx,                                                    # cxcywh, normalized, cuda
            "area": area,
            "labels": lb,                                                   # int64 cuda
            "image_id": torch.tensor(-1, device=x.device),                 # sentinel (unlabeled)
            "iscrowd": torch.zeros((n,), dtype=torch.int64, device=x.device),
            "orig_size": torch.tensor([H, W], device=x.device),
            "size": torch.tensor([H, W], device=x.device),
        })
    teacher.train()  # only affects dropout-less DINO buffers; safe to leave in eval too
    return targets
```

Notes:
- **Use the same `PostProcess`/`onnx_to_xyxy` selection** if you want pseudo-labels identical to
  deployment (top-`num_select`=150 then NMS). The simplified per-class-threshold path above is
  fine for MVP; add class-aware NMS (`util/postprocessing.py::filter_boxes`, postprocessing.py:54-87)
  before phase 3 to avoid duplicate pseudo-boxes inflating one-to-many positives.
- **Per-class thresholds:** ring=1 tends to be high-contrast; segment=0 is the faint,
  low-recall class. Start `thr_ring≈0.4`, `thr_seg≈0.25-0.3` and sweep. Lowering `thr_seg`
  trades precision for the faint-peak recall we care about; the label-completeness upside means
  some "false" positives are actually real missed peaks.
- **Empty pseudo-targets are valid.** DINO handles zero-object targets throughout
  (`prepare_for_cdn` at dn_components.py:39-47; `hs[0] += label_enc.weight[0,0]*0.0` at dino.py:272;
  the DN loss zero-fill at dino.py:534-541). An image with no confident pseudo-box just
  contributes only negatives — which is correct and useful.
- Teacher boxes are already normalized cxcywh in the weak-view frame. If you enabled a shared
  χ-flip (§4a), the strong view has the same flip, so **no coordinate transform is needed** as
  long as weak and strong share the flip. If weak is identity and strong is flipped (don't do
  this), you'd have to flip `cy→1-cy`.

### 4d. Loss — `total = L_sup + λ · L_unsup`

Reuse `SetCriterion` (dino.py:333-628) **unchanged** for both branches. Two forward passes
through the criterion, summed with a ramp-up weight λ:

```
L_total = L_sup(student(strong_synth), synth_targets)         # existing path
        + λ(epoch) · L_unsup(student(strong_real),  pseudo_targets)
```

- **λ (unsup weight).** Semi-DETR uses wu=4 (COCO-Partial) / 2.0 (Full). Start λ≈2.0 for the
  2-class setting and **ramp** from 0 over the first few post-burn-in epochs
  (`λ = λ_max · min(1,(epoch-semi_start)/λ_warmup)`), to avoid an early shock from noisy
  pseudo-labels. Expose as `args.unsup_loss_weight`, `args.unsup_warmup_epochs`.
- **Where set.** New args in config (§4f); ramp computed in `train_one_epoch_semi` (§4e).
- **Reusing the weight_dict.** `criterion.weight_dict` (dino.py:764-810) already scales
  ce/bbox/giou (+ aux/interm/enc/dn). The unsupervised branch produces the **same** loss keys.
  Apply the existing per-key weights, then multiply the whole unsupervised sum by λ:
  `L_unsup = λ · Σ_k weight_dict[k]·loss_unsup[k]`. Do **not** try to split λ per key.
- **DN on pseudo-targets — disable it.** Reason it through:
  - DN (`prepare_for_cdn`, dn_components.py:20-137) adds *denoising* queries built by adding
    controlled label/box noise to **ground-truth** boxes, and supervises reconstructing the
    clean GT. Its premise is that the targets are *correct*. Pseudo-labels are already noisy,
    so denoising toward them reinforces teacher error (confirmation bias) with no clean anchor.
  - Mechanically, the DN loss is keyed on `dn_meta` (dino.py:503-541) which only exists when the
    student forward is `model(samples, targets)` with `training=True` (dino.py:261-268). For the
    unsupervised branch, call the student **without** targets → `dn_meta=None` → the criterion
    takes the zero-fill DN branch (dino.py:534-541) → DN losses are exactly 0. So the clean way
    to disable DN is simply: **student forward on real images uses `model(strong_real)` (no
    targets)**, while the synthetic branch keeps `model(strong_synth, synth_targets)` (DN on).
  - Keep aux + interm + enc losses **on** for the unsupervised branch (they help and cost
    nothing extra to enable).

### 4e. Training-loop change — `train_one_epoch_semi`

New function in `engine.py` (leave `train_one_epoch` untouched for pure-synthetic /
backward-compat). Sketch (AMP/logging elided; copy the finite-check, backward, EMA-step, and
metric-logger scaffolding verbatim from engine.py:24-99):

```python
def train_one_epoch_semi(model, criterion, synth_loader, real_loader, optimizer, device,
                         epoch, ema_m, args, cfg, logger=None):
    model.train(); criterion.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    teacher = ema_m.module                                 # EMA teacher (utils.py:377)

    semi_on = args.use_semi and epoch >= args.semi_start_epoch
    lam = 0.0
    if semi_on:
        # one-time hard seed of the teacher at the semi boundary (utils.py:395-396)
        if epoch == args.semi_start_epoch and not getattr(args, "_teacher_seeded", False):
            ema_m.set(model); args._teacher_seeded = True
        lam = args.unsup_loss_weight * min(1.0, (epoch - args.semi_start_epoch + 1)
                                                 / max(1, args.unsup_warmup_epochs))

    real_iter = iter(real_loader) if semi_on else None
    for samples, targets in synth_loader:                  # synthetic-labeled batch
        samples  = samples.to(device)
        targets  = [{k: v.to(device) for k, v in t.items()} for t in targets]  # engine.py:44

        with torch.cuda.amp.autocast(enabled=args.amp):
            # ---- supervised branch (DN ON: model(samples, targets), engine.py:47-48) ----
            out_sup  = model(samples, targets) if args.use_dn else model(samples)
            ld_sup   = criterion(out_sup, targets)
            L = sum(ld_sup[k] * criterion.weight_dict[k]
                    for k in ld_sup if k in criterion.weight_dict)         # engine.py:55

            # ---- unsupervised branch ----
            if semi_on:
                try:    weak, strong, flips = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader); weak, strong, flips = next(real_iter)
                weak, strong = weak.to(device), strong.to(device)
                pseudo = make_pseudo_targets(teacher, weak, args, cfg)     # §4c (no grad)

                strong_in = strong.repeat(1, args.num_channels, 1, 1)      # match input (main.py:214)
                out_uns = model(strong_in)                                 # NO targets => DN OFF (§4d)
                ld_uns  = criterion(out_uns, pseudo)
                L_uns   = sum(ld_uns[k] * criterion.weight_dict[k]
                              for k in ld_uns if k in criterion.weight_dict)
                L = L + lam * L_uns

        optimizer.zero_grad()
        (scaler.scale(L).backward() if args.amp else L.backward())
        if args.clip_max_norm > 0:
            if args.amp: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        (scaler.step(optimizer), scaler.update()) if args.amp else optimizer.step()

        if args.use_ema and epoch >= args.ema_epoch:       # engine.py:92-94 (unchanged)
            ema_m.update(model)
        # ... metric_logger.update(...) as engine.py:96-99, log lam, L_uns separately ...
    # return resstat like engine.py:113-119
```

Wire it in `main.py` (replace the call at main.py:404-406 when `args.use_semi`):

```python
if getattr(args, "use_semi", False):
    train_stats = train_one_epoch_semi(model, criterion, data_loader, real_loader,
                                        optimizer, device, epoch, ema_m, args, cfg, logger)
else:
    train_stats = train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                                  args.clip_max_norm, ..., ema_m=ema_m)   # existing main.py:404-406
```

Build `real_loader` **once** before the epoch loop (main.py:392) — unlike the synthetic dataset
it need not be rebuilt each epoch (the synthetic one is rebuilt at main.py:395 to reseed
`FastSimulation`; the real corpus is static). One synthetic batch (`__len__`=1000 →
500 batches/epoch at bs=2, main.py:88-90) is paired with one real batch; `real_iter` cycles the
12,991-frame corpus as needed.

### 4f. New config/args knobs + checkpoint/resume

Add to a **new config** `config/DINO/DINO_4scale_swin_semi.py` (inherit the swin config; mirror
the SSL-config pattern at `config/DINO/DINO_4scale_swin_ssl.py`, launched by
`backbone_curation/ssl/run_detector_ssl.sbatch`):

| Knob | Suggested | Meaning |
|---|---|---|
| `use_semi` | `True` | master switch for the semi path |
| `unlabeled_h5` | `.../backbone_ssl_corpus.h5` | real corpus |
| `unlabeled_batch_size` | `2` | real frames per step |
| `unlabeled_workers` | `4` | CPU workers (real ds is CPU, §4a) |
| `strong_aug` | `"v2"` | `augment`/`augment_v2` selector (backbone_transform.py:41,57) |
| `semi_geom_flip` | `False` | shared χ-flip (MVP: off → no box transform) |
| `semi_start_epoch` | e.g. `50` | burn-in length; pseudo-labeling starts here |
| `unsup_loss_weight` | `2.0` | λ_max (Semi-DETR wu) |
| `unsup_warmup_epochs` | `5` | ramp length for λ |
| `pseudo_thr_ring` / `pseudo_thr_seg` | `0.4` / `0.3` | per-class score thresholds (§4c) |
| `use_ema` | **`True`** | override config:124 (`False`) |
| `ema_decay` | `0.999` | Semi-DETR momentum (config:125 is 0.9997) |
| `ema_epoch` | `= semi_start_epoch` | gate teacher updates (engine.py:93) |
| **phase-3 only** | | |
| `hybrid_matching` | `False`→`True` | enable one-to-many→one-to-one (§5) |
| `hybrid_t1_epochs` | e.g. `20` | epochs of one-to-many after `semi_start_epoch` (Semi-DETR: T1=60k iters) |
| `hybrid_topk_M` | `4` | positives per pseudo-GT in one-to-many |
| `use_plm` | `False` | GMM pseudo-label mining (optional) |

**Checkpoint / resume implications:**
- Teacher state is **already** saved/restored as `ema_model` (main.py:425-428 / 334-339). No new
  checkpoint plumbing needed. If a run resumes past `semi_start_epoch` but the checkpoint predates
  EMA (`ema_model` absent), main.py:337-339 re-inits the teacher from the student — acceptable,
  but prefer to enable `use_ema` from the burn-in start so the teacher is a smoothed student by
  the time pseudo-labeling begins.
- **Burn-in resume is automatic:** the semi/λ logic is a pure function of `epoch`
  (`epoch >= semi_start_epoch`, λ ramp), and `args.start_epoch` is restored from the checkpoint
  (main.py:344). Resubmitting the same sbatch across 72h sessions (auto-resume from
  `checkpoint.pth` in `output_dir`, main.py:325-326 / 482-483) picks the phase back up correctly.
- Persist the one-time `_teacher_seeded` guard implicitly via `epoch == semi_start_epoch`: on a
  mid-run resume you will already be past that epoch, so `ema_m.set` won't re-fire — the restored
  `ema_model` is used as-is, which is what you want.

---

## 5. Stage-wise hybrid matching (Semi-DETR core) — concrete change

Goal: for the **unsupervised** branch, during the first `T1` of the semi phase, assign each
pseudo-GT to the **top-M** lowest-cost queries (one-to-many) instead of a single query, then
switch to one-to-one. Keep the **supervised** branch one-to-one throughout (its GT is clean;
DINO already tuned for it).

Minimal version reusing existing matchers:

- One-to-one already exists: `HungarianMatcher` (matcher.py:25-95), the default (`matcher_type`
  config:105).
- One-to-many building block: generalize `SimpleMinsumMatcher` (matcher.py:98-175). Today it
  assigns each target its single argmin query (matcher.py:169-173). Change the reduction to
  **top-M** queries per target:

```python
# matcher.py — new TopkMatcher (one-to-many), based on SimpleMinsumMatcher (matcher.py:98-175)
class TopkMatcher(SimpleMinsumMatcher):
    def __init__(self, *a, topk=4, **kw):
        super().__init__(*a, **kw); self.topk = topk
    @torch.no_grad()
    def forward(self, outputs, targets):
        # ... identical cost build as matcher.py:139-164 (cost_class + cost_bbox + cost_giou) ...
        C = (self.cost_bbox*cost_bbox + self.cost_class*cost_class + self.cost_giou*cost_giou
             ).view(bs, num_queries, -1)                      # matcher.py:163-164
        sizes = [len(v["boxes"]) for v in targets]            # matcher.py:166
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            cost_i = c[i]                                     # (num_queries, n_tgt_i)
            n_tgt = cost_i.shape[1]
            if n_tgt == 0:
                indices.append((torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long)))
                continue
            k = min(self.topk, num_queries)
            topq = cost_i.topk(k, dim=0, largest=False).indices   # (k, n_tgt) lowest-cost queries
            src = topq.t().reshape(-1)                             # each tgt -> k queries
            tgt = torch.arange(n_tgt, device=cost_i.device).repeat_interleave(k)
            indices.append((src, tgt))
        return [(torch.as_tensor(s,dtype=torch.int64), torch.as_tensor(t,dtype=torch.int64))
                for s,t in indices]
```

Then let `SetCriterion` choose the matcher **per call**. Two clean routes:

- *Route 1 (least invasive):* give `SetCriterion` a second matcher `self.matcher_o2m` and a
  runtime flag `self.use_o2m`. In `SetCriterion.forward` (dino.py:474-628), swap `self.matcher`
  for `self.matcher_o2m` at the three matcher sites (dino.py:486 main, dino.py:549 aux,
  dino.py:591 interm, dino.py:609 enc) when `self.use_o2m` is set. The engine sets
  `criterion.use_o2m = hybrid_on_for_this_step` right before the **unsupervised**
  `criterion(out_uns, pseudo)` call and clears it before the supervised call. Because
  `num_boxes` normalization already divides by the number of target boxes (dino.py:492-497),
  the many-to-one duplication is handled correctly as long as the loss reduction stays
  `sum/num_boxes` — but note one-to-many increases the count of matched positives, so consider
  normalizing `loss_bbox/loss_giou` by the number of **matched pairs** rather than pseudo-GT
  count to keep the box-loss scale stable (Semi-DETR does the equivalent). Keep it simple in MVP;
  revisit if box loss explodes.

- *Route 2:* pass `matcher_override` as a kwarg through `SetCriterion.forward`. More explicit,
  touches the signature.

**Where the stage switch lives.** In `train_one_epoch_semi` (§4e), compute
`hybrid_on = args.hybrid_matching and (epoch < args.semi_start_epoch + args.hybrid_t1_epochs)`
and set `criterion.use_o2m = hybrid_on` for the unsupervised `criterion(...)` call only. After
`T1`, `hybrid_on` is False and the unsupervised branch reverts to one-to-one (`HungarianMatcher`)
— exactly Semi-DETR's stage-wise schedule.

**Where PLM sits (optional, phase 3).** PLM refines *which pseudo-GT* enter the unsupervised
criterion. It runs **inside `make_pseudo_targets` (§4c)**, after the score-threshold prefilter:
compute the matching cost between candidate pseudo-boxes and the teacher's own high-count
proposals, fit a 2-component GMM per image (`sklearn.mixture.GaussianMixture(2)`), keep boxes in
the low-cost cluster (threshold = reliable cluster center). This is a drop-in replacement for the
fixed `score > thr` filter and does not touch the matcher. Start without it.

---

## 6. Risks / caveats specific to this project

1. **Sim→real domain shift is the whole premise, but the teacher starts sim-only.** Early
   pseudo-labels on real frames may be systematically biased (e.g. miss faint high-q segments —
   the exact failure mode). Mitigations: burn-in (`semi_start_epoch`), λ ramp
   (`unsup_warmup_epochs`), lower `pseudo_thr_seg`, and — crucially — hybrid matching (phase 3)
   so a few correct pseudo-boxes still supervise many queries.
2. **Confirmation bias / pseudo-label drift on faint peaks.** The student can collapse to the
   teacher's blind spots. Guard with EMA decay 0.999 (slow teacher), λ ramp, and monitoring:
   log per-class pseudo-box **count** and mean score per epoch; if segment pseudo-count decays
   toward 0, the loop is eating its own tail — raise λ later / lower `thr_seg` / shorten burn-in.
   The labeled eval (41 + organic) is the ground-truth guardrail; it runs every `eval_interval`
   (main.py:441-458) and is unchanged.
3. **RGB-COCO → polar-single-channel transfer is unproven.** The Swin-L is ImageNet/COCO-pretrained
   RGB; our input is single-channel [0,1] repeated to `num_channels=1` (config:40, main.py:214).
   This risk already exists in the current pipeline; Semi-DETR doesn't add to it, and the whole
   point is to *reduce* the residual sim→real gap with real data.
4. **Label-completeness is an upside, not just noise.** Many current FPs are real peaks the GT
   missed. Teacher pseudo-labels on real frames can legitimately add those positives — so a
   *drop* in apparent precision on the tiny labeled sets is not necessarily bad; weight recall
   (esp. segment) heavily when reading AP.
5. **On-GPU synthetic dataset stays `num_workers=0`.** Do not "optimize" the synthetic loader to
   workers>0 — `FastSimulation` builds cuda tensors (main.py:54-86). The **real** loader is a
   separate CPU dataset (§4a Option A) and may use workers.
6. **Memory.** Two student forwards per step (synthetic + real) plus one teacher forward roughly
   ~1.5-2× the activation memory of the current step, on a 512×1024 Swin-L with 900 queries.
   Keep `unlabeled_batch_size` small (2), lean on `use_checkpoint=True` (config:41) and `--amp`.
   The teacher forward is `no_grad` (cheap on memory).
7. **Eval path and ONNX export unaffected.** `evaluate_giwaxs_ap` (main.py:182-231),
   `PostProcess` (dino.py:639-688), and the deployed `util/postprocessing.py` are untouched;
   the ONNX graph exported later is identical in structure. Semi-supervision changes only the
   weights inside `checkpoint.pth`.

---

## 7. Recommended first experiment (+ follow-up)

### First experiment (smallest test of the hypothesis) — MVP, no hybrid matching
Goal: does adding teacher pseudo-labels from the real corpus move labeled AP (esp. organic /
segment recall) vs the synthetic-only baseline?

Change set: `datasets/real_unlabeled.py` (§4a), `make_pseudo_targets` (§4c),
`train_one_epoch_semi` (§4e, **no** hybrid matching), the main.py wiring (§4e), and a new config
`config/DINO/DINO_4scale_swin_semi.py`.

Config (phase-1 values): `use_semi=True`, `use_ema=True`, `ema_decay=0.999`,
`semi_start_epoch=50`, `ema_epoch=50`, `unsup_loss_weight=2.0`, `unsup_warmup_epochs=5`,
`pseudo_thr_ring=0.4`, `pseudo_thr_seg=0.3`, `strong_aug="v2"`, `semi_geom_flip=False`,
`unlabeled_batch_size=2`, `hybrid_matching=False`.
Warm-start from the current best checkpoint via `pretrain_model_path` (main.py:346-370) so the
teacher isn't sim-random even at epoch 0, OR keep `semi_start_epoch=50` to burn in from scratch.

Launch — mirror `backbone_curation/ssl/run_detector_ssl.sbatch` (auto-resume from
`checkpoint.pth` in `--output_dir`, main.py:325-326 / 482-483; resubmit across 72h sessions):

```bash
#!/bin/bash
#SBATCH --job-name=dino_semi
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_semi-%j.out
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_semi1
export PYTHONPATH=$REPO
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming $OUT_DIR/checkpoint.pth" || echo "[run] fresh start"
cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_semi.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path /path/to/current_best.pth   # optional warm start
```

(Or pass the knobs inline with `--options use_semi=True use_ema=True semi_start_epoch=50 ...`
exactly as the SSL scripts do, e.g. `scripts/DINO_train_submitit_swin.sh:4-8`.)

**Read-out:** compare `exp_ap_41.txt` and `exp_ap_organic.txt` (written per eval at
main.py:453-454) against the synthetic-only baseline over matched epochs. Success = organic AP
up and/or segment recall up without 41-set collapse. Also watch the logged per-class pseudo-box
counts (§6.2).

### Fuller follow-up
1. **Threshold / λ sweep** (phase 2): grid `pseudo_thr_seg ∈ {0.2,0.25,0.3}`,
   `unsup_loss_weight ∈ {1,2,4}`, `semi_start_epoch ∈ {25,50,100}`.
2. **Hybrid matching** (phase 3): `hybrid_matching=True`, `hybrid_t1_epochs≈20`,
   `hybrid_topk_M=4` — the Semi-DETR core; expect the biggest faint-recall gain here.
3. **PLM** (optional): swap the fixed threshold in `make_pseudo_targets` for the 2-component GMM
   miner.
4. **Cross-view consistency** (phase 4): only if 1-3 plateau.
5. Fold the winner into the deployed **ensemble** and re-export ONNX (unchanged export path).

---

## 8. References

- **Semi-DETR** — Zhang et al., *Semi-DETR: Semi-Supervised Object Detection with Detection
  Transformers*, CVPR 2023. arXiv:2307.08095 — https://arxiv.org/abs/2307.08095 (PDF:
  https://arxiv.org/pdf/2307.08095 ; HTML: https://ar5iv.labs.arxiv.org/abs/2307.08095).
  Verified: EMA momentum 0.999; pseudo-label confidence threshold 0.4; unsupervised loss weight
  wu=4 (COCO-Partial)/2.0 (COCO-Full); one-to-many stage T1=60k iters (Partial)/180k (Full);
  matching-cost weights α=1, β=6, focal γ=2; cost-based 2-component GMM PLM (threshold = reliable
  cluster center); cross-view query consistency = `MSE(ô_student, detach(ô_teacher))` on
  RoIAlign→MLP query features, weight wc=1.
- **Sparse Semi-DETR** — Shehzadi et al., CVPR 2024 (Query Refinement Module + Reliable
  Pseudo-Label Filtering). arXiv:2404.01819 — https://arxiv.org/abs/2404.01819.
- **STEP-DETR** — "Super Teacher" for semi-supervised DETR, ICCV 2025.
- **DINO** (detector base) — Zhang et al., *DINO: DETR with Improved DeNoising Anchor Boxes*,
  ICLR 2023. arXiv:2203.03605 — https://arxiv.org/abs/2203.03605.
- **DN-DETR** (denoising queries this repo uses) — Li et al., CVPR 2022. arXiv:2203.01305.
- **DETRs with Hybrid Matching (H-DETR)** — Jia et al., CVPR 2023 (one-to-many auxiliary
  matching). arXiv:2207.13080 — https://arxiv.org/abs/2207.13080.

### Repo anchor points (touch list)
- `main.py:394-458` — epoch loop; add real loader (once, ~main.py:392) + branch to
  `train_one_epoch_semi` (~main.py:404-406).
- `engine.py` — new `train_one_epoch_semi` (model on §4e); reuse EMA step engine.py:92-94.
- `datasets/real_unlabeled.py` — **new**; reuse `backbone_curation/backbone_transform.py`
  (transform + augments) and the reader pattern of
  `backbone_curation/ssl/ssl_dataset.py::SimMIMDataset`.
- `models/dino/dino.py:474-628` — `SetCriterion.forward`: add optional one-to-many matcher swap
  (Route 1, §5) for the unsupervised call.
- `models/dino/matcher.py:98-175` — add `TopkMatcher` (one-to-many) from `SimpleMinsumMatcher`;
  extend `build_matcher` (matcher.py:178-191).
- `config/DINO/DINO_4scale_swin_semi.py` — **new** (inherit swin config; mirror
  `config/DINO/DINO_4scale_swin_ssl.py`); set the §4f knobs.
- `util/utils.py:373-396` (`ModelEma`) — reused as-is (teacher = `.module`, seed via `.set`).
- `util/postprocessing.py:32-87` — reused as-is inside `make_pseudo_targets` (optional class-aware
  NMS on pseudo-boxes).
</content>
</invoke>
