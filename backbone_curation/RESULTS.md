# SSL-backbone experiments — consolidated results (2026-06)

Outcome record of the self-supervised backbone effort (ROADMAP.md "Idea A"): SimMIM
pretraining of the swin-L 48x6 backbone on the curated **12,991-frame real unlabeled polar
corpus** (`backbone_ssl_corpus.h5`, curated from 68,375 raw frames; eval-leak check CLEAN —
see `README.md` in this directory), then supervised detector training on synthetic data with
that backbone init. All experiments judged **only** by labeled `ap_total` on the 41 + organic
eval sets (per-epoch pygid eval, pre/post matched to the deployed mlgidDETECT); SSL val loss
was never used as a success metric. Strata: `ap_high/med/low` = GT visibility 1.0 / 0.5 / 0.1.

## Bottom line

1. **Round-1 SSL backbone (`ssl1`) beats from-scratch on organic (+0.032), ties on 41** — the
   SSL init is the best *single* model and converged ~120 epochs earlier.
2. **All three refinements were NEGATIVE** (recipe v2, stage-freezing, 5-scale). Do not
   re-run them. The plain round-1 recipe at 4-scale is the deliverable.
3. **The deployed best is the ssl1 + baseline ENSEMBLE** (the two models are complementary):
   **organic 0.605 / 41 0.780**, improving every stratum incl. faint peaks. Deployment doc:
   `ENSEMBLE_DEPLOY.md`.

## Final comparison table (best ap_total per run)

| run | backbone init | scales | 41 | organic | verdict |
|---|---|---|---|---|---|
| **ensemble (ssl1+baseline)** | — | 4 | **0.780** | **0.605** | **DEPLOYED BEST** (`ensemble_eval.py`) |
| **ssl1** (round-1 SSL) | simmim1 | 4 | 0.762 @ep258 | **0.586** @ep238 | best single; WINNER |
| baseline (from-scratch) | random | 4 | **0.768** @ep338 | 0.554 @ep360 | wins 41 by noise-level margin |
| ssl2 (recipe v2) | simmim2 | 4 | 0.751 | 0.578 | ✗ below ssl1 on both |
| frozen (stages 0-2 frozen) | simmim1 | 4 | 0.754 | 0.568 | ✗ −0.018 organic vs ssl1 |
| ssl5 | simmim1 | 5 | 0.731 | 0.566 | ✗ worst; erases SSL edge |
| 5scratch | random | 5 | 0.743 | 0.572 | ✗ 5-scale hurts even from scratch |

Baseline run: `/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434`.
SSL runs: `/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/`
(`dino_ssl1`, `dino_ssl2`, `dino_ssl1_frozen`, `dino_ssl5`, `dino_5scale_scratch`).
Plot scripts (this directory): `plot_compare.py` (baseline vs ssl1), `plot_compare_detectors.py`
(ssl1 vs ssl2), `plot_state.py`.

## Round 1 — simmim1 → `ssl1` (WINNER)

- SimMIM pretrain (`ssl/train_simmim.py`, recipe v1: mask 0.6 / 32px grid, vflip+gamma+noise
  aug): ran to ep180; reconstruction converged by ~ep40 (val L1 flat at 0.1096). Detector
  backbone was initialized from the **ep77 snapshot** — SSL val loss beyond convergence did
  not matter.
- Detector `dino_ssl1` (`config/DINO/DINO_4scale_swin_ssl.py` = base + `backbone_dir`; NO
  freezing, NO extra scales): **organic 0.586 vs 0.554 from-scratch (+0.032, and reached
  baseline-peak AP ~120 epochs earlier); 41: 0.762 vs 0.768 (−0.006, within noise).**
- The feared late-epoch wash-out of the SSL init did NOT occur on organic.
- Note: baseline is *genuinely random init* (base config has no `backbone_dir`); ImageNet
  weights are not loadable into the elongated 48x6 window anyway, so from-scratch is the
  correct control.

## Refinements — all three NEGATIVE (2026-06-19)

- **Recipe v2 (`ssl2`: mask 0.70 + richer aug_v2): ✗** marginally below v1 on both sets
  (0.751 / 0.578). Conclusion: the SSL-recipe lever is **saturated on this corpus** — better
  masking/aug cannot buy more; only more *diverse data* could (see "Untried levers").
- **Stage-freezing (`frozen`: patch_embed + stages 0-1 frozen during detector training): ✗**
  0.754 / 0.568 (−0.018 organic vs ssl1). Round 1 showed no wash-out, so freezing only
  removed beneficial adaptation. Configs kept for reference:
  `DINO_4scale_swin_ssl_frozen.py`, `ssl/run_detector_ssl_frozen.sbatch`.
- **5-scale (`ssl5` / `5scratch`): ✗ and it ERASES the SSL advantage.** 5-scale is worse than
  4-scale for BOTH inits, and the organic SSL edge collapses (difference-of-differences:
  +0.032 at 4-scale → −0.006 at 5-scale). The extra stride-4 P2 level is NOT a route to
  faint-peak recall in this setup. Configs: `DINO_5scale_swin_ssl.py`, `DINO_5scale_swin_scratch.py`.

## Ensemble (deployed best)

`ssl1` and the baseline are **complementary** (baseline better on 41, ssl1 much better on
organic). Detection-level fusion — pool each model's top-225, then the production class-aware
NMS — beats both on BOTH sets: **organic 0.605** (+0.037 vs best single), **41 0.780**
(+0.022), and improves every stratum including faint peaks (organic ap_low 0.384→0.424,
41 ap_low 0.620→0.652). Reproduce: `ensemble_eval.py --eval_file <41|organic>` (GPU; builds
both swin-L models; uses each run's `checkpoint.pth` = LAST epoch). Deployment/export:
`ENSEMBLE_DEPLOY.md` (use `export_onnx_ensemble.py`).

## Semi-supervised pseudo-labeling (2026-07) — NEGATIVE

The follow-on Semi-DETR-style mean-teacher experiment series (4 runs, `detector_runs/dino_semi1-4`)
concluded as a **documented negative**: MVP hard-pseudo-label training on the 13k real corpus ends
below its own warm-start AP at every ablated operating point (teacher maturity / EMA loop / λ /
loss content each isolated). Full autopsies + the label-incompleteness confound + what a revisit
should try first: **`MODIFICATIONS.md` phase I**. The ssl1+baseline ensemble above remains the
deployed best.

## Cross-DINO portable subset (2026-07)

Porting the two ONNX-safe pieces of Cross-DINO (arXiv:2505.21868) as fine-tunes from `ssl1`, gating
on organic AP (`docs/CROSS_DINO_INVESTIGATION.md`).

- **Exp A — Boost Loss + Category-Size soft label: DECLINED (negative).** β-sweep is monotonically
  negative: β=1.0 (`dino_boost1`) organic **0.42**, β=0.5 (`dino_boost2`) organic **0.525**, β=0
  (≡ plain focal ≡ ssl1) organic **0.586**. Our boxes are uniformly tiny-cs (elongated), so the
  size-weighting has no diversity to exploit and is pure tax — best operating point is "off". Details:
  `MODIFICATIONS.md` phase J.
- **Exp B — CCTM feature-enrichment module: DECLINED (trustworthy null).** `dino_cctm2` (CCTM @10× LR,
  identity-at-init) converged at organic **0.567** / 41 **0.760** — no lift over ssl1 0.586. A
  mechanistic diagnostic (cctm2 vs ssl1, 817 organic GT peaks) shows why: CCTM lifts recall only on
  already-easy peaks (bright 0.69→0.73, ring 0.83→0.88) and leaves our ceiling **untouched** — faint
  0.33→0.33, high-q 0.44→0.44 — at a small precision cost. It reinjects backbone detail, which cannot
  add faint-peak sensitivity that is not in the backbone to begin with. Details: `MODIFICATIONS.md`
  phase K.
- **Cross-DINO overall: investigated-and-declined.** Both portable modules (Boost, CCTM) failed the
  organic gate → the Strip-MLP backbone is **not** ported. Diagnostic take-home: the ceiling is
  *sensitivity to faint / high-q peaks*, which detail-routing architectures do not address. Next lever:
  Co-DETR/Co-DINO training-only aux heads (adds sensitivity, zero inference cost).

## Diagnosis carried forward

`ssl1`'s dominant remaining failure is **faint peaks**: organic ap_low ~0.42 vs ap_high
~0.69 (41: 0.63 vs 0.88). The detector trains on 100% synthetic data whose pixel
distribution is measurably too bright/spiky vs real (`diagnostics/synth_vs_real_hist.png`).
This diagnosis motivated the two follow-on tracks:

- **Semi-supervised pseudo-labeling on the same 13k real corpus** (Semi-DETR-style
  mean-teacher) — implemented, running: see `docs/SEMI_DETR_INTEGRATION.md` and
  `MODIFICATIONS.md` phase I. Natural combination once validated: SSL backbone init + semi
  training (`_base_ = ['DINO_4scale_swin_ssl.py']` in the semi config).
- **Corpus diversity** — the only untried *SSL* lever. The 68k raw corpus contains only
  ~4.6k genuinely distinct frames (giant near-static in-situ scans); recipe knobs are
  saturated, so further SSL gains require NEW scans, not new hyperparameters. Idea pool:
  `ssl/RECIPE_v3.md`.
