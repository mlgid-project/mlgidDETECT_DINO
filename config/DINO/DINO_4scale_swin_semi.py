# Semi-supervised (Semi-DETR MVP) config: the 2-class ring/segment DINO baseline plus
# mean-teacher pseudo-labeling on the 13k real unlabeled GIWAXS corpus.
# Design + rationale: docs/SEMI_DETR_INTEGRATION.md (phases 0-1; phase-3 hybrid matching
# is wired but OFF). TRAINING-ONLY change: the exported ONNX inference graph is unaffected.
#
# Backbone init is from-scratch (base config). To start from the SimMIM SSL backbone
# instead, switch _base_ to ['DINO_4scale_swin_ssl.py']; to warm-start the FULL detector
# (recommended, shortens/skips burn-in), pass --pretrain_model_path <checkpoint.pth>.
_base_ = ['DINO_4scale_swin.py']

# --- semi-supervised (Semi-DETR MVP) ---
# v2 settings after the run-1 collapse (job 2650922, epochs 50-159; see MODIFICATIONS.md
# phase I): the young from-scratch teacher was UNDER-confident on rings in the real domain,
# thr_ring=0.4 starved ring pseudo-labels within ~8 epochs (class collapse, 41 AP
# 0.63 -> 0.10), then segment hallucinations exploded (~200/frame). v2 = warm start from a
# mature checkpoint + swapped thresholds + per-image cap + halved lambda.
# v3 = FROZEN-TEACHER DIAGNOSTIC after run-2 drift (job 2651525, dino_semi2): counters
# stayed healthy (no collapse) but ALL AP curves declined monotonically for ~50 semi epochs
# (teacher organic 0.55->0.41, 41 0.74->0.66) with slow ring-count inflation (19->34/batch)
# = confirmation-bias drift through the EMA loop. v3 severs the loop: ema_decay=1.0 freezes
# the teacher as (settled) ssl1, so pseudo-labels cannot drift by construction; lambda halved
# to 0.5. Interpretation: if AP STILL declines -> the pseudo-label signal itself is harmful
# at these thresholds (next: drop box-regression on pseudo-labels / quality weighting); if AP
# holds or improves -> the feedback loop was the culprit (next: slower EMA / re-anchoring).
#
# v3 VERDICT (job 2652578, dino_semi3, 100 semi epochs): loop severed cleanly (counters flat
# 18/18 for 100 ep) and no collapse — but organic sat ~0.51 then eroded to ~0.48 vs the 0.573
# warm-start point (41 ~0.72 vs 0.746). Even a good frozen teacher's pseudo-labels are a net
# TAX at these settings -> the harmful half is the signal itself, most plausibly the BOX
# COORDINATES (confidence vouches for what, not where — Unbiased Teacher, ICLR 2021).
# v4 = CLASSIFICATION-ONLY unsupervised loss: unsup_cls_only=True drops L1/GIoU on
# pseudo-labels (boxes still steer the Hungarian matching, they carry no coordinate gradient).
# Frozen teacher + lambda 0.5 kept (one variable at a time). If organic holds ~0.57 or climbs,
# reintroduce geometry via quality weighting (SSRT-DETR S3.5); if it still erodes, record the
# MVP as a documented negative (3 clean ablations: loop, lambda, loss content).
use_semi = True
unlabeled_h5 = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5'
unlabeled_batch_size = 2
unlabeled_workers = 4      # real dataset is CPU-side (datasets/real_unlabeled.py) -> workers OK
strong_aug = 'v2'          # photometric-only strong aug for the student view ('v1'|'v2')
semi_geom_flip = False     # shared chi-flip of both views (MVP: off -> no box transform anywhere)
semi_start_epoch = 5       # SHORT re-stabilization burn-in — ASSUMES full-detector warm start
                           # via --pretrain_model_path (run_detector_semi.sbatch); use ~50 if
                           # training from scratch (but see run-1 collapse note above)
unsup_loss_weight = 0.5    # lambda_max (run 2 drifted at 1.0 — half the gradient budget was
                           # pseudo-label noise)
unsup_warmup_epochs = 5    # linear lambda ramp over the first epochs of the semi phase
pseudo_thr_ring = 0.30     # LOWER than seg: the teacher is under-confident on rings in the
                           # real domain (run-1 lesson — 0.4 starved rings entirely)
pseudo_thr_seg = 0.35      # raised from 0.3 (segment hallucinations drove the run-1 explosion)
pseudo_max_per_img = 30    # hard top-k cap per frame after class-aware NMS
unsup_cls_only = True      # v4: drop L1/GIoU on pseudo-labels, keep only classification
                           # (see run-3 verdict above; Unbiased Teacher move)

# --- EMA teacher (required by use_semi; ModelEma already existed in util/utils.py) ---
use_ema = True
ema_decay = 1.0            # FROZEN teacher (update is a no-op): teacher = student hard-seeded
                           # at semi_start_epoch (= warm-started ssl1 after 5 settling epochs),
                           # then never changes. Restore 0.999 for mean-teacher dynamics.
ema_epoch = 5              # EMA steps start here (= semi_start_epoch; teacher is hard-seeded
                           # from the student at the semi boundary, engine.train_one_epoch_semi)

# --- phase 3: stage-wise hybrid matching (one-to-many -> one-to-one) — wired, default OFF ---
hybrid_matching = False
hybrid_t1_epochs = 20      # one-to-many epochs after semi_start_epoch, then one-to-one
hybrid_topk_M = 4          # queries supervised per pseudo-box during one-to-many
