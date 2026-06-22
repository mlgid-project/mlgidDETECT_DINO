# backbone_curation/

End-to-end pipeline to build an **unlabeled real-GIWAXS corpus** from the `DINO_BACKBONE`
tar shards and **self-supervised pretrain** the detector's swin-L backbone on it (Idea A
from `../ROADMAP.md`). Two halves: corpus curation (this dir) and SimMIM SSL (`ssl/`).

Heavy derived data (fingerprints, manifest, the corpus h5, training runs) lives **outside the
repo** in `/mnt/lustre/work/.../DINO_BACKBONE_curation/`; only code + small validation figures
are tracked here. Run everything with the `DINO_GIWAXS` conda env.

## Curation pipeline (order)
| step | script | output |
|---|---|---|
| fingerprint shards | `extract_shards.py` (+`fp_common.py`) | per-tar desc/phash/quality/md5 |
| fingerprint eval frames | `extract_eval.py` | `eval_fp.npz` (41.h5 + organic) |
| **leak check** | `detect_leaks.py` → `iq_match.py` → `combined_confirm.py` (+`validate_sensitivity.py`) | eval-vs-shard match; **result: no leak** |
| profiles for I(q)/I(χ) | `extract_profiles.py` | per-frame radial/azimuthal profiles |
| redundancy + manifest | `dedup.py` → `finalize_manifest.py` | `manifest.tsv`, `keep_keys.txt` (**68,375 → 12,991**) |
| build corpus | `corpus_builder.py` | `backbone_ssl_corpus.h5` (12991×512×1024 uint8, **unprocessed**) |
| detector-match transform | `backbone_transform.py` | `/255 [0,1]` + physics aug (`to_model_input`, `augment`) |

**Leak verdict:** no 41.h5/organic frame is in the shards (provenance clean; phash min-Hamming 12;
calibrated `I(q)`∧`I(χ)` max 0.914 vs same-frame 0.999). The 2-D pixel matcher is unreliable
across conversion pipelines — the trustworthy signal is q-calibrated azimuthal `I(q)` + `I(χ)`.

## SSL (`ssl/`) — SimMIM, not vanilla MAE
Swin is windowed (can't drop tokens), so we use **SimMIM**. The encoder is the *exact* detector
backbone (`swin_L_384_22k`, window 48×6, `in_chans=1`) so weights transfer directly.
- `simmim_model.py` — encoder + mask-token + PixelShuffle decoder, L1 on masked∧valid pixels
- `ssl_dataset.py` — h5 reader, whole-scan val split, scan-balanced sampler
- `train_simmim.py` — AdamW + cosine + warmup, AMP, grad-accum, `--resume`
- `export_backbone.py` — SimMIM ckpt → detector-loadable swin weights (+ load-test)
- `run_simmim.sbatch` / `run_detector_ssl.sbatch` — A100 / 72h, **idempotent auto-resume**

Validated on A100: 197M params, batch 16 ≈ 23 GB, ~22 min/epoch.

## Wiring into the detector
`../config/DINO/DINO_4scale_swin_ssl.py` = base config + `backbone_dir` pointing at the exported
SSL weights (named `swin_large_patch4_window12_384_22k.pth`). Train with that config, then compare
`ap_total` on 41 + organic (per-epoch eval) against the from-scratch baseline (organic ~0.55 / 41 ~0.76).

Full command sequence: see the project chat / `MEMORY` notes.
