# diagnostics/

Read-only analysis scripts (and their output figures) used to diagnose the 2-class ring/segment
model. None of these change the model or training — they load the best checkpoint and analyze it.
Full write-up of the findings is in `../ROADMAP.md`; change log in `../MODIFICATIONS.md`.

## How to run
From the repo root, with the `DINO_GIWAXS` conda env, e.g.:
```bash
PYTHONPATH=. /home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python diagnostics/diagnose_C.py
```
Each script writes its figure next to itself (in this directory). Hardcoded inputs at the top of
each script: `CKPT` (best checkpoint) and `DSET` (organic eval h5) — edit if paths change.

## Scripts & figures
| script | what it does | figure |
|---|---|---|
| `diagnose_C.py` | Where AP is lost: recall by visibility / ring-vs-segment / q-position, and FP rate + score distribution (organic, score>0.3). | `diagnose_C.png` |
| `diagnose_rings.py` | Tests the ring/physics hypothesis: I(q) percentile at each detection's q and FP q-distance to nearest real peak. Showed FPs are ON rings → ring-aware rejection won't work. | `diagnose_rings.png` |
| `sweep_nms.py` | Sweeps segment-NMS IoU (0.4→0.1); shows AP is flat → FPs are not NMS duplicates. (Prints only, no figure.) | — |
| `viz_fp.py` | Overlays GT (green) / matched-TP (blue dashed) / unmatched-FP (red) on the images for expert review. Confirmed the red FPs are genuine hallucinations. | `viz_fp.png` |
| `hires_probe.py` | Matched-operating-point faint/high-q recall probe, hires 512×2048 vs ssl1 512×1024 (phase R gate). Sweeps the score threshold per model and compares at equal detection count. (Prints only.) | — |
| `label_completeness.py` | Are the deployed ensemble's FPs real unlabeled peaks? Standard vs label-adjusted precision (phase Q). | `label_completeness.png` |
| `prominence_probe.py` | Topographic prominence (0-dim superlevel-set persistence) of every labeled peak vs whether the ensemble found it. **Showed the misses are NOT contrast-limited** (separation AUC 0.489 organic). Runs on organic + 41. | `prominence_probe.png` |
| `nearmiss_probe.py` | Replays the ensemble's stages (900 queries → top-225 → NMS → score>0.3) and buckets every miss by where it was lost. **Showed 77% of misses have a model response and 84.5% sit within 8 q-px of a detected peak → a χ-separation problem.** (Prints only.) | — |

## Other figures copied here (generated inline during the Path A audit / training comparison)
| figure | what it shows |
|---|---|
| `synth_vs_real_hist.png`, `synth_vs_real_corrected.png`, `synth_after_fix.png`, `synth_vs_real_final.png` | Path A synthetic-vs-real pixel-distribution / masking audit (before and after the masking fix). |
| `ap_curves.png` | 2-class run AP curves (organic + 41) vs the old 91-class baseline. |
| `compare_pathA_vs_old.png` | Path A (realistic masking) vs old-sim run — showed no AP gain (Path A reverted). |

## Headline conclusions (see ROADMAP.md for detail)

> **Updated 2026-08-18 by `prominence_probe.py` + `nearmiss_probe.py` (MODIFICATIONS.md S/T).** The
> "faint peaks" reading below is WRONG. Prominence does not predict which peaks are missed
> (AUC 0.489 on organic; detection rate flat across prominence deciles), 77% of misses have a model
> response somewhere in the pipeline, and 84.5% sit within 8 q-px of a peak the model DID detect —
> median χ-separation 3.9 px against ~8 px-tall boxes. The ceiling is **peak separation along χ**,
> not sensitivity. The FP claim below is also superseded: phase Q showed most FPs are real unlabeled
> peaks. Read the two bullets that follow as historical.

- AP is lost mostly on **faint (low-visibility), high-q, segment** peaks (recall), plus ~12 confident
  **hallucination** FPs/image (precision; expert-confirmed real hallucinations, not missed labels).
- **Physics post-processing does not apply** (symmetry out-of-frame; FPs are on-ring; not NMS dupes).
- Both failure modes look like a **synthetic→real content gap** → the principled lever is a
  self-supervised backbone on real+sim data (data-gated), not more synthetic tweaks.
