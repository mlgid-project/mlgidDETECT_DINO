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

## Other figures copied here (generated inline during the Path A audit / training comparison)
| figure | what it shows |
|---|---|
| `synth_vs_real_hist.png`, `synth_vs_real_corrected.png`, `synth_after_fix.png`, `synth_vs_real_final.png` | Path A synthetic-vs-real pixel-distribution / masking audit (before and after the masking fix). |
| `ap_curves.png` | 2-class run AP curves (organic + 41) vs the old 91-class baseline. |
| `compare_pathA_vs_old.png` | Path A (realistic masking) vs old-sim run — showed no AP gain (Path A reverted). |

## Headline conclusions (see ROADMAP.md for detail)
- AP is lost mostly on **faint (low-visibility), high-q, segment** peaks (recall), plus ~12 confident
  **hallucination** FPs/image (precision; expert-confirmed real hallucinations, not missed labels).
- **Physics post-processing does not apply** (symmetry out-of-frame; FPs are on-ring; not NMS dupes).
- Both failure modes look like a **synthetic→real content gap** → the principled lever is a
  self-supervised backbone on real+sim data (data-gated), not more synthetic tweaks.
