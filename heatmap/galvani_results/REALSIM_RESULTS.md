# Real-background simulator ("realsim") runs: results on record

Written 2026-10-14. Heatmap detector (frozen SimMIM swin-L, 60 epochs, plain recipe: bs 4 x 250 steps, head lr 3e-4, lr x0.1 at epoch 45, seed 42, TF32) trained on the
real-background simulator frames, all on Galvani (a100-galvani). Evaluated on the final checkpoint with `evaluate.py`, headline = **+nms** (class-aware NMS, score floor 0.1, top-225), AP = `ap_total`.
Run folders (args.json, evaluate_final.txt, exp_ap_*.txt, train.log): `hm_runs/.../hm_simmim_frozen_ridge_realsim<E>_tf32_seed42_2.80_1.30`; the text results are also in git on branch `heatmap` under `heatmap/galvani_results/`.
Diagnosis material (sheets, analysis texts, README with every step): `realsim_diagnosis_2026-10-13/` (colorbox: `plots/sheets_*/`, `plots/logs/`).

## Reference points (same evaluation)
- Plain (legacy) simulator, frozen SimMIM, ridge target: organic 0.6057 / 41 0.7259.
- DINO single models (organic / 41): ssl1 0.568 / 0.744, lr 4e-5 0.622 / 0.763, boxconv1 0.588 / 0.752.

## Results (+nms, final checkpoint)
Score > 0.3 columns: recall / ring recall / precision / FP. NN<5 = recall of GT peaks whose nearest neighbour is < 5 px away (organic n=121, 41 n=73), at score > 0.1.

| run | organic AP | 41 AP | organic: recall / ring rec / prec / FP (score>0.3) | 41: recall / ring rec / prec / FP (score>0.3) | NN<5 organic / 41 |
|---|---|---|---|---|---|
| E1b | 0.6059 | 0.5850 | 0.643 / 0.667 / 0.704 / 221 | 0.674 / 0.661 / 0.662 / 578 | 0.479 / 0.411 |
| E5 | 0.6006 | 0.6213 | 0.621 / 0.633 / 0.754 / 165 | 0.704 / 0.743 / 0.715 / 472 | 0.455 / 0.425 |
| E6 | 0.6266 | 0.6104 | 0.632 / 0.667 / 0.740 / 181 | 0.715 / 0.733 / 0.663 / 610 | 0.455 / 0.397 |
| E7 | 0.5581 | 0.5500 | 0.594 / 0.567 / 0.658 / 252 | 0.689 / 0.691 / 0.581 / 833 | 0.446 / 0.425 |
| E8 | 0.5302 | 0.6346 | 0.479 / 0.567 / 0.856 / 66 | 0.582 / 0.626 / 0.846 / 178 | 0.388 / 0.411 |
| E9 | 0.6316 | 0.6154 | 0.646 / 0.600 / 0.766 / 161 | 0.738 / 0.767 / 0.675 / 597 | 0.446 / 0.425 |
| E10 | 0.5587 | 0.6019 | 0.558 / 0.600 / 0.751 / 151 | 0.628 / 0.674 / 0.781 / 295 | 0.372 / 0.411 |
| E11a | 0.6083 | 0.5707 | 0.645 / 0.533 / 0.712 / 213 | 0.671 / 0.681 / 0.673 / 547 | 0.413 / 0.397 |
| E11b | 0.6120 | 0.5901 | 0.670 / 0.667 / 0.687 / 249 | 0.713 / 0.737 / 0.636 / 684 | 0.413 / 0.438 |

Every number in the table is parsed from the `[hm+nms]` blocks of each run folder's `evaluate_final.txt`, which also holds the rest (recall seg/ring, preds, NN buckets at both score cut-offs, AP by GT confidence, operating points).

Best realsim run per set: organic E9 0.6316 (E6 0.6266), 41 E8 0.6346 (E10 0.5587 / 0.6019, E11a 0.6083 / 0.5707, E11b 0.6120 / 0.5901). All realsim runs are **below the plain simulator on 41 by 0.09-0.18** (0.7259) and on organic E6 / E9 are slightly above the plain simulator (0.6266 / 0.6316 vs 0.6057), E8 and E7 below.
Never reached: DINO's 41 (0.744-0.763), best realsim 41 is 0.6346.
Single seed (42) everywhere; no realsim replicates were run (E5 seeds 2 and 3 were prepared, then cancelled by the user). The unfrozen runs of the same recipe swung by about 0.07 in organic AP from epoch to epoch (0.540-0.606), so small differences between single realsim runs are not separable.

## What each run changed (all builds on the previous appearance settings; every option default-off in `realsim/realbkg_simulation.py`)
- **Base appearance (E1b/E5):** mosaic donor backgrounds (90 bare-Si frames, fresh crop + PCA envelope per frame), hkl-bank peaks, 4 worker processes, amplitude_mode fitted, ring_gain 0.4, ring_contrast_min 0.3, ring_chi_texture 0.05, ring_chi_corr (8,64), seg_sc_scale 0.6, ring_sq_scale 1.0, ring_w_coef 2.2, ring_no_elongate, ring_cap 4.0, ring_chi_floor 0.5, ring_follow_bkg, width_jitter (0.2,0.832), bkg_noise_blur (0,1.2), bkg_noise_power 1.5, seg_wide_frac 0, p_ring 0.85, n_powder (1,2), rings_cap (6,30), n_oriented (1,2), spots_cap (10,60). **E1b** = this; its noise correction (bkg_noise_blur / power) cut the 41 false positives of the first version E1.
- **E5:** + `mask_keep=[0,3,6,7]` (the four detector masks with q_max 3.6-4.0, matching real 41's q range 2.7-3.5) + `ring_x_share=(0.106,0.179,0.257,0.238,0.220)` (ring positions across the five q-column fifths as in real 41). Fixed the ring positions.
- **E6:** E5 + `sq_q_slope=0.6` (peak width grows with q), `n_oriented=(2,3)`, `spots_cap=(25,120)` -> ~20.7 segments and ~15 rings per frame (real 41: 24 / 17).
- **E7:** E6 + `bkg_flat_q=1.0` (flatten the background fall-off along q) + `ring_rel_shuffle=0.5` (decouple ring brightness from q). Image statistics now match real 41 (flat mean/noise, high-q ring amplitude 0.226 vs 0.234), but **worse** (organic 0.5581, 41 0.5500, 10.4 ring FPs per 41 frame vs 3.1 for E6). Matching image statistics alone does not help.
- **E8:** E6 appearance with sparse frames: n_powder (1,1), rings_cap (4,10), n_oriented (1,1), spots_cap (6,25) -> ~5.4 rings / ~5.1 segments per frame. Best 41 (0.6346) but worst organic (0.5302): fewer false positives (41 FP 178) but recall drops (organic recall 0.479, 41 0.582). Sparser frames help a little on 41, overcrowding is not the main cause.
- **E9:** E6 + `mask_wedge_only_frac=0.2` (20 % of frames use only the generated missing wedge, row-wise filled between first/last valid pixel, no detector gap bands). First change that moved the high-q ring hit rate (+0.11 vs E6), best organic (0.6316), 41 0.6154.

## Findings from the diagnosis (images + analysis, see README in `realsim_diagnosis_2026-10-13/`)
- High-q real rings (q column >= 781): missed rings are NOT detected at all (nothing within 10 px): E1b 36 %, E5 38 %, E6 43 %, E7 54 % of missed ones; not detected-but-unmatched (<= 1 %). Ring hit rate at q column >= 781 on 41: E5 0.53, E6 0.46, E7 0.44, E8 0.45, E9 0.57; plain simulator 0.82. Not box shape, ring count, width, brightness or background level.
- Real 41 frames have no detector gap bands (every simulated frame had 2-4 curved ones), a narrowing tip at high q (63-92 % invalid in the last bins), q range 2.7-3.5 A^-1.
- Bright border along the invalid-pixel mask (+0.05 rim at 1-3 px) traced to `_correlate_noise` letting invalid pixels into the high-pass; fixed by `bkg_noise_edge_fix=True` (mask-aware blurs), verified flat (0.423-0.438 at 0-12 px vs 0.52 rim without; real 41 0.44-0.48). Used from E10 on.
- Hypotheses that failed: `ring_rel_shuffle` and `bkg_flat_q` (E7 worse than E6 on both sets).

## Open / next
- **E10 (job 2966092, 2026-10-14, 60 epochs):** E8 structure (one powder phase + one oriented crystal) with a bit more rings and spots (rings_cap (6,16), spots_cap (10,40) -> 8.6 rings / 8.4 segments per frame) and `bkg_noise_edge_fix=True`. Final +nms: organic 0.5587, 41 0.6019 (E8 0.5302 / 0.6346; E9 0.6316 / 0.6154). Organic improved over E8 (+0.03), 41 fell (-0.03); score>0.3 recall rose on both sets (organic 0.479 -> 0.558, 41 0.582 -> 0.628) and precision fell (organic 0.856 -> 0.751, 41 0.846 -> 0.781). Organic AP during training ranged 0.52-0.57 after epoch 24 (epoch 54: 0.5680). Candidate E10b (rings_cap (8,24), spots_cap (15,60), 11.5 / 12.2 per frame) not trained.
- **E11a / E11b (jobs 2966880 / 2966881, 2026-10-14, 60 epochs):** E6 settings + edge fix + wedge-only masks in 100 % (E11a) / 50 % (E11b) of the frames, to test whether the detector gap bands cause the high-q ring misses. Final +nms: E11a organic 0.6083, 41 0.5707; E11b organic 0.6120, 41 0.5901 (E6 0.6266 / 0.6104, E9 with 20 % 0.6316 / 0.6154). **Result: more wedge-only frames did not help, 41 got worse (100 % 0.5707, 50 % 0.5901, 20 % 0.6154, 0 % 0.6104), and 41 false positives at score > 0.3 rose (E9 597, E11b 684, E11a 547 with lower precision).** Hypothesis 'the gap bands cause the high-q ring misses' is not supported by the AP; the high-q ring hit rate of these two runs was not measured.
- To discuss afterwards: ring boxes sitting inside the dark invalid tip at the far right of some sim frames (seen in E10 sheets, not yet investigated).
- Later, user ideas: rebuild the hkl bank/peak lists from the pygid sim and add in-plane crystal rotation, once AP is higher.
