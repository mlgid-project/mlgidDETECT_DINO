Realsim diagnosis of 2026-10-13 (galvani, inference/simulator only, no training). Fetch with git show origin/heatmap:<file>; put the PNGs next to the other candidate sheets in hm_runs/plots/ (one folder per sheet set) and the .txt in hm_runs/plots/logs/.
- logs/miss_analysis_realsimE1b_final.txt: heatmap/miss_analysis.py on the E1b final model (thr 0.3). The detection dump (dets_final.pkl, 77 MB) stays on galvani in .../DINO_HEATMAP/hm_simmim_frozen_ridge_realsimE1b_tf32_seed42_2.80_1.30/.
- logs/ring_stats-2963235.txt: ring width / q position, E1b simulator vs real 41 GT (heatmap/ring_stats_sim.py).
- logs/ring_q_diag-2963236.txt: where the bank's powder entries end in q (diagnostic; its frame q_max line is the donor frames', not the mask geometries' -- the simulator uses the mask q_max 3.6-7.8).
- logs/lowq_masks-2963243.txt: same with only 4 detector geometries (mask_keep=[0,3,6,7], q_max 3.6-4.0).
- logs/lowq_xshare-2963245.txt: plus ring_x_share=(0.106,0.179,0.257,0.238,0.220) (ring positions follow the real 41 histogram).
- sheets_noise/: E1 / E3 compositions with the correlated background noise (the E1b training setting); sheets_lowq/, sheets_lowq_xshare/: the two new variants. Left = real 41, right = simulated; green = segment box, magenta = ring box.
