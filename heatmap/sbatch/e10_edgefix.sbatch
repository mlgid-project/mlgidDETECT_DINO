#!/bin/bash
#SBATCH --job-name=hm_e10_edgefix
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/e10_edgefix-%j.out
# E10 candidates (user 2026-10-14): E8 structure (ONE powder phase + ONE oriented crystal per frame, p_ring 0.85) but a bit more rings and spots per structure.
#   E8 reference: rings_cap (4,10), spots_cap (6,25) -> ~5.4 rings / ~5.1 segments per frame.   E6: (6,30)/(25,120) with 1-2 / 2-3 structures.
#   E10a = rings_cap (6,16), spots_cap (10,40)      E10b = rings_cap (8,24), spots_cap (15,60)
# User 2026-10-14: sim frames 1, 3, 6 of the E10 sheets show bright borders along the invalid-pixel mask. Cause (edge_profile.py): _correlate_noise lets the invalid pixels enter the high-pass (+0.05 rim at 1-3 px).
# Fix = bkg_noise_edge_fix=True (mask-aware blurs). Same candidates, same seed, with the fix; plus the brightness-vs-distance profile of E10a/E10b with and without the fix.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
R=$D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5;sq_q_slope=0.6;seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,1);n_oriented=(1,1)"
A="rings_cap=(6,16);spots_cap=(10,40)"
B="rings_cap=(8,24);spots_cap=(15,60)"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)"
$PY -u heatmap/sheet_job.py --out $D/sheets_e10_edgefix --n 6 --seed 5 --init "$INIT;bkg_noise_edge_fix=True" --cfg "E10a_fix:$APP;$A" --cfg "E10b_fix:$APP;$B" 2>&1 | grep -v Warning | tail -3
$PY -u heatmap/sim_pickle.py --out $D/_e10a_nofix.pkl --name e10a --n 100 --init "$INIT" --cfg "$APP;$A" 2>&1 | grep -v "Warning\|^\[" | tail -1
$PY -u heatmap/sim_pickle.py --out $D/_e10a_fix.pkl --name e10afix --n 100 --init "$INIT;bkg_noise_edge_fix=True" --cfg "$APP;$A" 2>&1 | grep -v "Warning\|^\[" | tail -1
$PY -u heatmap/edge_profile.py real41=$R/dets_final.pkl:41 E10a_nofix=$D/_e10a_nofix.pkl:e10a E10a_edgefix=$D/_e10a_fix.pkl:e10afix 2>&1 | grep -v Warn
