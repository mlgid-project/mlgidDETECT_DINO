#!/bin/bash
#SBATCH --job-name=hm_e6_seg
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/e6_seg-%j.out
# E6 segment-count candidates (all with sq_q_slope=0.6 on top of E5; real 41: 24.0 segments/frame, D gave 16.4): ring/segment statistics vs real 41 for (A) E5, (B) + sq_q_slope 0.6 (peak width grows with q), (C) + more segments, (D) both; then contact sheets of D. No training.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
E1="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(1,2);spots_cap=(10,60)"
SL="sq_q_slope=0.6"
RB="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30)"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)"
$PY -u heatmap/ring_stats_sim.py --dump $D/hm_simmim_frozen_ridge_realsimE1b_tf32_seed42_2.80_1.30/dets_final.pkl --n 150 --init "$INIT" \
  --cfg "F_or13_spots30-150:$APP;$SL;$RB;n_oriented=(1,3);spots_cap=(30,150)" \
  --cfg "G_or23_spots25-120:$APP;$SL;$RB;n_oriented=(2,3);spots_cap=(25,120)" \
  --cfg "H_or24_spots25-120:$APP;$SL;$RB;n_oriented=(2,4);spots_cap=(25,120)" \
  --cfg "I_or23_spots40-180:$APP;$SL;$RB;n_oriented=(2,3);spots_cap=(40,180)" 2>&1 | grep -v "Warning\|^\[mosaic\|^\[masks\|^\[realbkg\|^\[realsim" | sed -n '/== F_/,$p'
