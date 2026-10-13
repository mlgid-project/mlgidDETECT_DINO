#!/bin/bash
#SBATCH --job-name=hm_edge2
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/edge_profile2-%j.out
# Bright rim along invalid pixels in simulated frames? mean image value vs distance to the nearest invalid pixel: real 41, E6 sim (with correlated noise), same sim without correlated noise (E1-style), legacy sim.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
R=$D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
APP0="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832)"
E6="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(2,3);spots_cap=(25,120);sq_q_slope=0.6"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)"
APPC="$APP0;bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
$PY -u heatmap/sim_pickle.py --out $R/simE6_edgefix.pkl --name edgefix --n 100 --init "$INIT;bkg_noise_edge_fix=True" --cfg "$APPC;$E6" 2>&1 | grep -v "Warning\|^\[" | tail -1
$PY -u heatmap/edge_profile.py real41=$R/dets_final.pkl:41 realsimE6_noisecorr=$R/simE6.pkl:realsimE6 E9fill_masks=$R/simE6_fill.pkl:fill realsim_edgefix=$R/simE6_edgefix.pkl:edgefix 2>&1 | grep -v Warn
