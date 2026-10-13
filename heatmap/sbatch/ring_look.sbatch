#!/bin/bash
#SBATCH --job-name=hm_ring_look
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/ring_look-%j.out
# How do rings LOOK at q column >= 781 (where E6 hits only 0.46 of real 41 rings, plain 0.82)? Real 41 vs the legacy simulator (the plain model's training data) vs the E6 realsim, amplitude / width / noise / snr / gap, split at x = 781.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
R=$D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
E6="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(2,3);spots_cap=(25,120);sq_q_slope=0.6"
$PY -u heatmap/dump_dets.py --ckpt $R/final_checkpoint.pth --out $R/dets_legacysim.pkl --sets 41 --sim 150 2>&1 | grep -v Warning | tail -2
$PY -u heatmap/sim_pickle.py --out $R/simE6.pkl --name realsimE6 --n 150 --init "mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)" --cfg "$APP;$E6" 2>&1 | grep -v "Warning\|^\[" | tail -2
$PY -u heatmap/ring_compare.py real41=$R/dets_final.pkl:41 legacysim=$R/dets_legacysim.pkl:sim realsimE6=$R/simE6.pkl:realsimE6 --max_frames 150 --x_split 781 2>&1 | grep -v Warning
