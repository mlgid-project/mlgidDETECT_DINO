#!/bin/bash
#SBATCH --job-name=hm_calib_noise
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/calibrate_sim-%j.out
# Calibration of the correlated background noise (realsim option bkg_noise_blur) against real 41, with the E1 appearance settings. Draws frames only (no training).
# Prints, per setting, simulated vs real quartiles of the lag-1 autocorrelation (BACKGROUND block), ring and segment appearance.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
B="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;seg_wide_frac=0.0;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;R_w_sq=0.2;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True"
$PY -u heatmap/calibrate_sim.py --n 40 --seed 8 \
  --cfg "n0:$B" \
  --cfg "n1:$B;bkg_noise_blur=(0.3,0.9)" \
  --cfg "n2:$B;bkg_noise_blur=(0.4,1.2)" \
  --cfg "n3:$B;bkg_noise_blur=(0.5,1.5)" \
  --cfg "n4:$B;bkg_noise_blur=(0.6,1.8)"
