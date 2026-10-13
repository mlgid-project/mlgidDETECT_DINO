#!/bin/bash
#SBATCH --job-name=hm_ring_stats
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/ring_stats-%j.out
# Ring box width / position of the E1b simulator setting vs real 41 ground truth (needs the E1b detection dump for the 41 GT).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
E1="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(1,2);spots_cap=(10,60)"
$PY -u heatmap/ring_stats_sim.py --dump $D/hm_simmim_frozen_ridge_realsimE1b_tf32_seed42_2.80_1.30/dets_final.pkl --n 150 --cfg "E1b:$APP;$E1" 2>&1 | grep -v Warning
