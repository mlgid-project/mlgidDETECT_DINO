#!/bin/bash
#SBATCH --job-name=hm_e8_sparse
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/e8_sparse-%j.out
# Sparse frames (user idea: overcrowding may keep the model from learning): E6 appearance (E5 + sq_q_slope 0.6), far fewer rings and peaks.
#   S1 = ONE structure per frame (rings OR segments, p_ring 0.5), rings_cap (3,8), spots_cap (5,25)
#   S2 = rings AND segments in the same frame but few: n_powder (1,1), rings_cap (4,10), n_oriented (1,1), spots_cap (6,25)
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5;sq_q_slope=0.6;seg_wide_frac=0.0"
S1="single_structure=True;p_ring=0.5;rings_cap=(3,8);spots_cap=(5,25)"
S2="p_ring=0.85;n_powder=(1,1);rings_cap=(4,10);n_oriented=(1,1);spots_cap=(6,25)"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)"
$PY -u heatmap/ring_stats_sim.py --dump $D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30/dets_final.pkl --n 150 --init "$INIT" \
  --cfg "S1_single_sparse:$APP;$S1" --cfg "S2_few_both:$APP;$S2" 2>&1 | grep -v "Warning\|^\[mosaic\|^\[masks\|^\[realbkg\|^\[realsim" | sed -n '/== S1/,$p'
$PY -u heatmap/sheet_job.py --out $D/sheets_e8sparse --n 6 --seed 5 --init "$INIT" --cfg "E8S1:$APP;$S1" --cfg "E8S2:$APP;$S2" 2>&1 | grep -v Warning | tail -3
