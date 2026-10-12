#!/bin/bash
#SBATCH --job-name=hm_sheets
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/sheets-%j.out
# Contact sheets (real 41 frames vs calibrated simulator, with correlated background noise 'm3'): E1 and E3 compositions. Draws frames only (no training).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP/sheets_noise
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;R_w_sq=0.2;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
E1="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(1,2);spots_cap=(10,60)"
E3="single_structure=True;p_ring=0.5;rings_cap=(6,30);spots_cap=(25,75);seg_wide_frac=0.4"
$PY -u heatmap/sheet_job.py --out $OUT --n 6 --seed 5 \
  --cfg "E1noise:$APP;$E1" \
  --cfg "E3noise:$APP;$E3"
