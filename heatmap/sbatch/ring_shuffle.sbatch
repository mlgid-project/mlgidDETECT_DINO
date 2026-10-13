#!/bin/bash
#SBATCH --job-name=hm_ring_shuf
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/ring_shuffle-%j.out
# E6 simulator with ring_rel_shuffle = 0.5 / 1.0 (ring brightness decoupled from q): ring appearance at q column >= 781 vs real 41 (amplitude 0.234, flank noise 0.198, chi cv 0.066), + contact sheet of 1.0.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
R=$D/hm_simmim_frozen_ridge_realsimE6_tf32_seed42_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5"
E6="seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(2,3);spots_cap=(25,120);sq_q_slope=0.6"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220)"
for S in 0.5 1.0; do
  $PY -u heatmap/sim_pickle.py --out $R/simE6_shuf$S.pkl --name shuf$S --n 150 --init "$INIT" --cfg "$APP;$E6;ring_rel_shuffle=$S" 2>&1 | grep -v "Warning\|^\[" | tail -1
done
$PY -u heatmap/ring_compare.py real41=$R/dets_final.pkl:41 shuf0.5=$R/simE6_shuf0.5.pkl:shuf0.5 shuf1.0=$R/simE6_shuf1.0.pkl:shuf1.0 --max_frames 150 --x_split 781 2>&1 | grep -v "Warning\|nanmean" | grep -E "^==|amplitude|noise|chi variation|snr"
$PY -u heatmap/sheet_job.py --out $D/sheets_e7shuf --n 6 --seed 5 --init "$INIT" --cfg "E7shuf1:$APP;$E6;ring_rel_shuffle=1.0" 2>&1 | grep -v Warning | tail -1
