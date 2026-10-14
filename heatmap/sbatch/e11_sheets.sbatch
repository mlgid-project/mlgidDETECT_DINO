#!/bin/bash
#SBATCH --job-name=hm_e11_sheets
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/e11_sheets-%j.out
# E10 candidates (user 2026-10-14): E8 structure (ONE powder phase + ONE oriented crystal per frame, p_ring 0.85) but a bit more rings and spots per structure.
#   E8 reference: rings_cap (4,10), spots_cap (6,25) -> ~5.4 rings / ~5.1 segments per frame.   E6: (6,30)/(25,120) with 1-2 / 2-3 structures.
#   E10a = rings_cap (6,16), spots_cap (10,40)      E10b = rings_cap (8,24), spots_cap (15,60)
# E11 settings (E6 + wedge-only masks + edge fix): contact sheets for the user, 100 % and 50 % of frames with the wedge-only mask.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
D=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
export PYTHONPATH=$REPO
cd $REPO
APP="amplitude_mode='fitted';intensity_decades=None;ring_gain=0.4;ring_contrast_min=0.3;ring_chi_texture=0.05;ring_chi_corr=(8.0,64.0);seg_sc_scale=0.6;ring_sq_scale=1.0;ring_w_coef=2.2;ring_no_elongate=True;ring_cap=4.0;ring_chi_floor=0.5;ring_follow_bkg=True;width_jitter=(0.2,0.832);bkg_noise_blur=(0.0,1.2);bkg_noise_power=1.5;sq_q_slope=0.6;seg_wide_frac=0.0;p_ring=0.85;n_powder=(1,2);rings_cap=(6,30);n_oriented=(2,3);spots_cap=(25,120)"
INIT="mask_keep=[0,3,6,7];ring_x_share=(0.106,0.179,0.257,0.238,0.220);bkg_noise_edge_fix=True"
$PY -u heatmap/sheet_job.py --out $D/sheets_e11 --n 6 --seed 5 --init "$INIT;mask_wedge_only_frac=1.0" --cfg "E11a:$APP" 2>&1 | grep -v Warning | tail -2
$PY -u heatmap/sheet_job.py --out $D/sheets_e11 --n 6 --seed 5 --init "$INIT;mask_wedge_only_frac=0.5" --cfg "E11b:$APP" 2>&1 | grep -v Warning | tail -2
