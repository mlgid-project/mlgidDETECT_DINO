#!/bin/bash
#SBATCH --job-name=dump_cmp_dets
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/dump_cmp_dets-%j.out
# Detections (no images, tiny pickles) of three models on the new-box organic + 41 files for the side-by-side figures on colorbox1:
#   unfrozen heatmap seed1 bb1e-5 (3.0,3.0; GT scaled x1.5), NEW DINO baseline A (dino_nb_legacy_lr4e5_1, +-1 sigma = 2,2, GT unscaled),
#   OLD DINO baseline (dino_lr4e5_1, box 2.80,1.30 -> GT scaled x 0.65, y 1.4 via HM_GT_BOX_COEF=2.8,1.3).
# One job for all three (no separate tiny jobs). Pickles: frames in dataset order, img=None, gt already scaled to the model's training box convention.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
W=/mnt/lustre/work/schreiber/szb389/datasets
OUT=$REPO/heatmap/galvani_results/dets_compare_2026-10-26
mkdir -p $OUT
export PYTHONPATH=$REPO HM_DATA_DIR=$W/newbox_visfix_links
cd $REPO
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"
HM_GT_BOX_COEF=3.0,3.0 $PY -u heatmap/dump_dets.py --no_img --ckpt $W/DINO_HEATMAP/hm_simmim_UNFROZEN_ridge_long_tf32_seed1_newbox_3.0_3.0_galvani/final_checkpoint.pth --out $OUT/hm_unfrozen_seed1_bb1e-5_3.0_3.0.pkl --sets organic 41; echo "hm unfrozen rc=$?"
$PY -u heatmap/dump_dets_dino.py --no_img --ckpt $W/DINO_BACKBONE_curation/detector_runs/dino_nb_legacy_lr4e5_1/checkpoint.pth --out $OUT/dino_new_nb_legacy_lr4e5_1_2.0_2.0.pkl; echo "dino new rc=$?"
HM_GT_BOX_COEF=2.8,1.3 $PY -u heatmap/dump_dets_dino.py --no_img --ckpt $W/DINO_BACKBONE_curation/detector_runs/dino_lr4e5_1/checkpoint.pth --out $OUT/dino_old_lr4e5_1_2.80_1.30.pkl; echo "dino old rc=$?"
ls -la $OUT
