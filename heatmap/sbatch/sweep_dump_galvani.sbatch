#!/bin/bash
#SBATCH --job-name=sweep_dump
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/work/schreiber/szb389/datasets/latent_sweep/sweep_dump-%j.out
# Region sweep, Galvani part (2026-11-16): the detections of every complete 3.0,3.0 model whose checkpoint lives here (unfrozen ViT-H+ 9.6 GB, unfrozen SimMIM 2.3 GB, the DINO 3,3 run 2.5 GB)
# on the 49 labelled frames (organic 8 + 41 41), labels scaled to 3.0,3.0. One small pickle per model (no images) in $OUT/dets/; the checkpoints stay here, colorbox1 copies the pickles only.
# Inference only, about 1-3 min per model. Skips models that already have a pickle, so a timeout is fixed by resubmitting. Code: the DINO repo on Galvani (branch dev, has heatmap/dump_dets.py and dump_dets_dino.py).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
RUNS=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
DINO=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_nb_3.0_3.0_lr4e5_1
OUT=/mnt/lustre/work/schreiber/szb389/datasets/latent_sweep
export PYTHONPATH=$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF=3.0,3.0
export HM_DINOV3_REPO=/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_repo
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH GIWAXS_BANK
mkdir -p $OUT/dets
cd $REPO
echo "host $(hostname) job $SLURM_JOB_ID repo branch $(git branch --show-current) $(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ok=0; bad=0
for d in $RUNS/hm_*newbox_3.0_3.0*galvani; do
  run=$(basename $d)
  [ -f $d/evaluate_final_gtscaled.txt ] && [ -f $d/args.json ] && [ -f $d/checkpoint.pth ] || continue
  f=$OUT/dets/$run.pkl
  [ -f $f ] && { echo "have $run"; continue; }
  if $PY -u heatmap/dump_dets.py --ckpt $d/checkpoint.pth --out $f.tmp --no_img > $OUT/dets/$run.log 2>&1 && [ -s $f.tmp ]; then mv $f.tmp $f; ok=$((ok+1)); echo "ok   $run"; else rm -f $f.tmp; bad=$((bad+1)); echo "FAIL $run: $(tail -n 2 $OUT/dets/$run.log | tr '\n' ' ' | cut -c1-200)"; fi
done
# DINO 3,3 (epoch 379): pure-PyTorch MSDeformAttn path, no compiled ops needed
f=$OUT/dets/dino_nb_3.0_3.0_lr4e5_1.pkl
if [ -f $f ]; then echo "have dino"; elif $PY -u heatmap/dump_dets_dino.py --ckpt $DINO/checkpoint.pth --out $f.tmp --box_coef 3.0,3.0 > $OUT/dets/dino.log 2>&1 && [ -s $f.tmp ]; then mv $f.tmp $f; ok=$((ok+1)); echo "ok   dino"; else rm -f $f.tmp; bad=$((bad+1)); echo "FAIL dino: $(tail -n 2 $OUT/dets/dino.log | tr '\n' ' ' | cut -c1-200)"; fi
echo "done: $ok ok, $bad failed, $(ls $OUT/dets/*.pkl | wc -l) pickles in $OUT/dets"
