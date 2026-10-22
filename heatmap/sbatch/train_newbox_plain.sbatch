#!/bin/bash
#SBATCH --job-name=hm_nb_plain
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_nb_plain-%j.out
# H1 recipe on the new-box eval files (frozen SimMIM, legacy sim, ridge target, plain 60-epoch recipe, bs 4 x 250, lr 3e-4, drop 45, TF32) with box_coef COEF,COEF.
# Seed replicates of the colorbox queue-8/9 runs (seed 42 there). Env: SEED (default 1), COEF (default 2.5). In-training evals AND the final scoring use the GT scaled to the run's box convention (HM_GT_BOX_COEF=auto),
# evaluate_final.txt = fixed GT (HM_GT_BOX_COEF unset), evaluate_final_gtscaled.txt = scaled GT. Resumable (<out>/checkpoint.pth).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
SEED=${SEED:-1}; COEF=${COEF:-2.5}; BC="$COEF,$COEF"
OUT=$BASE/hm_simmim_frozen_ridge_tf32_seed${SEED}_newbox_${COEF}_${COEF}_galvani
export PYTHONPATH=$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links
cd $REPO
mkdir -p "$OUT"
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD) seed $SEED box_coef $BC"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON=(--bb simmim1 --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2 --seed $SEED --box_coef $BC)
if [ ! -f "$OUT/checkpoint.pth" ]; then
  SM=$BASE/_smoke_nb_plain_${SEED}_${COEF}; rm -rf "$SM"; mkdir -p "$SM"
  if HM_GT_BOX_COEF=auto $PY -u heatmap/train.py --out "$SM" "${COMMON[@]}" --max_steps 40 > "$SM/smoke.log" 2>&1; then echo "smoke OK"; grep "GT box scale" "$SM/smoke.log"; tail -2 "$SM/smoke.log"
  else echo "smoke FAILED -> abort"; tail -15 "$SM/smoke.log"; exit 1; fi
else echo "resuming from $OUT/checkpoint.pth"; fi
HM_GT_BOX_COEF=auto $PY -u heatmap/train.py --out "$OUT" "${COMMON[@]}" >> "$OUT/train.log" 2>&1
RC=$?; echo "train rc=$RC"
if [ $RC = 0 ] && [ -f "$OUT/checkpoint.pth" ]; then
  cp "$OUT/checkpoint.pth" "$OUT/final_checkpoint.pth"
  $PY -u heatmap/evaluate.py hm="heatmap:$OUT/final_checkpoint.pth" > "$OUT/evaluate_final.txt" 2>&1; echo "eval (fixed GT) rc=$?"
  HM_GT_BOX_COEF=auto $PY -u heatmap/evaluate.py hm="heatmap:$OUT/final_checkpoint.pth" > "$OUT/evaluate_final_gtscaled.txt" 2>&1; echo "eval (scaled GT) rc=$?"
fi
