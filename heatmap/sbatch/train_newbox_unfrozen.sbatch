#!/bin/bash
#SBATCH --job-name=hm_nb_unfrozen
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_nb_unfrozen-%j.out
# The "long" unfrozen recipe (swin trains, ridge target, TF32, bs 8 x 125 steps, 120 epochs, head lr 4.2e-4, lr x0.1 at 90 and 112, eval every 2 epochs) on the NEW-BOX eval files with box_coef BOXC (default 3.0,3.0).
# Env: SEED (default 1), LR_BB (backbone lr, default 1e-5), BOXC (a,w). In-training evals AND the final scoring use the GT scaled to the run's box convention (HM_GT_BOX_COEF=auto); scaled GT is the only final scoring.
# The swin is NOT frozen, so the train-time and eval-time GT convention must match the sim box convention: it does (same BOXC for --box_coef and the auto-scaled GT).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
SEED=${SEED:-1}; BC=${BOXC:-3.0,3.0}; BN=${BC/,/_}
LR_BB=${LR_BB:-1e-5}          # backbone lr; the default keeps every earlier run unchanged. Other values write to ..._bblr<LR_BB>_... (e.g. sbatch --export=ALL,SEED=1,FORCE_CKPT=1,LR_BB=4e-5 ...)
OUT=$BASE/hm_simmim_UNFROZEN_ridge_long_tf32_seed${SEED}_newbox_${BN}_galvani
if [ "$LR_BB" != 1e-5 ]; then OUT=${OUT%_galvani}_bblr${LR_BB}_galvani; fi
export PYTHONPATH=$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF=auto
cd $REPO
mkdir -p "$OUT"
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/gpu_mem.log"; sleep 60; done ) &
MON=$!
COMMON="--bb simmim1 --ring_target ridge --tf32 --lr 4.2e-4 --lr_drops 90 112 --epochs 120 --eval_interval 2 --bs 8 --steps_per_epoch 125 --unfreeze --lr_backbone $LR_BB --bb_drop_path 0 --seed $SEED --box_coef $BC"
CK=""
if [ "${FORCE_CKPT:-0}" = 1 ] && [ ! -f "$OUT/checkpoint.pth" ]; then CK="--grad_ckpt"; echo "FORCE_CKPT: using --grad_ckpt without smoke"
elif [ ! -f "$OUT/checkpoint.pth" ]; then
  SM=$BASE/_smoke_nb_unfrozen_s${SEED}_${BN}; rm -rf "$SM"; mkdir -p "$SM"
  if $PY -u heatmap/train.py --out "$SM" $COMMON --max_steps 12 > "$SM/smoke_plain.log" 2>&1; then echo "smoke bs 8 plain OK"; tail -2 "$SM/smoke_plain.log"
  else
    echo "smoke bs 8 plain FAILED:"; tail -5 "$SM/smoke_plain.log"
    if $PY -u heatmap/train.py --out "$SM" $COMMON --grad_ckpt --max_steps 12 > "$SM/smoke_ckpt.log" 2>&1; then CK="--grad_ckpt"; echo "smoke bs 8 + grad_ckpt OK -> using --grad_ckpt"; tail -2 "$SM/smoke_ckpt.log"
    else echo "smoke with grad_ckpt FAILED too -> abort"; tail -8 "$SM/smoke_ckpt.log"; kill $MON; exit 1; fi
  fi
else echo "resuming from $OUT/checkpoint.pth"; grep -q grad_ckpt "$OUT/args.json" && grep -q '"grad_ckpt": true' "$OUT/args.json" && CK="--grad_ckpt"; fi
$PY -u heatmap/train.py --out "$OUT" $COMMON $CK >> "$OUT/train.log" 2>&1
RC=$?; echo "train rc=$RC"
if [ $RC = 0 ] && [ -f "$OUT/checkpoint.pth" ]; then
  cp "$OUT/checkpoint.pth" "$OUT/final_checkpoint.pth"
  $PY -u heatmap/evaluate.py hm=heatmap:"$OUT/final_checkpoint.pth" > "$OUT/evaluate_final_gtscaled.txt" 2>&1; echo "eval (scaled GT) rc=$?"
fi
# optional: re-score finished runs whose evaluate_final_gtscaled.txt is empty (env EXTRA_EVAL="<run_dir> <run_dir> ...", scores each <run_dir>/checkpoint.pth, same job so no extra small Slurm jobs)
for D in ${EXTRA_EVAL:-}; do
  if [ -f "$D/checkpoint.pth" ]; then $PY -u heatmap/evaluate.py hm=heatmap:"$D/checkpoint.pth" > "$D/evaluate_final_gtscaled.txt" 2>&1; echo "extra eval (scaled GT) $D rc=$?"
  else echo "extra eval: no checkpoint in $D"; fi
done
kill $MON
