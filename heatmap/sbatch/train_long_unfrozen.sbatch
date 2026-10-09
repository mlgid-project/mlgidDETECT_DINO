#!/bin/bash
#SBATCH --job-name=hm_long_unfrozen
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_long_unfrozen-%j.out
# Run C ("long", frozen SimMIM swin, hm_simmim_frozen_ridge_long_tf32_2.80_1.30) with the swin UNFROZEN. Everything else as C:
# ridge ring target, TF32, bs 8 x 125 steps (1000 images/epoch), 120 epochs, head lr 4.2e-4, lr x0.1 at epochs 90 and 112, eval every 2 epochs.
# New: swin trains with lr 1e-5 (the repo default --lr_backbone, untuned), stochastic depth OFF (--bb_drop_path 0, as in the frozen runs).
# A 12-step smoke at bs 8 decides whether gradient checkpointing (--grad_ckpt, same maths, ~30% slower) is needed to fit in GPU memory.
# Resumable: if the time limit hits, resubmit this same file (it reads <out>/checkpoint.pth).
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
OUT=$BASE/hm_simmim_UNFROZEN_ridge_long_tf32_2.80_1.30
export PYTHONPATH=$REPO
cd $REPO
mkdir -p "$OUT"
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/gpu_mem.log"; sleep 60; done ) &
MON=$!
COMMON="--bb simmim1 --ring_target ridge --tf32 --lr 4.2e-4 --lr_drops 90 112 --epochs 120 --eval_interval 2 --bs 8 --steps_per_epoch 125 --unfreeze --lr_backbone 1e-5 --bb_drop_path 0"
CK=""
if [ ! -f "$OUT/checkpoint.pth" ]; then
  SM=$BASE/_smoke_unfrozen; rm -rf "$SM"; mkdir -p "$SM"
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
  $PY -u heatmap/evaluate.py hm=heatmap:"$OUT/final_checkpoint.pth" > "$OUT/evaluate_final.txt" 2>&1; echo "eval rc=$?"
fi
kill $MON
