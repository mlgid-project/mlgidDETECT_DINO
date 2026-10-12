#!/bin/bash
#SBATCH --job-name=hm_realsim
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_realsim-%j.out
# Heatmap detector trained on the REAL-BACKGROUND simulator (realsim/), plain 60-epoch frozen-SimMIM recipe (the 'plain TF32' recipe: ridge ring target,
# TF32, bs 4 x 250 steps, head lr 3e-4, lr x0.1 at epoch 45, eval every 2 epochs; +nms evaluation as always), so it is comparable with the plain seeds
# (organic 0.6320/0.5989/0.6027, 41 0.7260/0.7225/0.7266) and with the DINO references.
# Simulator settings = the 'single' variant the user approved from the images: ONE structure per image (25 % ring-only frames, 75 % segment-only),
# 25-75 reflections drawn per structure, widths as measured (no width_jitter). Frames are made by SIM_WORKERS separate processes (0.19 s/frame each, 13 GB peak RAM each).
# Resumable (reads <out>/checkpoint.pth). Name suffix and options can be overridden: sbatch --export=ALL,TAG=x,SEED=1,EXTRA="--realsim_opt ..." ...
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
SEED=${SEED:-42}; TAG=${TAG:-realsim_single}; SIM_WORKERS=${SIM_WORKERS:-4}
OUT=$BASE/hm_simmim_frozen_ridge_${TAG}_tf32_seed${SEED}_2.80_1.30
export PYTHONPATH=$REPO OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $REPO
mkdir -p "$OUT"
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/gpu_mem.log"; free -m | awk 'NR==2{print "ram_used_MiB",$3}' >> "$OUT/gpu_mem.log"; sleep 60; done ) &
MON=$!
SIM="--sim realbkg --sim_workers $SIM_WORKERS --realsim_opt single_structure=True p_ring=0.25 spots_cap=(25,75)"
COMMON="--bb simmim1 --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2 --seed $SEED $SIM $EXTRA"
if [ ! -f "$OUT/checkpoint.pth" ]; then
  SM=$BASE/_smoke_realsim_$TAG; rm -rf "$SM"; mkdir -p "$SM"
  if $PY -u heatmap/train.py --out "$SM" $COMMON --max_steps 40 > "$SM/smoke.log" 2>&1; then echo "smoke OK"; tail -3 "$SM/smoke.log"
  else echo "smoke FAILED -> abort"; tail -15 "$SM/smoke.log"; kill $MON; exit 1; fi
else echo "resuming from $OUT/checkpoint.pth"; fi
$PY -u heatmap/train.py --out "$OUT" $COMMON >> "$OUT/train.log" 2>&1
RC=$?; echo "train rc=$RC"
if [ $RC = 0 ] && [ -f "$OUT/checkpoint.pth" ]; then
  cp "$OUT/checkpoint.pth" "$OUT/final_checkpoint.pth"
  $PY -u heatmap/evaluate.py hm=heatmap:"$OUT/final_checkpoint.pth" > "$OUT/evaluate_final.txt" 2>&1; echo "eval rc=$?"
fi
kill $MON
