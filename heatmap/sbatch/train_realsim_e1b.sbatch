#!/bin/bash
#SBATCH --job-name=hm_realsim_E1b
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP/heatmap/sbatch/hm_realsim_e1b-%j.out
# E1b = E1 (calibrated real-background simulator, rings + segments in the same frame, ~15 ring boxes/frame; plain 60-epoch frozen-SimMIM recipe, seed 42, +nms evaluation)
# with ONE change: correlated background noise (bkg_noise_blur=(0.0,1.2), bkg_noise_power=1.5, calibrated to the real 41 lag-1 autocorrelation 0.19 / 0.34, quartiles 0.05-0.79).
# E1 (white-ish noise, colorbox1): organic 0.6111, 41 0.4976 (41 precision 0.515, 1104 false positives per 41 frames). E1b tests whether the noise correlation removes the false positives.
# Frames from 4 worker processes (13 GB RAM peak each). Resumable (<out>/checkpoint.pth). Env: SEED, SIM_WORKERS.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
SEED=${SEED:-42}; SIM_WORKERS=${SIM_WORKERS:-4}
OUT=$BASE/hm_simmim_frozen_ridge_realsimE1b_tf32_seed${SEED}_2.80_1.30
export PYTHONPATH=$REPO OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $REPO
mkdir -p "$OUT"
echo "git: $(git branch --show-current) $(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$OUT/ram_used_MiB.log"; sleep 60; done ) &
MON=$!
SIM=(--sim realbkg --sim_workers $SIM_WORKERS --realsim_opt "amplitude_mode='fitted'" intensity_decades=None ring_gain=0.4 ring_contrast_min=0.3 ring_chi_texture=0.05 "ring_chi_corr=(8.0,64.0)"
     seg_sc_scale=0.6 ring_sq_scale=1.0 ring_w_coef=2.2 ring_no_elongate=True ring_cap=4.0 ring_chi_floor=0.5 ring_follow_bkg=True "width_jitter=(0.2,0.832)"
     "bkg_noise_blur=(0.0,1.2)" bkg_noise_power=1.5
     seg_wide_frac=0.0 p_ring=0.85 "n_powder=(1,2)" "rings_cap=(6,30)" "n_oriented=(1,2)" "spots_cap=(10,60)")
COMMON=(--bb simmim1 --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2 --seed $SEED "${SIM[@]}")
if [ ! -f "$OUT/checkpoint.pth" ]; then
  SM=$BASE/_smoke_realsim_e1b; rm -rf "$SM"; mkdir -p "$SM"
  if $PY -u heatmap/train.py --out "$SM" "${COMMON[@]}" --max_steps 40 > "$SM/smoke.log" 2>&1; then echo "smoke OK"; tail -3 "$SM/smoke.log"
  else echo "smoke FAILED -> abort"; tail -15 "$SM/smoke.log"; kill $MON; exit 1; fi
else echo "resuming from $OUT/checkpoint.pth"; fi
$PY -u heatmap/train.py --out "$OUT" "${COMMON[@]}" >> "$OUT/train.log" 2>&1
RC=$?; echo "train rc=$RC"
if [ $RC = 0 ] && [ -f "$OUT/checkpoint.pth" ]; then
  cp "$OUT/checkpoint.pth" "$OUT/final_checkpoint.pth"
  $PY -u heatmap/evaluate.py hm="heatmap:$OUT/final_checkpoint.pth" > "$OUT/evaluate_final.txt" 2>&1; echo "eval rc=$?"
fi
kill $MON
