#!/bin/bash
# colorbox1 (detached, one job): heatmap detector on the REAL-BACKGROUND simulator (realsim/), plain 60-epoch frozen-SimMIM recipe, +nms evaluation.
# Same flags as every plain run (ridge, TF32, bs 4 x 250, lr 3e-4, lr x0.1 at epoch 45, eval every 2 epochs, seed 42). Simulator: ONE structure per image,
# 25 % ring-only / 75 % segment-only, 25-75 reflections drawn per structure, widths as measured (the 'single' variant the user approved from the images).
# Frames come from ONE worker process (0.19 s/frame, 13 GB peak RAM; colorbox1 has 30 GB), so expect ~190 s/epoch = ~3.2 h; peak GPU ~6.4 GB.
# Steps: smoke (40 steps, throw-away dir) -> train -> evaluate.py -> visualize (nms) -> score_sweep, as in queue_coord_unfrozen_b.sh. A failed step stops the script.
# Status: hm_runs/_tools/queue3_status.txt, last line 'QUEUE3 ALL DONE'. Re-running skips finished steps. Waits while any heatmap/train.py runs.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue3_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets
export GIWAXS_WORK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
EPOCHS=60
NAME=hm_simmim_frozen_ridge_realsim_single_tf32_seed42_2.80_1.30
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue3.lock"
flock -n 9 || { log "another queue3 instance holds the lock -> exit"; exit 0; }
R=$RUNS/$NAME
SIM=(--sim realbkg --sim_workers 1 --realsim_opt single_structure=True p_ring=0.25 "spots_cap=(25,75)")
COMMON=(--ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --bb_path "$SIMMIM" --seed 42 "${SIM[@]}")
log "QUEUE3 START (pid $$)"
while pgrep -f 'heatmap/train.py' >/dev/null; do sleep 60; done
mkdir -p "$R"
if grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "SKIP train $NAME"; else
  if [ ! -f "$R/checkpoint.pth" ]; then
    SM=$RUNS/_smoke_realsim; rm -rf "$SM"; mkdir -p "$SM"
    if "$PY" -u heatmap/train.py --out "$SM" "${COMMON[@]}" --max_steps 40 > "$SM/smoke.log" 2>&1; then log "SMOKE ok: $(tail -n 1 "$SM/smoke.log")"
    else log "SMOKE FAILED -> stopped: $(tail -n 8 "$SM/smoke.log" | tr '\n' '|')"; exit 1; fi
  fi
  log "START train $NAME :: ${COMMON[*]}"
  : > "$R/gpu_mem.log"; ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print "ram_used_MiB",$3}' >> "$R/gpu_mem.log"; sleep 5; done ) & SAMP=$!
  "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" >> "$R/train.log" 2>&1
  rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  log "END   train $NAME rc=$rc"
fi
if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then
  log "FAILED $NAME: training incomplete. Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; log "QUEUE3 STOPPED"; exit 1
fi
[ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
if grep -q '+nms\] 41: ap_total' "$R/evaluate_final.txt" 2>/dev/null; then log "SKIP evaluate"; else
  "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final.txt" 2>&1; log "EVAL rc=$?"; fi
if [ -f "$R/images/nms/41/closepairs.png" ]; then log "SKIP images"; else
  HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/visualize.py --ckpt "$R/final_checkpoint.pth" --out "$R/images" --sets organic 41 --thr 0.3 --decode nms > "$R/visualize_nms.log" 2>&1; log "IMAGES(nms) rc=$?"; fi
if ls "$R"/score_sweep/*_score_sweep.txt >/dev/null 2>&1; then log "SKIP sweep"; else
  mkdir -p "$R/score_sweep"
  HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/score_sweep.py "$NAME=$R/final_checkpoint.pth" --out "$R/score_sweep" --decode nms --topk 225 900 > "$R/score_sweep.log" 2>&1; log "SWEEP rc=$?"; fi
log "QUEUE3 ALL DONE"
