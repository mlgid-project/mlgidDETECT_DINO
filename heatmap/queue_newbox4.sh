#!/bin/bash
# colorbox1 queue 10 (detached, one job at a time), started 2026-10-23: seed 1/2 replicates of the queue-9 box-size runs, same H1 recipe (frozen SimMIM swin, ridge target, TF32, legacy simulator, bs 4 x 250, lr 3e-4, 60 ep, lr_drop 45, eval every 2,
# newbox_visfix files with the clamp patch). Order: 3.5 s1, 4.0 s1, 3.5 s2, 4.0 s2. Training runs with HM_GT_BOX_COEF=auto (in-training evals use the scaled GT). After each run: evaluate_final.txt (fixed GT, HM_GT_BOX_COEF unset) AND evaluate_final_gtscaled.txt (HM_GT_BOX_COEF=auto),
# +nms images and score sweep on the fixed GT as in queue 9. A failing run is logged and the queue continues. Status file: hm_runs/_tools/queue10_status.txt, last line QUEUE10 ALL DONE. Shares the GPU lock. Results under hm_runs/newbox/.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue10_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH HM_GT_BOX_COEF
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue10.lock"
flock -n 9 || { log "another queue10 instance holds the lock -> exit"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"
flock 8
log "GPU lock acquired"
eval_done()   { grep -q '+nms\] 41: ap_total' "$1/evaluate_final.txt" 2>/dev/null; }
images_done() { [ -f "$1/images/nms/41/closepairs.png" ]; }
sweep_done()  { ls "$1"/score_sweep/*_score_sweep.txt >/dev/null 2>&1; }
# one_run <name> <last epoch index> <train.py args...>
one_run() {
  local name=$1 last=$2; shift 2
  local R=$RUNS/newbox/runs/$name rc peak SAMP
  mkdir -p "$R"
  if grep -q "^\[epoch $last\] loss" "$R/train.log" 2>/dev/null; then log "SKIP train $name"; else
    log "START train $name :: $*"
    : > "$R/gpu_mem.log"; ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; sleep 5; done ) & SAMP=$!
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" --bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 "$@" >> "$R/train.log" 2>&1
    rc=$?
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null; peak=$(sort -n "$R/gpu_mem.log" | tail -n 1)
    log "END   train $name rc=$rc (peak GPU memory, sampled every 5 s: ${peak} MiB)"
  fi
  if ! grep -q "^\[epoch $last\] loss" "$R/train.log" 2>/dev/null; then
    log "FAILED $name: training incomplete. Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if eval_done "$R"; then log "SKIP evaluate $name"; else
    "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final.txt" 2>&1; log "EVAL  $name rc=$?"; fi
  if grep -q '+nms\] 41: ap_total' "$R/evaluate_final_gtscaled.txt" 2>/dev/null; then log "SKIP evaluate_gtscaled $name"; else
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $name rc=$?"; fi
  if images_done "$R"; then log "SKIP images $name"; else
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/visualize.py --ckpt "$R/final_checkpoint.pth" --out "$R/images" --sets organic 41 --thr 0.3 --decode nms > "$R/visualize_nms.log" 2>&1; log "IMAGES(nms) $name rc=$?"; fi
  if sweep_done "$R"; then log "SKIP sweep $name"; else
    mkdir -p "$R/score_sweep"
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/score_sweep.py "$name=$R/final_checkpoint.pth" --out "$R/score_sweep" --decode nms --topk 225 900 > "$R/score_sweep.log" 2>&1
    rc=$?; if sweep_done "$R"; then log "SWEEP $name ok (rc=$rc)"; else log "SWEEP $name FAILED (rc=$rc)"; fi; fi
}

log "QUEUE10 START (pid $$)"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
H1=(--bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2)
# plan <name> <box_coef> <seed>
plan() {
  local name=$1 coef=$2 seed=$3
  one_run "$name" 59 "${H1[@]}" --seed "$seed" --box_coef "$coef" || log "QUEUE10: $name failed, continuing"
}
plan hm_simmim_frozen_ridge_tf32_seed1_newbox_3.50_3.50 3.5,3.5 1
plan hm_simmim_frozen_ridge_tf32_seed1_newbox_4.00_4.00 4.0,4.0 1
plan hm_simmim_frozen_ridge_tf32_seed2_newbox_3.50_3.50 3.5,3.5 2
plan hm_simmim_frozen_ridge_tf32_seed2_newbox_4.00_4.00 4.0,4.0 2
log "QUEUE10 ALL DONE"
