#!/bin/bash
# colorbox1 queue (detached, one job at a time): coordinate channels, unfrozen swin x2, run B replicates x3.
# Every run: plain 60-epoch recipe (--tf32 --ring_target ridge --bs 4 --steps_per_epoch 250 --lr 3e-4 --lr_drop 45), post steps as in runc_queue.sh.
# Order and rough duration (117 s/epoch frozen, ~6 h unfrozen at bs 4, 12-step measurement):
#   0. smoke of --chan coord (12 steps)                                     ~1 min
#   1. coord seed 1   (HE + x/W + y/H into the stem, swin frozen)           ~2.1 h
#   2. unfrozen swin seed 1 (bs 4, --grad_ckpt, lr_backbone 1e-5)           ~6.2 h   (peak GPU 11.5 of 12.2 GB in the smoke: OOM -> queue stops)
#   3. run B seed 1   (boxconv1 backbone, frozen)                           ~2.1 h
#   4. run B seed 2                                                         ~2.1 h
#   5. unfrozen swin seed 2                                                 ~6.2 h
#   6. run B seed 3                                                         ~2.1 h
# Total ~21 h + ~5 min post steps per run. No end-time limit. A failed training stops the whole queue (no fallback, recipe never changed silently).
# Status file: hm_runs/_tools/queue2_status.txt, last line 'QUEUE2 ALL DONE'. Re-running the script skips finished steps.
# Needs on this machine (applied from origin/heatmap, single files): heatmap/train.py, models/heatmap_head.py (chan coord, grad_ckpt), heatmap/summarize_runs.py.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
BOXCONV=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/boxconv1_backbone.pth
STATUS=$RUNS/_tools/queue2_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
EPOCHS=60
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue2.lock"
flock -n 9 || { log "another queue2 instance holds the lock -> exit"; exit 0; }
train_done()  { grep -q "^\[epoch $((EPOCHS-1))\] loss" "$1/train.log" 2>/dev/null; }
eval_done()   { grep -q '+nms\] 41: ap_total' "$1/evaluate_final.txt" 2>/dev/null; }
images_done() { [ -f "$1/images/nms/41/closepairs.png" ]; }
sweep_done()  { ls "$1"/score_sweep/*_score_sweep.txt >/dev/null 2>&1; }

# one_run <name> <bb file for visualize/sweep> <train.py args...>
one_run() {
  local name=$1 postbb=$2; shift 2
  local R=$RUNS/$name rc peak SAMP
  mkdir -p "$R"
  if train_done "$R"; then log "SKIP train $name"; else
    log "START train $name :: $*"
    : > "$R/gpu_mem.log"; ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; sleep 5; done ) & SAMP=$!
    "$PY" -u heatmap/train.py --out "$R" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 \
        --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 "$@" >> "$R/train.log" 2>&1
    rc=$?
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null; peak=$(sort -n "$R/gpu_mem.log" | tail -n 1)
    log "END   train $name rc=$rc (peak GPU memory, sampled every 5 s: ${peak} MiB)"
  fi
  if ! train_done "$R"; then
    log "FAILED $name: training incomplete (no fallback). Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"
    return 1
  fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if eval_done "$R"; then log "SKIP evaluate $name"; else
    "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final.txt" 2>&1; log "EVAL  $name rc=$?"; fi
  if images_done "$R"; then log "SKIP images $name"; else
    HM_BB_PATH="$postbb" "$PY" -u heatmap/visualize.py --ckpt "$R/final_checkpoint.pth" --out "$R/images" --sets organic 41 --thr 0.3 --decode nms > "$R/visualize_nms.log" 2>&1; log "IMAGES(nms) $name rc=$?"; fi
  if sweep_done "$R"; then log "SKIP sweep $name"; else
    mkdir -p "$R/score_sweep"
    HM_BB_PATH="$postbb" "$PY" -u heatmap/score_sweep.py "$name=$R/final_checkpoint.pth" --out "$R/score_sweep" --decode nms --topk 225 900 > "$R/score_sweep.log" 2>&1
    rc=$?; if sweep_done "$R"; then log "SWEEP $name ok (rc=$rc)"; else log "SWEEP $name FAILED (rc=$rc)"; fi; fi
}

COORD="--bb_path $SIMMIM --chan coord"
UNF="--bb_path $SIMMIM --chan he --unfreeze --lr_backbone 1e-5 --bb_drop_path 0 --grad_ckpt"
BOX="--bb ssl1 --bb_path $BOXCONV"
log "QUEUE2 START (pid $$)"
while pgrep -f 'heatmap/train.py' >/dev/null; do sleep 60; done

# 0. smoke of the new coord channels (12 steps, throw-away dir)
SM=$RUNS/_smoke_coord; rm -rf "$SM"; mkdir -p "$SM"
if "$PY" -u heatmap/train.py --out "$SM" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 $COORD --max_steps 12 > "$SM/smoke.log" 2>&1; then
  log "SMOKE coord ok: $(tail -n 1 "$SM/smoke.log")"
else log "SMOKE coord FAILED -> queue stopped: $(tail -n 5 "$SM/smoke.log" | tr '\n' '|')"; exit 1; fi

one_run hm_simmim_frozen_ridge_coord_tf32_seed1_2.80_1.30 "$SIMMIM" $COORD --seed 1            || { log "QUEUE2 STOPPED (coord seed 1)"; exit 1; }
one_run hm_simmim_unfrozen_ridge_he_tf32_seed1_2.80_1.30 "$SIMMIM" $UNF --seed 1                || { log "QUEUE2 STOPPED (unfrozen seed 1)"; exit 1; }
one_run hm_boxconv1_frozen_ridge_tf32_seed1_2.80_1.30 "$BOXCONV" $BOX --seed 1                  || { log "QUEUE2 STOPPED (B seed 1)"; exit 1; }
one_run hm_boxconv1_frozen_ridge_tf32_seed2_2.80_1.30 "$BOXCONV" $BOX --seed 2                  || { log "QUEUE2 STOPPED (B seed 2)"; exit 1; }
one_run hm_simmim_unfrozen_ridge_he_tf32_seed2_2.80_1.30 "$SIMMIM" $UNF --seed 2                || { log "QUEUE2 STOPPED (unfrozen seed 2)"; exit 1; }
one_run hm_boxconv1_frozen_ridge_tf32_seed3_2.80_1.30 "$BOXCONV" $BOX --seed 3                  || { log "QUEUE2 STOPPED (B seed 3)"; exit 1; }
log "QUEUE2 ALL DONE"
