#!/bin/bash
# colorbox1 queue 7 (detached, one job at a time), started 2026-10-21 after the user's go: H1 and H3 on the re-fitted validation files (visibility-fixed copies) with +-1 sigma boxes
# (--box_coef 2,2 = a_coef,w_coef 2,2), legacy simulator, frozen SimMIM swin, seed 42. Results go to the separate group hm_runs/newbox/ (not comparable with the 2.80/1.30 rows).
#   H1  plain recipe as the tf32 replicates: bs 4 x 250, lr 3e-4, 60 ep, lr_drop 45, eval every 2 (replicates: every 5)
#   H3  run C recipe: bs 8 x 125, lr 4.2e-4, 120 ep, lr_drops 90 112, eval every 2
# A 40-step smoke run precedes H1 (output hm_runs/_scratch/newbox_smoke); if it fails the queue stops. Post steps per run: evaluate.py (+nms headline), +nms images, score sweep.
# Data: HM_DATA_DIR=~/Documents/datasets/validationsSets/newbox_visfix (organic_labeled.h5 = organics_visfix.h5, 41.h5 = perovskites_nan_wedge_visfix.h5; md5 checked 2026-10-21).
# Needs the 5-line clamp patch in util/pygidloader.py (infinite ring angle_width). Status file: hm_runs/_tools/queue7_status.txt, last line 'QUEUE7 ALL DONE'. Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue7_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue7.lock"
flock -n 9 || { log "another queue7 instance holds the lock -> exit"; exit 0; }
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
    "$PY" -u heatmap/train.py --out "$R" --bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --box_coef 2,2 "$@" >> "$R/train.log" 2>&1
    rc=$?
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null; peak=$(sort -n "$R/gpu_mem.log" | tail -n 1)
    log "END   train $name rc=$rc (peak GPU memory, sampled every 5 s: ${peak} MiB)"
  fi
  if ! grep -q "^\[epoch $last\] loss" "$R/train.log" 2>/dev/null; then
    log "FAILED $name: training incomplete. Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if eval_done "$R"; then log "SKIP evaluate $name"; else
    "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final.txt" 2>&1; log "EVAL  $name rc=$?"; fi
  if images_done "$R"; then log "SKIP images $name"; else
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/visualize.py --ckpt "$R/final_checkpoint.pth" --out "$R/images" --sets organic 41 --thr 0.3 --decode nms > "$R/visualize_nms.log" 2>&1; log "IMAGES(nms) $name rc=$?"; fi
  if sweep_done "$R"; then log "SKIP sweep $name"; else
    mkdir -p "$R/score_sweep"
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/score_sweep.py "$name=$R/final_checkpoint.pth" --out "$R/score_sweep" --decode nms --topk 225 900 > "$R/score_sweep.log" 2>&1
    rc=$?; if sweep_done "$R"; then log "SWEEP $name ok (rc=$rc)"; else log "SWEEP $name FAILED (rc=$rc)"; fi; fi
}

log "QUEUE7 START (pid $$)"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
SM=$RUNS/_scratch/newbox_smoke; rm -rf "$SM"; mkdir -p "$SM"
if "$PY" -u heatmap/train.py --out "$SM" --bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --box_coef 2,2 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2 --seed 42 --max_steps 40 > "$SM/smoke.log" 2>&1
then log "SMOKE ok: $(tail -n 1 "$SM/smoke.log")"; else log "SMOKE FAILED -> queue stops: $(tail -n 5 "$SM/smoke.log" | tr '\n' '|')"; exit 1; fi
one_run hm_simmim_frozen_ridge_tf32_seed42_newbox_2.00_2.00 59 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 2 --seed 42 || log "QUEUE7: H1 failed, continuing"
one_run hm_simmim_frozen_ridge_long_tf32_seed42_newbox_2.00_2.00 119 --lr 4.2e-4 --lr_drops 90 112 --epochs 120 --eval_interval 2 --bs 8 --steps_per_epoch 125 --seed 42 || log "QUEUE7: H3 failed, continuing"
log "QUEUE7 ALL DONE"
