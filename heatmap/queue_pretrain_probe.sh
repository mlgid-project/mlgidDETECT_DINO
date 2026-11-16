#!/bin/bash
# colorbox1 queue 29 (detached), user go 2026-11-06 ("yes queue that aswell"): waits for QUEUE28 ALL DONE, then the frozen probe of the pretrained backbone.
# Recipe identical to the frozen ConvNeXt-L 120-epoch run of queue 26 (0.6675 / 0.6733, seed 42), only --bb_path differs. Skipped when the pretraining did not finish or its log has NaN.
# RAM watchdog kills the training when MemAvailable < 6 GiB; it resumes from its own checkpoint.pth (up to 4 attempts, each waits for 9 GB). HM_BB_PATH must stay unset.
# Status: hm_runs/_tools/queue29_status.txt (last line QUEUE29 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
DATA=/mnt/DATA/mlgidDETECT_DINO_HEATMAP
RUNS=$DATA/hm_runs
PRE=$DATA/pretrain_runs/convnextL_3k
STATUS=$RUNS/_tools/queue29_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF HM_BB_PATH GIWAXS_BANK
EPOCHS=120
NAME=hm_dinov3convnextL_pretrain3k_frozen_ridge_long_tf32_seed42_newbox_3.0_3.0
R=$RUNS/newbox/runs/$NAME
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue29.lock"
flock -n 9 || { log "another queue29 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }

log "QUEUE29 START (pid $$), waiting for QUEUE28 ALL DONE"
until grep -q 'QUEUE28 ALL DONE' "$RUNS/_tools/queue28_status.txt" 2>/dev/null; do sleep 60; done
if ! grep -q 'PRETRAIN DONE' "$PRE/train.log" 2>/dev/null || [ ! -f "$PRE/backbone_teacher.pth" ]; then log "SKIPPED: pretraining did not finish"; log "QUEUE29 ALL DONE"; exit 1; fi
if grep -qi 'nan' "$PRE/train.log"; then log "SKIPPED: NaN in the pretraining log"; log "QUEUE29 ALL DONE"; exit 1; fi
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done

BASE=(--ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drops 90 112 --eval_interval 2 --box_coef 3.0,3.0 --sim legacy)
mkdir -p "$R"
for att in 1 2 3 4; do
  grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
  wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 30 min before attempt $att of $NAME"
  log "START train $NAME attempt $att"
  : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${BASE[@]}" --bb dinov3_convnext_large --bb_path "$PRE/backbone_teacher.pth" --seed 42 >> "$R/train.log" 2>&1 & TP=$!
  ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $NAME"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END   train $NAME attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
done
if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $NAME: training incomplete after 4 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; log "QUEUE29 ALL DONE"; exit 1; fi
[ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $NAME rc=$?"
log "DONE $NAME: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
log "QUEUE29 ALL DONE"
