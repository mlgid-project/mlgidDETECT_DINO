#!/bin/bash
# colorbox1 queue 35 (detached), user: keep the GPU busy. Pretraining of queue 34 is done (patch loss 7.6 -> 2.66, no collapse, frozen probe 0.6828 / 0.6876), so this only runs the UNFROZEN probe of that backbone (seed 42, recipe of queue 32: --unfreeze --grad_ckpt --lr_backbone 1e-5).
# Compare unfrozen: public 0.6953 / 0.6910 (two seeds), 3k collapsed 0.7123 / 0.6815, 10k collapsed 0.711 / 0.687. The pretraining block below is skipped (PRETRAIN DONE is in its train.log).
# --prototypes 1024 --proto_patch 1024 --patch_stats $PRE/patch_stats.pt --patch_g1, stop rule --min_perp_frac 0.1 --check_from 600 (exit 3 -> no probe).
# Then the frozen probe, seed 42, recipe of queue 32 / 29, compare 10k frozen 0.6800 / 0.6941, 3k frozen 0.6404 / 0.7035, public frozen 0.6675 / 0.6733.
# RAM watchdog kills a run when MemAvailable < 6 GiB; both resume from their own checkpoint. An out-of-memory error stops the queue, no recipe change. Status: hm_runs/_tools/queue35_status.txt (last line QUEUE35 ALL DONE). Shares the GPU lock. Start with bash.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
DATA=/mnt/DATA/mlgidDETECT_DINO_HEATMAP
RUNS=$DATA/hm_runs
PRE=$DATA/pretrain_runs/convnextL_3k_sk2
W=$PRE/backbone_teacher.pth
STATUS=$RUNS/_tools/queue35_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF HM_BB_PATH GIWAXS_BANK
EPOCHS=120
EXCL=$DATA/moneta_sweep/holdout/pretrain_exclude_v2.tsv
STEPS=3000
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue35.lock"
flock -n 9 || { log "another queue35 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 240); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }

log "QUEUE35 START (pid $$)"
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
mkdir -p "$PRE"
for att in 1 2 3 4 5 6; do
  grep -q 'PRETRAIN DONE\|PRETRAIN STOPPED' "$PRE/train.log" 2>/dev/null && break
  wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 2 h before pretraining attempt $att"
  log "START pretrain attempt $att"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$PRE/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$PRE/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  "$PY" -u heatmap/pretrain_dinov3.py --out "$PRE" --corpus "$DATA/corpus_v2/scans" "$DATA/corpus_v2/scans_other" --exclude "$EXCL" --flip_chi --bs 6 --accum 11 --workers 6 --steps $STEPS --head_steps 300 --log_every 10 --save_every 250 --grad_ckpt \
      --sinkhorn 3 --t_warmup 0.04 0.07 1000 --koleo 0.1 --gram 10 --prototypes 1024 --proto_patch 1024 --patch_stats $PRE/patch_stats.pt --patch_g1 --min_perp_frac 0.1 --check_from 600 >> "$PRE/stdout.log" 2>&1 & TP=$!
  ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the pretraining"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END pretrain attempt $att rc=$rc (peak GPU $(sort -n "$PRE/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$PRE/ram_used_MiB.log" | tail -n 1) MiB): $(grep -E '^step' "$PRE/stdout.log" | tail -n 1 | cut -c1-260)"
  [ "$rc" = 3 ] && break
  if tail -n 40 "$PRE/stdout.log" | grep -q 'CUDA out of memory'; then log "FAILED: CUDA out of memory, stopping"; break; fi
done
if grep -q 'PRETRAIN STOPPED' "$PRE/train.log" 2>/dev/null; then log "STOPPED: collapse rule fired: $(grep 'PRETRAIN STOPPED' "$PRE/train.log" | tail -n 1)"; log "QUEUE35 ALL DONE"; exit 1; fi
if ! grep -q 'PRETRAIN DONE' "$PRE/train.log" 2>/dev/null || [ ! -f "$W" ]; then log "FAILED: pretraining incomplete: $(tail -n 3 "$PRE/stdout.log" | tr '\n' '|' | cut -c1-300)"; log "QUEUE35 ALL DONE"; exit 1; fi
if grep -qi 'nan' "$PRE/train.log"; then log "SKIPPED probe: NaN in the pretraining log"; log "QUEUE35 ALL DONE"; exit 1; fi
log "DONE pretraining $PRE"

BASE=(--ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drops 90 112 --eval_interval 2 --box_coef 3.0,3.0 --sim legacy)
run_one() {
  local NAME=$1; shift
  local R=$RUNS/newbox/runs/$NAME att rc
  mkdir -p "$R"
  for att in 1 2 3 4 5 6; do
    grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
    wait_ram 12000 || log "WARNING: MemAvailable stayed below 12 GB for 2 h before attempt $att of $NAME"
    log "START train $NAME attempt $att"
    : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${BASE[@]}" "$@" >> "$R/train.log" 2>&1 & TP=$!
    ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $NAME"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
    wait $TP; rc=$?
    kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
    log "END   train $NAME attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
  done
  if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $NAME: training incomplete after 6 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $NAME rc=$?"
  log "DONE $NAME: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
}
run_one hm_dinov3convnextL_pretrain3k_sk2_UNFROZEN_ridge_long_tf32_seed42_newbox_3.0_3.0_bblr1e-5 --bb dinov3_convnext_large --bb_path "$W" --seed 42 --unfreeze --grad_ckpt --lr_backbone 1e-5
log "QUEUE35 ALL DONE"
