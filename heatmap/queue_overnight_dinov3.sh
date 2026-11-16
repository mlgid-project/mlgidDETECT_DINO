#!/bin/bash
# colorbox1 queue 24 (detached), user go 2026-11-05 (colorbox1 session, "queue the DINOv3 runs just in case i am not back"): waits for QUEUE23 ALL DONE (E13a GNoME + flat), then in this order, each independent of the others:
#  1) frozen DINOv3 ConvNeXt-Large instead of the SimMIM swin, 2) frozen DINOv3 ViT-L (models/dinov3_backbone.py), both with the plain H1 recipe on the legacy sim: compare with plain H1 (0.6435 / 0.6993),
#  3) second seed of legacy + pooled_flat positions (seed 42: 0.6549 / 0.6911). recipe: ridge target, TF32, bs 4 x 250, head lr 3e-4, 60 epochs, lr drop 45, eval every 2, --box_coef 3.0,3.0, HM_GT_BOX_COEF=auto.
# To cancel: kill this script while it waits (status file says where it is). HM_BB_PATH must stay unset: evaluate.py would otherwise load the wrong backbone (each run stores its own bb_path).
# RAM watchdog kills a training when MemAvailable < 6 GiB; it resumes from ITS OWN checkpoint.pth (up to 4 attempts, each waits for 9 GB). Status: hm_runs/_tools/queue24_status.txt (last line QUEUE24 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
DV3=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/dinov3
BANK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim/bank/bank_v3peaks_v3only.npz
STATUS=$RUNS/_tools/queue24_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF HM_BB_PATH GIWAXS_BANK
EPOCHS=60
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue24.lock"
flock -n 9 || { log "another queue24 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }
log "QUEUE24 START (pid $$), waiting for QUEUE23 ALL DONE"
until grep -q 'QUEUE23 ALL DONE' "$RUNS/_tools/queue23_status.txt" 2>/dev/null; do sleep 60; done
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done

BASE=(--ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --box_coef 3.0,3.0 --sim legacy)
# run_one NAME  -- then the training args
run_one() {
  local NAME=$1; shift; local R=$RUNS/newbox/runs/$NAME att rc SAMP WD TP
  mkdir -p "$R"
  for att in 1 2 3 4; do
    grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
    wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 30 min before attempt $att of $NAME"
    log "START train $NAME attempt $att"
    : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${BASE[@]}" "$@" >> "$R/train.log" 2>&1 & TP=$!
    ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $NAME"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
    wait $TP; rc=$?
    kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
    log "END   train $NAME attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
  done
  if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $NAME: training incomplete after 4 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $NAME rc=$?"
  log "DONE $NAME: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
}

run_one hm_dinov3convnextL_frozen_ridge_tf32_seed42_newbox_3.0_3.0 --bb dinov3_convnext_large --bb_path "$DV3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth" --seed 42
run_one hm_dinov3vitL_frozen_ridge_tf32_seed42_newbox_3.0_3.0 --bb dinov3_vitl16 --bb_path "$DV3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" --seed 42
run_one hm_simmim_frozen_ridge_legacyPygidPosPooledFlat_tf32_seed2_newbox_3.0_3.0 --bb simmim1 --bb_path "$SIMMIM" --seed 2 --sim_pos_bank "$BANK" --sim_pos_mode pooled_flat
log "QUEUE24 ALL DONE"
