#!/bin/bash
# colorbox1 queue 15 (detached), user go 2026-11-02 (colorbox1 session): ORDER (user decisions 2026-11-02): (A) generate 60000 realsim_v2 images, then (C) resume the old realsim (E9 composition) with the legacy flips from checkpoint.pth (stopped by the user after epoch 23, no clear improvement; resumes at epoch 24), then (B) train the heatmap model on the realsim_v2 images.
#  A: realsim_v2/gen.sh v2_60k 60000 16   (CPU only, nice 19, resumable; 16 workers; watchdog in gen.sh). Data: /mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/v2_60k
#     settings: generate_dataset.py defaults + iomin_sr 0.5 (realsim_v2/gen_v2.py), min_snr 1.1, one structure per image, modes oriented/both/rings 0.25/0.5/0.25, stored layout flip_chi False.
#  B: H1 recipe (frozen SimMIM swin, ridge target, TF32, bs 4 x 250, head lr 3e-4, 60 epochs, lr drop 45, eval every 2, seed 42, --box_coef 3.0,3.0, HM_GT_BOX_COEF=auto), --sim realsim_v2 with random vertical/horizontal flips (p 0.5 each, as the legacy simulator).
#  C: same recipe on the realsim E9 composition (as queue13, 1 sim worker, peak RAM ~15 GB; 2 workers need ~26 GB and were killed by the watchdog) with --sim_flip 1 (the same random vertical/horizontal flips).
# Scoring: evaluate_final_gtscaled.txt (scaled GT only). RAM watchdog kills the training (and its sim worker) when MemAvailable < 6 GiB; a killed training resumes from checkpoint.pth (up to 4 attempts, each waits for 14 GB).
# Status: hm_runs/_tools/queue15_status.txt (last line QUEUE15 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
V2=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/v2_60k
STATUS=$RUNS/_tools/queue15_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
export GIWAXS_WORK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
export HM_BB_PATH=$SIMMIM
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF
EPOCHS=60
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue15.lock"
flock -n 9 || { log "another queue15 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }
cleanup_orphans() {
  local p pp
  for p in $(pgrep -f "heatmap-smoke/bin/python -c from multiprocessing.spawn"); do pp=$(ps -o ppid= -p "$p" | tr -d ' '); [ -n "$pp" ] && ! ps -o cmd= -p "$pp" | grep -q 'heatmap/train.py' && kill "$p" 2>/dev/null && log "killed orphan sim worker $p"; done
}
log "QUEUE15 START (pid $$)"

exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done

# run_one NAME  -- then the training args after "--"
run_one() {
  local NAME=$1; shift; local R=$RUNS/newbox/runs/$NAME att rc SAMP WD TP
  mkdir -p "$R"
  for att in 1 2 3 4; do
    grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
    wait_ram 14000 || log "WARNING: MemAvailable stayed below 14 GB for 30 min before attempt $att of $NAME"
    log "START train $NAME attempt $att"
    : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" "$@" >> "$R/train.log" 2>&1 & TP=$!
    ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $NAME"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
    wait $TP; rc=$?
    kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
    cleanup_orphans
    log "END   train $NAME attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
  done
  if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $NAME: training incomplete after 4 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if grep -q '+nms\] 41: ap_total' "$R/evaluate_final_gtscaled.txt" 2>/dev/null; then log "SKIP evaluate $NAME"; else
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $NAME rc=$?"; fi
  log "DONE $NAME: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
}
BASE=(--bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --seed 42 --box_coef 3.0,3.0)

# ---------- A: generate ----------
N_DONE=$(ls "$V2/shards"/shard_*.zip 2>/dev/null | wc -l)
if [ "$N_DONE" -ge 240 ]; then log "SKIP generation (240 shards present)"; else
  log "START generation v2_60k ($N_DONE of 240 shards done)"
  realsim_v2/gen.sh v2_60k 60000 16; log "END generation rc=$? ($(ls "$V2/shards"/shard_*.zip 2>/dev/null | wc -l) shards, failures: $(cat "$V2"/failures/*.jsonl 2>/dev/null | wc -l))"
fi
[ "$(ls "$V2/shards"/shard_*.zip 2>/dev/null | wc -l)" -ge 240 ] || { log "generation incomplete -> QUEUE15 STOPPED (rerun the script to resume)"; exit 1; }


# ---------- C: old realsim (E9 composition) with flips ----------
SIM=("amplitude_mode='fitted'" intensity_decades=None ring_gain=0.4 ring_contrast_min=0.3 ring_chi_texture=0.05 "ring_chi_corr=(8.0,64.0)" seg_sc_scale=0.6 ring_sq_scale=1.0 ring_w_coef=2.2
     ring_no_elongate=True ring_cap=4.0 ring_chi_floor=0.5 ring_follow_bkg=True "width_jitter=(0.2,0.832)" "bkg_noise_blur=(0.0,1.2)" bkg_noise_power=1.5 "mask_keep=[0,3,6,7]"
     "ring_x_share=(0.106,0.179,0.257,0.238,0.220)" seg_wide_frac=0.0 p_ring=0.85 "n_powder=(1,2)" "rings_cap=(6,30)" "n_oriented=(2,3)" "spots_cap=(25,120)" sq_q_slope=0.6 mask_wedge_only_frac=0.2)
COMMON=("${BASE[@]}" --sim realbkg --sim_workers 1 --sim_flip 1 --realsim_opt "${SIM[@]}")
run_one hm_simmim_frozen_ridge_realsimE9flip_tf32_seed42_newbox_3.0_3.0

# ---------- B: realsim_v2 ----------
COMMON=("${BASE[@]}")
run_one hm_simmim_frozen_ridge_realsimv2flip_tf32_seed42_newbox_3.0_3.0 --sim realsim_v2 --v2_data "$V2" --v2_flip random

log "QUEUE15 ALL DONE"
