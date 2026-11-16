#!/bin/bash
# colorbox1 queue 13 (detached, one job), prepared 2026-10-26, user go given in the colorbox1 session 2026-10-26: E9 re-baseline on the current setup.
# E9 = realsim composition of the Galvani run hm_simmim_frozen_ridge_realsimE9_tf32_seed42_2.80_1.30 (args.json of that run: preset r6, E6 settings + mask_wedge_only_frac 0.2, no edge fix),
# trained now with --box_coef 3.0,3.0 (the simulator draws boxes at 3,3), H1 recipe (frozen SimMIM swin, ridge target, TF32, bs 4 x 250, head lr 3e-4, 60 epochs, lr drop 45, eval every 2), seed 42.
# Training AND final scoring use HM_GT_BOX_COEF=auto (GT scaled to 3,3: evaluate_final_gtscaled.txt only); validation files newbox_visfix.
# One sim worker process (13 GB RAM, ~200 s/epoch, ~3.7 h); RAM watchdog kills the training if MemAvailable < 6 GiB.
# Status file: hm_runs/_tools/queue13_status.txt, last line QUEUE13 ALL DONE. Shares the GPU lock. Result under hm_runs/newbox/runs/.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue13_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
export GIWAXS_WORK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH HM_GT_BOX_COEF
EPOCHS=60
NAME=hm_simmim_frozen_ridge_realsimE9_tf32_seed42_newbox_3.0_3.0
R=$RUNS/newbox/runs/$NAME
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue13.lock"
flock -n 9 || { log "another queue13 instance holds the lock -> exit"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"
flock 8
log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
SIM=("amplitude_mode='fitted'" intensity_decades=None ring_gain=0.4 ring_contrast_min=0.3 ring_chi_texture=0.05 "ring_chi_corr=(8.0,64.0)" seg_sc_scale=0.6 ring_sq_scale=1.0 ring_w_coef=2.2
     ring_no_elongate=True ring_cap=4.0 ring_chi_floor=0.5 ring_follow_bkg=True "width_jitter=(0.2,0.832)" "bkg_noise_blur=(0.0,1.2)" bkg_noise_power=1.5 "mask_keep=[0,3,6,7]"
     "ring_x_share=(0.106,0.179,0.257,0.238,0.220)" seg_wide_frac=0.0 p_ring=0.85 "n_powder=(1,2)" "rings_cap=(6,30)" "n_oriented=(2,3)" "spots_cap=(25,120)" sq_q_slope=0.6 mask_wedge_only_frac=0.2)
COMMON=(--bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --seed 42 --box_coef 3.0,3.0
        --sim realbkg --sim_workers 1 --realsim_opt "${SIM[@]}")
mkdir -p "$R"
log "QUEUE13 START (pid $$)"
if grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "SKIP train $NAME"; else
  if [ ! -f "$R/checkpoint.pth" ]; then
    SM=$RUNS/_scratch/_smoke_realsimE9_newbox; rm -rf "$SM"; mkdir -p "$SM"
    if HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$SM" "${COMMON[@]}" --max_steps 40 > "$SM/smoke.log" 2>&1; then log "SMOKE ok: $(tail -n 1 "$SM/smoke.log")"
    else log "SMOKE FAILED -> stopped: $(tail -n 8 "$SM/smoke.log" | tr '\n' '|')"; exit 1; fi
  fi
  log "START train $NAME :: ${COMMON[*]}"
  : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" >> "$R/train.log" 2>&1 & TP=$!
  # watchdog: kills the training AND its sim worker child (a SIGTERM to train.py alone leaves the worker running as an orphan, 6-7 GB, as in attempts 1 and 2)
  ( while :; do av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END   train $NAME rc=$rc (peak GPU memory, sampled every 5 s: $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
fi
if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then
  log "FAILED $NAME: training incomplete. Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; log "QUEUE13 STOPPED"; exit 1
fi
[ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
if grep -q '+nms\] 41: ap_total' "$R/evaluate_final_gtscaled.txt" 2>/dev/null; then log "SKIP evaluate_gtscaled"; else
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED rc=$?"; fi
log "QUEUE13 ALL DONE"
