#!/bin/bash
# colorbox1 queue 21 (detached), user go 2026-11-04 (colorbox1 session): POSITIONS pooled_flat + WIDTHS from the same pygidSIM peaks (NOT STARTED, needs the user go; queue the next isolation step after the pooled_flat result). Was: POOLED variant (user go 2026-11-05): every position an i.i.d. draw from the pool of all pygidSIM peaks, reweighted to a uniform density over the image (--sim_pos_mode pooled_flat; smoke test: x / y shares track the legacy sim). FRESH run (aborts if the run folder already has a train.log) of the LEGACY simulator (FastSimulation, the plain H1 image creation) with ONE change:
#  the peak POSITIONS come from pygidSIM (v3 peak bank, move_fromMW=True): per frame and kind the legacy count n is kept, the n positions are a random subset of one random structure + fill from further structures
#  (simulation.py SimulationConfig.pos_bank / train.py --sim_pos_bank). Widths, intensities, NMS, gap / dark-area filters, background, noise: legacy. Purpose: find out where realsim fails to beat the legacy sim.
#  recipe = plain H1 (frozen SimMIM swin, ridge target, TF32, bs 4 x 250, head lr 3e-4, 60 epochs, lr drop 45, eval every 2, seed 42, --box_coef 3.0,3.0, HM_GT_BOX_COEF=auto, sim legacy, in-process, no sim workers).
#  Compare with plain H1 seed 42 3.0 (organic 0.6398 / 41 0.7030).
# RAM watchdog kills the training when MemAvailable < 6 GiB; a killed training resumes from ITS OWN checkpoint.pth (up to 4 attempts, each waits for 9 GB). Status: hm_runs/_tools/queue21_status.txt (last line QUEUE21 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue21_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
BANK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim/bank/bank_v3peaks_v3only.npz
RUNNAME=hm_simmim_frozen_ridge_legacyPygidPosPooledFlatWidths_tf32_seed42_newbox_3.0_3.0
export HM_BB_PATH=$SIMMIM
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF
EPOCHS=60
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue21.lock"
flock -n 9 || { log "another queue21 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }
log "QUEUE21 START (pid $$)"
[ -e "$RUNS/newbox/runs/$RUNNAME/train.log" ] && { log "ABORT: $RUNNAME already has a train.log; this queue only starts FRESH runs"; exit 1; }
[ -f "$BANK" ] || { log "ABORT: bank $BANK missing"; exit 1; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done

R=$RUNS/newbox/runs/$RUNNAME
mkdir -p "$R"
BASE=(--bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --seed 42 --box_coef 3.0,3.0)
COMMON=("${BASE[@]}" --sim legacy --sim_pos_bank "$BANK" --sim_pos_mode pooled_flat --sim_pos_widths 1)
for att in 1 2 3 4; do
  grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
  wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 30 min before attempt $att"
  log "START train $RUNNAME attempt $att"
  : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" >> "$R/train.log" 2>&1 & TP=$!
  ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $RUNNAME"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END   train $RUNNAME attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
done
if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $RUNNAME: training incomplete after 4 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; log "QUEUE21 ALL DONE"; exit 1; fi
[ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $RUNNAME rc=$?"
log "DONE $RUNNAME: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
log "QUEUE21 ALL DONE"
