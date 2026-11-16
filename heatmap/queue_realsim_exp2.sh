#!/bin/bash
# colorbox1 queue 15 (detached; copy of queue_realsim_exp.sh with train flags per plan line): realsim experiments on top of E9, run one after the other, read from a PLAN FILE that is re-read after every run (new lines can be appended while it runs).
# User go 2026-10-26 (colorbox1 session): investigate the E9 gap on 41, then run improved realsim variants overnight, look at each result, plan the next one (the colorbox1 session appends lines).
# Plan file: hm_runs/_tools/realsim_exp_plan.txt, one experiment per line:  TAG<TAB>option option ...   (realsim options as key=value, they replace / extend the E9 list below);
# '#' = comment, 'END' = stop when everything before it is done. Marker of a finished experiment: hm_runs/_tools/realsim_exp_done/<TAG>.
# Every experiment: smoke (first one only) -> train (H1 recipe, 60 epochs, head lr 3e-4, lr drop 45, seed 42, frozen SimMIM, ridge, TF32, --box_coef 3.0,3.0, HM_GT_BOX_COEF=auto: scaled GT only)
#   -> evaluate_final_gtscaled.txt -> diag/ (detection dump, miss analysis against the plain H1 seed-42 3,3 model, score sweep; all scaled GT 3,3).
# RAM: watchdog kills the training AND its sim workers when MemAvailable < 6 GiB; a killed training is resumed from checkpoint.pth (up to 4 attempts, each waits for MemAvailable >= 14 GB first).
# Sim workers: 2 if the smoke test leaves >= 9.5 GB available, else 1.
# Status: hm_runs/_tools/queue14_status.txt. Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
PLAN=$RUNS/_tools/realsim_exp_plan2.txt
DONE=$RUNS/_tools/realsim_exp_done
STATUS=$RUNS/_tools/queue15_status.txt
PLAIN_DUMP=$RUNS/_scratch/e9_diag/dets_plain_seed42_3.0.pkl
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
export GIWAXS_WORK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
export HM_BB_PATH=$SIMMIM
unset HM_SCORE_FLOOR HM_AMP HM_GT_BOX_COEF
EPOCHS=60
cd "$REPO" || exit 1
mkdir -p "$DONE"
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue15.lock"
flock -n 9 || { log "another queue15 instance holds the lock -> exit"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "QUEUE15 START (pid $$), waiting for the GPU lock"
flock 8
log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }
cleanup_orphans() {      # sim workers of a killed training (spawn children of this env whose parent is no longer a train.py)
  local p pp
  for p in $(pgrep -f "heatmap-smoke/bin/python -c from multiprocessing.spawn"); do pp=$(ps -o ppid= -p "$p" | tr -d ' '); [ -n "$pp" ] && ! ps -o cmd= -p "$pp" | grep -q 'heatmap/train.py' && kill "$p" 2>/dev/null && log "killed orphan sim worker $p"; done
}
# merged option list: E9 base, overridden / extended by the plan line -> one option per line
merge_opts() {
  "$PY" -I - "$@" <<'E'
import sys
base = ["amplitude_mode='fitted'", 'intensity_decades=None', 'ring_gain=0.4', 'ring_contrast_min=0.3', 'ring_chi_texture=0.05', 'ring_chi_corr=(8.0,64.0)', 'seg_sc_scale=0.6', 'ring_sq_scale=1.0',
        'ring_w_coef=2.2', 'ring_no_elongate=True', 'ring_cap=4.0', 'ring_chi_floor=0.5', 'ring_follow_bkg=True', 'width_jitter=(0.2,0.832)', 'bkg_noise_blur=(0.0,1.2)', 'bkg_noise_power=1.5',
        'mask_keep=[0,3,6,7]', 'ring_x_share=(0.106,0.179,0.257,0.238,0.220)', 'seg_wide_frac=0.0', 'p_ring=0.85', 'n_powder=(1,2)', 'rings_cap=(6,30)', 'n_oriented=(2,3)', 'spots_cap=(25,120)',
        'sq_q_slope=0.6', 'mask_wedge_only_frac=0.2']
for o in sys.argv[1:]:
    k = o.split('=', 1)[0]; base = [b for b in base if b.split('=', 1)[0] != k] + [o]
print('\n'.join(base))
E
}
WORKERS=""
smoke_workers() {        # decide 1 or 2 sim workers from the RAM left after a 40-step smoke with 2 workers
  local SM=$RUNS/_scratch/_smoke_realsim_exp mn=999999 v TP
  rm -rf "$SM"; mkdir -p "$SM"
  HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$SM" "${COMMON[@]}" --sim_workers 2 --max_steps 40 > "$SM/smoke.log" 2>&1 & TP=$!
  while kill -0 $TP 2>/dev/null; do v=$(avail); [ "$v" -lt "$mn" ] && mn=$v; sleep 2; done
  wait $TP; local rc=$?
  log "SMOKE rc=$rc (2 workers): min MemAvailable ${mn} MiB; $(tail -n 1 "$SM/smoke.log" | cut -c1-160)"
  [ $rc -ne 0 ] && { log "SMOKE FAILED -> queue stopped: $(tail -n 8 "$SM/smoke.log" | tr '\n' '|')"; exit 1; }
  if [ "$mn" -ge 9500 ]; then WORKERS=2; else WORKERS=1; fi
  log "using $WORKERS sim worker(s)"
}
one_exp() {              # one_exp <TAG> <options...>
  local TAG=$1; shift
  local NAME=hm_simmim_frozen_ridge_realsim${TAG}_tf32_seed42_newbox_3.0_3.0
  local R=$RUNS/newbox/runs/$NAME att rc SAMP WD TP last
  mkdir -p "$R"
  local RS=() TF=() tok nm
  for tok in "$@"; do if [[ "$tok" == +* ]]; then nm="${tok#+}"; TF+=("--${nm%%=*}" "${tok#*=}"); else RS+=("$tok"); fi; done
  mapfile -t OPTS < <(merge_opts "${RS[@]}")
  COMMON=(--bb simmim1 --bb_path "$SIMMIM" --ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --seed 42 --box_coef 3.0,3.0
          --sim realbkg "${TF[@]}" --realsim_opt "${OPTS[@]}")
  [ -z "$WORKERS" ] && smoke_workers
  for att in 1 2 3 4; do
    grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null && break
    wait_ram 14000 || log "WARNING: MemAvailable stayed below 14 GB for 30 min before attempt $att of $TAG"
    log "START train $NAME attempt $att (workers $WORKERS) :: opts $*"
    : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" --sim_workers $WORKERS >> "$R/train.log" 2>&1 & TP=$!
    ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the training $TAG"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
    wait $TP; rc=$?
    kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
    cleanup_orphans
    log "END   train $TAG attempt $att rc=$rc (peak GPU $(sort -n "$R/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$R/ram_used_MiB.log" | tail -n 1) MiB)"
  done
  if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "FAILED $TAG: training incomplete after 4 attempts: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; touch "$DONE/$TAG"; echo failed > "$DONE/$TAG"; return 1; fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if grep -q '+nms\] 41: ap_total' "$R/evaluate_final_gtscaled.txt" 2>/dev/null; then log "SKIP evaluate $TAG"; else
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $TAG rc=$?"; fi
  mkdir -p "$R/diag/sweep"
  HM_GT_BOX_COEF=3.0,3.0 "$PY" -u heatmap/dump_dets.py --ckpt "$R/final_checkpoint.pth" --out "$R/diag/dets.pkl" --box_coef 3.0,3.0 > "$R/diag/dump.log" 2>&1; log "DUMP $TAG rc=$?"
  "$PY" -u heatmap/miss_analysis.py "$PLAIN_DUMP" "$R/diag/dets.pkl" --sets 41 organic --thr 0.3 > "$R/diag/miss_analysis_plain_vs_$TAG.txt" 2>&1; log "MISS $TAG rc=$?"
  HM_GT_BOX_COEF=3.0,3.0 "$PY" -u heatmap/score_sweep.py "$TAG=$R/final_checkpoint.pth" --out "$R/diag/sweep" --decode nms --topk 225 900 > "$R/diag/sweep/score_sweep.log" 2>&1; log "SWEEP $TAG rc=$?"
  rm -f "$R/diag/dets.pkl.tmp"
  echo ok > "$DONE/$TAG"; log "DONE $TAG: $(grep -E '\[hm\+nms\] (organic|41): ap_total' "$R/evaluate_final_gtscaled.txt" | sed 's/(AP by.*//' | tr -s ' ' | tr '\n' ';')"
}
while :; do
  line=$(grep -v '^[[:space:]]*#' "$PLAN" 2>/dev/null | awk -F'\t' -v d="$DONE" 'NF>=1 && $1!="" { if ($1=="END") {print "END"; exit} cmd="test -e " d "/" $1; if (system(cmd)!=0) {print; exit} }')
  if [ -z "$line" ]; then sleep 60; continue; fi
  if [ "$line" = "END" ]; then log "QUEUE15 ALL DONE (END marker reached)"; exit 0; fi
  TAG=$(echo "$line" | cut -f1); OPTLINE=$(echo "$line" | cut -f2-)
  read -r -a OA <<< "$OPTLINE"
  log "NEXT experiment $TAG: ${OA[*]}"
  one_exp "$TAG" "${OA[@]}"
done
