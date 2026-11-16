#!/bin/bash
# colorbox1 weekend queue 11 (detached, one job at a time), prepared 2026-10-23, requested by the galvani session (user go needed): 16 runs, frozen SimMIM swin, ridge target, TF32, legacy simulator, bs 4 x 250, eval every 2,
# newbox_visfix files with the clamp patch. Training AND final scoring use HM_GT_BOX_COEF=auto (GT scaled to the run's box convention; scaled GT only: evaluate_final_gtscaled.txt, no fixed-GT final eval, no score sweep/images).
# A head-lr sweep at 3.0,3.0 (lr 1e-3, 1e-4, 6e-4; seeds 1, 2)  B longer schedule at 3.0,3.0 (120 ep, lr drop 90; seeds 1, 2)  C box shape 3.0,2.0 and 2.0,3.0 (a_coef,w_coef; seeds 1, 2)
# D seed top-ups on colorbox: 2.5,2.5 seeds 1, 2 and 3.0,3.0 seeds 1, 2.
# Order: A s1 (3), B s1, C s1 (2), A s2 (3), B s2, C s2 (2), D (4). Run names follow the Galvani sbatch suffix style (_lr<lr>, _ep<epochs>), without the _galvani suffix. A failing run is logged and the queue continues.
# Status file: hm_runs/_tools/queue11_status.txt, last line QUEUE11 ALL DONE. Shares the GPU lock. Results under hm_runs/newbox/.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue11_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH HM_GT_BOX_COEF
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue11.lock"
flock -n 9 || { log "another queue11 instance holds the lock -> exit"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"
flock 8
log "GPU lock acquired"
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
  if grep -q '+nms\] 41: ap_total' "$R/evaluate_final_gtscaled.txt" 2>/dev/null; then log "SKIP evaluate_gtscaled $name"; else
    HM_GT_BOX_COEF=auto "$PY" -u heatmap/evaluate.py hm=heatmap:"$R/final_checkpoint.pth" > "$R/evaluate_final_gtscaled.txt" 2>&1; log "EVAL_GTSCALED $name rc=$?"; fi
}

log "QUEUE11 START (pid $$)"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
# plan <name suffix after "newbox_"> <box_coef> <seed> [lr] [epochs] [lr_drop]   (defaults lr 3e-4, 60 epochs, drop 45)
plan() {
  local name=hm_simmim_frozen_ridge_tf32_seed$3_newbox_$1 coef=$2 seed=$3 lr=${4:-3e-4} ep=${5:-60} drop=${6:-45}
  one_run "$name" $((ep - 1)) --bs 4 --steps_per_epoch 250 --lr "$lr" --epochs "$ep" --lr_drop "$drop" --eval_interval 2 --seed "$seed" --box_coef "$coef" || log "QUEUE11: $name failed, continuing"
}
# A s1, B s1, C s1
plan 3.0_3.0_lr1e-3 3.0,3.0 1 1e-3
plan 3.0_3.0_lr1e-4 3.0,3.0 1 1e-4
plan 3.0_3.0_lr6e-4 3.0,3.0 1 6e-4
plan 3.0_3.0_ep120 3.0,3.0 1 3e-4 120 90
plan 3.0_2.0 3.0,2.0 1
plan 2.0_3.0 2.0,3.0 1
# A s2, B s2, C s2
plan 3.0_3.0_lr1e-3 3.0,3.0 2 1e-3
plan 3.0_3.0_lr1e-4 3.0,3.0 2 1e-4
plan 3.0_3.0_lr6e-4 3.0,3.0 2 6e-4
plan 3.0_3.0_ep120 3.0,3.0 2 3e-4 120 90
plan 3.0_2.0 3.0,2.0 2
plan 2.0_3.0 2.0,3.0 2
# D seed top-ups
plan 2.50_2.50 2.5,2.5 1
plan 2.50_2.50 2.5,2.5 2
plan 3.00_3.00 3.0,3.0 1
plan 3.00_3.00 3.0,3.0 2
log "QUEUE11 ALL DONE"
