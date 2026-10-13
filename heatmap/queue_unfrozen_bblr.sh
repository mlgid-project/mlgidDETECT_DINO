#!/bin/bash
# colorbox1 queue (detached, one job at a time), planned 2026-10-13 to fill the machine until ~09:00 on 2026-10-14. Every run: plain 60-epoch recipe (--tf32 --ring_target ridge --bs 4
# --steps_per_epoch 250 --lr 3e-4 --lr_drop 45), swin UNFROZEN with gradient checkpointing, stochastic depth off, legacy simulator, seed 1; post steps as in queue_coord_unfrozen_b.sh.
#   1. unfrozen boxconv1 backbone, backbone lr 4e-5   (do two +0.03 gains on 41 add up? memory: only the SimMIM version has been measured, 11.4 of 12.2 GB)   ~6.2 h
#   2. unfrozen SimMIM backbone, backbone lr 4e-5     (does the galvani 4e-5 gain on 41 hold at this recipe? colorbox 1e-5 seeds: organic 0.602 / 41 0.762)       ~6.2 h
#   3. unfrozen SimMIM backbone, backbone lr 8e-5     (trend check: 1e-5 -> 4e-5 gave +0.017 on 41, -0.015 on organic)                                           ~6.2 h
# Total ~19 h + ~5 min post steps per run. A failed training (e.g. out of memory) is logged and the queue CONTINUES with the next planned run (no recipe is changed).
# Status file: hm_runs/_tools/queue6_status.txt, last line 'QUEUE6 ALL DONE'. Re-running skips finished steps. Shares the blocking GPU lock with the other queues.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
BOXCONV=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/boxconv1_backbone.pth
STATUS=$RUNS/_tools/queue6_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
EPOCHS=60
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue6.lock"
flock -n 9 || { log "another queue6 instance holds the lock -> exit"; exit 0; }
# one training at a time on the 12 GB card: queue6 and queue3 share this blocking lock (held until the script exits), whichever was started first runs first
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"
flock 8
log "GPU lock acquired"
train_done()  { grep -q "^\[epoch $((EPOCHS-1))\] loss" "$1/train.log" 2>/dev/null; }
eval_done()   { grep -q '+nms\] 41: ap_total' "$1/evaluate_final.txt" 2>/dev/null; }
images_done() { [ -f "$1/images/nms/41/closepairs.png" ]; }
sweep_done()  { ls "$1"/score_sweep/*_score_sweep.txt >/dev/null 2>&1; }

# rdir <run name> <group>: where a run lives -- at the top level of hm_runs, in any group subfolder (see heatmap/runs_layout.py), else in <group> (new run)
rdir() { local c; for c in "$RUNS/$1" "$RUNS"/*/"$1"; do [ -d "$c" ] && { echo "$c"; return; }; done; echo "$RUNS/$2/$1"; }
# one_run <name> <bb file for visualize/sweep> <group folder> <train.py args...>
one_run() {
  local name=$1 postbb=$2 grp=$3; shift 3
  local R rc peak SAMP; R=$(rdir "$name" "$grp")
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

UNF_BB() { echo "--bb_path $1 --chan he --unfreeze --lr_backbone $2 --bb_drop_path 0 --grad_ckpt"; }
log "QUEUE6 START (pid $$)"
while pgrep -f 'heatmap/train.py' >/dev/null; do sleep 60; done
one_run hm_boxconv1_unfrozen_ridge_tf32_seed1_2.80_1.30_bblr4e-5 "$BOXCONV" unfrozen --bb ssl1 $(UNF_BB $BOXCONV 4e-5) --seed 1 || log "QUEUE6: boxconv1 unfrozen failed, continuing"
one_run hm_simmim_unfrozen_ridge_he_tf32_seed1_2.80_1.30_bblr4e-5 "$SIMMIM" unfrozen $(UNF_BB $SIMMIM 4e-5) --seed 1 || log "QUEUE6: simmim bblr 4e-5 failed, continuing"
one_run hm_simmim_unfrozen_ridge_he_tf32_seed1_2.80_1.30_bblr8e-5 "$SIMMIM" unfrozen $(UNF_BB $SIMMIM 8e-5) --seed 1 || log "QUEUE6: simmim bblr 8e-5 failed, continuing"
log "QUEUE6 ALL DONE"
