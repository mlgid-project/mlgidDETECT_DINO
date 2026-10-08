#!/bin/bash
# Overnight queue on colorbox1 (run detached: nohup bash heatmap/overnight.sh > .../overnight.log 2>&1 &).
# 0) smoke  1) lr test (lr 1e-4, 60 ep)  2) boxconv1 frozen backbone (60 ep)  3) long run (batch 8, 120 ep, 2 lr drops).
# All use the ridge ring target and TF32; every run is scored with heatmap/evaluate.py (native + +nms) at its end.
REPO=${REPO:-/home/nicolerch/Documents/DINO/mlgidDETECT_DINO_HEATMAP}
PY=${PY:-/home/nicolerch/miniconda3/envs/heatmap-smoke/bin/python}
RUNS=${RUNS:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs}
SIMMIM=${SIMMIM:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth}
BOXCONV=${BOXCONV:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/boxconv1_backbone.pth}
export HM_DATA_DIR=${HM_DATA_DIR:-$HOME/Documents/datasets}
STATUS=$RUNS/_tools/overnight_status.txt
mkdir -p "$RUNS/_tools" "$RUNS/_interim"
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

run_train() {   # run_train <name> <train.py args...>
  local name=$1; shift
  mkdir -p "$RUNS/$name"
  log "START $name :: $*"
  "$PY" -u heatmap/train.py --out "$RUNS/$name" "$@" >> "$RUNS/$name/train.log" 2>&1
  log "END   $name rc=$?"
  [ -f "$RUNS/$name/checkpoint.pth" ] && cp "$RUNS/$name/checkpoint.pth" "$RUNS/$name/final_checkpoint.pth"
  "$PY" -u heatmap/evaluate.py hm=heatmap:"$RUNS/$name/final_checkpoint.pth" > "$RUNS/$name/evaluate_final.txt" 2>&1
  log "EVAL  $name rc=$?"
  # images on the real frames (GT / matches / false positives, heatmaps, close-pair crops); non-fatal
  HM_BB_PATH="${HM_BB_OVERRIDE:-$SIMMIM}" "$PY" -u heatmap/visualize.py --ckpt "$RUNS/$name/final_checkpoint.pth" \
      --out "$RUNS/$name/images" --sets organic 41 --thr 0.3 > "$RUNS/$name/visualize.log" 2>&1
  log "IMAGES $name rc=$?"
}

# -1) checks: vectorised ring targets == reference (else fall back to the slow reference builder); weights really load
if "$PY" heatmap/test_targets.py 60 >> "$STATUS" 2>&1; then log "targets test ok"; else
  log "TARGETS TEST FAILED -> using the reference (slower) builder, HM_REF_TARGETS=1"; export HM_REF_TARGETS=1; fi
"$PY" heatmap/check_backbone.py "$SIMMIM" >> "$STATUS" 2>&1 || { log "SimMIM weights check FAILED -> abort"; exit 1; }

# 0) smoke: 1 epoch of 8 steps, exercising ridge + tf32 + multi-step lr + the +nms logging + eval
SM=$RUNS/_interim/smoke_overnight; rm -rf "$SM"; mkdir -p "$SM"
"$PY" -u heatmap/train.py --out "$SM" --bb_path "$SIMMIM" --ring_target ridge --tf32 --lr_drops 5 7 \
      --epochs 1 --steps_per_epoch 8 --eval_interval 1 > "$SM/smoke.log" 2>&1
if ! grep -q "+nms (deployed)" "$SM/smoke.log"; then log "SMOKE FAILED -> abort (see $SM/smoke.log)"; exit 1; fi
log "SMOKE ok"

# 1) lr test: only lr changes vs the ridge run (1e-4 instead of 3e-4); TF32 on
A=hm_ridge_lr1e-4_tf32_2.80_1.30
run_train $A --bb_path "$SIMMIM" --ring_target ridge --tf32 --lr 1e-4 --epochs 60 --lr_drop 45 --eval_interval 1

# 2) boxconv1 backbone (frozen, 2.80/1.30 convention), ridge recipe at the original lr 3e-4
B=hm_boxconv1_frozen_ridge_tf32_2.80_1.30
# (checked here, not at start, so the weights file may arrive while run A is training)
BOX_OK=0; for i in $(seq 1 30); do [ -f "$BOXCONV" ] && break; sleep 60; done
"$PY" heatmap/check_backbone.py "$BOXCONV" backbone.0. >> "$STATUS" 2>&1 && BOX_OK=1 || log "boxconv weights missing/invalid -> run B skipped"
[ $BOX_OK = 1 ] && HM_BB_OVERRIDE="$BOXCONV" run_train $B --bb ssl1 --bb_path "$BOXCONV" --ring_target ridge --tf32 --lr 3e-4 --epochs 60 --lr_drop 45 --eval_interval 1

# 3) long run: batch 8 (125 steps = 1000 images per epoch), 120 epochs, lr drops at 90 and 112; lr from the lr test
LRL=$("$PY" heatmap/pick_lr.py "$RUNS/$A/evaluate_final.txt" 2>> "$STATUS" | tail -1)
log "long-run lr = $LRL"
run_train hm_simmim_frozen_ridge_long_tf32_2.80_1.30 --bb_path "$SIMMIM" --ring_target ridge --tf32 --bs 8 \
      --steps_per_epoch 125 --lr "$LRL" --epochs 120 --lr_drops 90 112 --eval_interval 2
log "ALL DONE"
