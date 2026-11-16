#!/bin/bash
# colorbox1 (detached): replicate seeds of E5 = E1b (calibrated real-background simulator, correlated background noise) + the 4 low-q detector geometries (mask_keep, q_max 3.6-4.0)
# + ring_x_share (ring positions follow the real 41 histogram). Plain 60-epoch frozen-SimMIM recipe (ridge ring target, TF32, bs 4 x 250, head lr 3e-4, lr x0.1 at epoch 45,
# eval every 2 epochs; +nms evaluation), same as the galvani E5 run (seed 42) and the plain seeds, so the numbers are comparable.
# Usage:  SEEDS="2 3" bash heatmap/queue_realsim_e5.sh        (default SEEDS="2 3")
# One sim worker process (13 GB RAM, 0.19 s/frame) => ~190 s/epoch, ~3.3 h per run. Steps per run: smoke (40 steps) -> train -> evaluate -> visualize (nms) -> score_sweep.
# Shares the blocking GPU lock with the other queues. Status: hm_runs/_tools/queue5_status.txt, last line 'QUEUE5 ALL DONE'. Re-running skips finished steps.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
STATUS=$RUNS/_tools/queue5_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets
export GIWAXS_WORK=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
EPOCHS=60
SEEDS=${SEEDS:-"2 3"}
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue5.lock"
flock -n 9 || { log "another queue5 instance holds the lock -> exit"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"
flock 8
log "GPU lock acquired"
rdir() { local c; for c in "$RUNS/$1" "$RUNS"/*/"$1"; do [ -d "$c" ] && { echo "$c"; return; }; done; echo "$RUNS/$2/$1"; }
COMMON_SIM=("amplitude_mode='fitted'" intensity_decades=None ring_gain=0.4 ring_contrast_min=0.3 ring_chi_texture=0.05 "ring_chi_corr=(8.0,64.0)" seg_sc_scale=0.6 ring_sq_scale=1.0 ring_w_coef=2.2
            ring_no_elongate=True ring_cap=4.0 ring_chi_floor=0.5 ring_follow_bkg=True "width_jitter=(0.2,0.832)")
log "QUEUE5 START (pid $$) seeds: $SEEDS"
while pgrep -f 'heatmap/train.py' >/dev/null; do sleep 60; done
for SEED in $SEEDS; do
  TAG=E5s$SEED
  COMP=(seg_wide_frac=0.0 p_ring=0.85 "n_powder=(1,2)" "rings_cap=(6,30)" "n_oriented=(1,2)" "spots_cap=(10,60)"
        "bkg_noise_blur=(0.0,1.2)" bkg_noise_power=1.5
        "mask_keep=['ID10_Eiger2CdTe4M_D626_22p5keV_2024_03','ID10_Eiger2_4M_D606_22p5keV_2025_04','P08_Eiger4M_D629_25keV_2026_07','P08_Eiger4M_D563_22p5keV_2026_07']"
        "ring_x_share=(0.106,0.179,0.257,0.238,0.220)")
  NAME=hm_simmim_frozen_ridge_realsimE5_tf32_seed${SEED}_2.80_1.30
  R=$(rdir "$NAME" realsim)
  COMMON=(--ring_target ridge --tf32 --bs 4 --steps_per_epoch 250 --lr 3e-4 --epochs $EPOCHS --lr_drop 45 --eval_interval 2 --bb_path "$SIMMIM" --seed $SEED
          --sim realbkg --sim_workers 1 --realsim_opt "${COMMON_SIM[@]}" "${COMP[@]}")
  mkdir -p "$R"
  if grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then log "SKIP train $NAME"; else
    if [ ! -f "$R/checkpoint.pth" ]; then
      SM=$RUNS/_scratch/_smoke_realsimE5_s$SEED; rm -rf "$SM"; mkdir -p "$SM"
      if "$PY" -u heatmap/train.py --out "$SM" "${COMMON[@]}" --max_steps 40 > "$SM/smoke.log" 2>&1; then log "SMOKE $TAG ok: $(tail -n 1 "$SM/smoke.log")"
      else log "SMOKE $TAG FAILED -> stopped: $(tail -n 8 "$SM/smoke.log" | tr '\n' '|')"; exit 1; fi
    fi
    log "START train $NAME :: ${COMMON[*]}"
    : > "$R/gpu_mem.log"; : > "$R/ram_used_MiB.log"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$R/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$R/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
    "$PY" -u heatmap/train.py --out "$R" "${COMMON[@]}" >> "$R/train.log" 2>&1
    rc=$?
    kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
    log "END   train $NAME rc=$rc"
  fi
  if ! grep -q "^\[epoch $((EPOCHS-1))\] loss" "$R/train.log" 2>/dev/null; then
    log "FAILED $NAME: training incomplete. Last log lines: $(tail -n 3 "$R/train.log" | tr '\n' '|')"; log "QUEUE5 STOPPED"; exit 1
  fi
  [ -f "$R/checkpoint.pth" ] && [ ! -f "$R/final_checkpoint.pth" ] && cp "$R/checkpoint.pth" "$R/final_checkpoint.pth"
  if grep -q '+nms\] 41: ap_total' "$R/evaluate_final.txt" 2>/dev/null; then log "SKIP evaluate $TAG"; else
    "$PY" -u heatmap/evaluate.py hm="heatmap:$R/final_checkpoint.pth" > "$R/evaluate_final.txt" 2>&1; log "EVAL $TAG rc=$?"; fi
  if [ -f "$R/images/nms/41/closepairs.png" ]; then log "SKIP images $TAG"; else
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/visualize.py --ckpt "$R/final_checkpoint.pth" --out "$R/images" --sets organic 41 --thr 0.3 --decode nms > "$R/visualize_nms.log" 2>&1; log "IMAGES(nms) $TAG rc=$?"; fi
  if ls "$R"/score_sweep/*_score_sweep.txt >/dev/null 2>&1; then log "SKIP sweep $TAG"; else
    mkdir -p "$R/score_sweep"
    HM_BB_PATH="$SIMMIM" "$PY" -u heatmap/score_sweep.py "$NAME=$R/final_checkpoint.pth" --out "$R/score_sweep" --decode nms --topk 225 900 > "$R/score_sweep.log" 2>&1; log "SWEEP $TAG rc=$?"; fi
done
log "QUEUE5 ALL DONE"
