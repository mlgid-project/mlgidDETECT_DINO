#!/bin/bash
# colorbox1 queue 42 (user go 2026-11-16): detections of every complete 3.0,3.0 heatmap model on the 49 labelled frames (organic 8 + 41 41), for the per-region sweep.
# One pickle per model in latent_map/sweep/dets/<run>.pkl (dump_dets.py --no_img, labels scaled to 3.0,3.0; boxes = +nms deployed decode, floor 0.1, top-225). Skips models that already have a pickle.
# Status: hm_runs/_tools/queue42_status.txt (last line QUEUE42 ALL DONE). Waits for the GPU lock and for free RAM (GUI_project sweeps have priority).
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
OUT=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/latent_map/sweep
STATUS=$RUNS/_tools/queue42_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix HM_GT_BOX_COEF=3.0,3.0 HM_DINOV3_REPO=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/dinov3/dinov3_repo
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH GIWAXS_BANK
cd "$REPO" || exit 1
mkdir -p "$OUT/dets"
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue42.lock"
flock -n 9 || { log "another queue42 instance holds the lock -> exit"; exit 0; }
log "QUEUE42 START (pid $$)"
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
"$PY" heatmap/sweep_list.py --runs "$RUNS" --out "$OUT/models.tsv" | tee -a "$STATUS"
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
n=0; ok=0; bad=0
tail -n +2 "$OUT/models.tsv" | while IFS=$'\t' read -r run dir ckpt rest; do
  n=$((n+1)); f="$OUT/dets/$run.pkl"
  [ -f "$f" ] && continue
  for i in $(seq 1 120); do [ "$(avail)" -ge 9000 ] && break; sleep 30; done      # up to 1 h for RAM, then try anyway
  if "$PY" -u heatmap/dump_dets.py --ckpt "$dir/$ckpt" --out "$f.tmp" --no_img > "$OUT/dets/$run.log" 2>&1 && [ -s "$f.tmp" ]; then mv "$f.tmp" "$f"; ok=$((ok+1)); else rm -f "$f.tmp"; bad=$((bad+1)); log "FAILED $run: $(tail -n 1 "$OUT/dets/$run.log" | cut -c1-150)"; fi
  [ $((n % 10)) = 0 ] && log "progress: $n models seen, $(ls "$OUT/dets"/*.pkl 2>/dev/null | wc -l) pickles"
done
log "pickles: $(ls "$OUT/dets"/*.pkl | wc -l) of $(($(wc -l < "$OUT/models.tsv") - 1))"
log "QUEUE42 ALL DONE"
