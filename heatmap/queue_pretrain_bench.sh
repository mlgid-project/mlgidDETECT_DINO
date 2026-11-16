#!/bin/bash
# colorbox1 queue 27 (detached), user go 2026-11-06 ("queue the benchmark"): waits for QUEUE26 ALL DONE, then measures memory and speed of heatmap/pretrain_dinov3.py on this GPU, 60 optimizer steps per batch size, backbone unfrozen from step 30.
# Benchmark only, keeps no weights. Status: hm_runs/_tools/queue27_status.txt (last line QUEUE27 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
DATA=/mnt/DATA/mlgidDETECT_DINO_HEATMAP
RUNS=$DATA/hm_runs
STATUS=$RUNS/_tools/queue27_status.txt
EXCL=$DATA/moneta_sweep/holdout/v1_confirmed_2026-11-06/pretrain_exclude.tsv
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue27.lock"
flock -n 9 || { log "another queue27 instance holds the lock -> exit"; exit 0; }
log "QUEUE27 START (pid $$), waiting for QUEUE26 ALL DONE"
until grep -q 'QUEUE26 ALL DONE' "$RUNS/_tools/queue26_status.txt" 2>/dev/null; do sleep 60; done
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
for BS in 2 3 4 6; do
  OUT=$DATA/pretrain_runs/bench_bs$BS
  rm -rf "$OUT"; mkdir -p "$OUT"
  log "START bench bs $BS"
  timeout 2400 "$PY" -u heatmap/pretrain_dinov3.py --out "$OUT" --corpus "$DATA/corpus_v2/scans" --exclude "$EXCL" --bs $BS --accum 1 --workers 6 --steps 60 --head_steps 30 --log_every 10 --save_every 100000 --grad_ckpt > "$OUT/stdout.log" 2>&1
  rc=$?
  log "END bench bs $BS rc=$rc: $(grep -E '^step' "$OUT/stdout.log" | tail -n 1 | cut -c1-130) $(grep -m1 -oE 'CUDA out of memory' "$OUT/stdout.log")"
  rm -f "$OUT"/checkpoint.pth* "$OUT"/backbone_teacher.pth
  grep -q 'CUDA out of memory' "$OUT/stdout.log" && break
done
log "QUEUE27 ALL DONE"
