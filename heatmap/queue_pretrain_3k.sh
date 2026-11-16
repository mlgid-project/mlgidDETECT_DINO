#!/bin/bash
# colorbox1 queue 28 (detached), user go 2026-11-06 ("already queue the 12 hour pretraining"): waits for QUEUE27 ALL DONE, then runs the short DINOv3 ConvNeXt-L pretraining test, 3,000 optimizer steps of about 64 frames, 300 of them heads only.
# Batch size comes from the benchmark: the fastest one that finished with peak allocated at most MAX_GB, accum so that bs * accum is about 64. Main corpus only, hold-out excluded.
# RAM watchdog kills the run when MemAvailable < 6 GiB; it resumes from its own checkpoint.pth (up to 4 attempts, each waits for 9 GB). An out-of-memory error stops the queue, no recipe change.
# Status: hm_runs/_tools/queue28_status.txt (last line QUEUE28 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
DATA=${Q28_DATA:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP}
RUNS=$DATA/hm_runs
STATUS=$RUNS/_tools/queue28_status.txt
EXCL=$DATA/moneta_sweep/holdout/v1_confirmed_2026-11-06/pretrain_exclude.tsv
OUT=$DATA/pretrain_runs/convnextL_3k
STEPS=3000
MAX_GB=10.5
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue28.lock"
flock -n 9 || { log "another queue28 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }

pick_bs() {   # prints "bs accum frames/s peak"; frames/s from the steps with the backbone open, the cumulative figure in the log includes startup and the heads-only steps
  "$PY" - "$DATA/pretrain_runs" "$MAX_GB" <<'EOF'
import re, sys
root, max_gb = sys.argv[1], float(sys.argv[2])
best = None
for bs in (2, 3, 4, 6):
    try:
        text = open(f'{root}/bench_bs{bs}/stdout.log').read()
    except OSError:
        continue
    r = {int(s): (float(f), float(p)) for s, f, p in re.findall(r'^step (\d+)/\d+ .* ([\d.]+) frames/s peak ([\d.]+) GB', text, re.M)}
    if 'CUDA out of memory' in text or 60 not in r or 40 not in r or r[60][1] > max_gb:
        continue
    rate = 20 * bs / (60 * bs / r[60][0] - 40 * bs / r[40][0])
    if best is None or rate > best[2]:
        best = (bs, round(64 / bs), rate, r[60][1])
if best:
    print(best[0], best[1], round(best[2], 2), best[3])
EOF
}

log "QUEUE28 START (pid $$), waiting for QUEUE27 ALL DONE"
until grep -q 'QUEUE27 ALL DONE' "$RUNS/_tools/queue27_status.txt" 2>/dev/null; do sleep 60; done
[ -n "$Q28_DRY" ] && { log "DRY pick: $(pick_bs)"; exit 0; }
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
read -r BS ACCUM RATE PEAK <<< "$(pick_bs)"
if [ -z "$BS" ]; then log "FAILED: no batch size finished the benchmark at most $MAX_GB GB"; log "QUEUE28 ALL DONE"; exit 1; fi
log "benchmark pick: bs $BS accum $ACCUM, $RATE frames/s, peak $PEAK GB"
mkdir -p "$OUT"
for att in 1 2 3 4; do
  grep -q 'PRETRAIN DONE' "$OUT/train.log" 2>/dev/null && break
  wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 30 min before attempt $att"
  log "START pretrain attempt $att"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$OUT/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  "$PY" -u heatmap/pretrain_dinov3.py --out "$OUT" --corpus "$DATA/corpus_v2/scans" --exclude "$EXCL" --bs "$BS" --accum "$ACCUM" --workers 6 --steps $STEPS --head_steps 300 --log_every 10 --save_every 250 --grad_ckpt >> "$OUT/stdout.log" 2>&1 & TP=$!
  ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the pretraining"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END pretrain attempt $att rc=$rc (peak GPU $(sort -n "$OUT/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$OUT/ram_used_MiB.log" | tail -n 1) MiB): $(grep -E '^step' "$OUT/stdout.log" | tail -n 1 | cut -c1-130)"
  if tail -n 40 "$OUT/stdout.log" | grep -q 'CUDA out of memory'; then log "FAILED: CUDA out of memory, stopping"; break; fi
done
if grep -q 'PRETRAIN DONE' "$OUT/train.log" 2>/dev/null; then log "DONE $OUT/backbone_teacher.pth"; else log "FAILED: pretraining incomplete: $(tail -n 3 "$OUT/stdout.log" | tr '\n' '|' | cut -c1-300)"; fi
log "QUEUE28 ALL DONE"
