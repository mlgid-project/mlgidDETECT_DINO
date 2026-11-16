#!/bin/bash
# colorbox1 queue 31 (detached), user go 2026-11-07: waits for the agent run deepseek_guide9_full to end, then pretrains DINOv3 ConvNeXt-L from the public weights for 10,000 optimizer steps of 66 frames, 300 of them heads only.
# Both corpora, hold-out v2 excluded (moneta_sweep/holdout/pretrain_exclude_v2.tsv), random chi flips. Otherwise the recipe of queue 28: bs 6, accum 11, lr, heads, losses.
# RAM watchdog kills the run when MemAvailable < 6 GiB; it resumes from its own checkpoint.pth (up to 6 attempts, each waits for 9 GB). An out-of-memory error stops the queue, no recipe change.
# Status: hm_runs/_tools/queue31_status.txt (last line QUEUE31 ALL DONE). Shares the GPU lock.
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
DATA=/mnt/DATA/mlgidDETECT_DINO_HEATMAP
RUNS=$DATA/hm_runs
STATUS=$RUNS/_tools/queue31_status.txt
EXCL=$DATA/moneta_sweep/holdout/pretrain_exclude_v2.tsv
OUT=$DATA/pretrain_runs/convnextL_10k_both
STEPS=10000
cd "$REPO" || exit 1
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue31.lock"
flock -n 9 || { log "another queue31 instance holds the lock -> exit"; exit 0; }
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
wait_ram() { local i; for i in $(seq 1 60); do [ "$(avail)" -ge "$1" ] && return 0; sleep 30; done; return 1; }

log "QUEUE31 START (pid $$), waiting for the agent run deepseek_guide9_full"
while pgrep -f 'run_frames_paralle[l].sh .* deepseek_guide9_full' >/dev/null; do sleep 60; done
log "agent run ended"
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
mkdir -p "$OUT"
for att in 1 2 3 4 5 6; do
  grep -q 'PRETRAIN DONE' "$OUT/train.log" 2>/dev/null && break
  wait_ram 9000 || log "WARNING: MemAvailable stayed below 9 GB for 30 min before attempt $att"
  log "START pretrain attempt $att"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/gpu_mem.log"; free -m | awk 'NR==2{print $3}' >> "$OUT/ram_used_MiB.log"; sleep 5; done ) & SAMP=$!
  "$PY" -u heatmap/pretrain_dinov3.py --out "$OUT" --corpus "$DATA/corpus_v2/scans" "$DATA/corpus_v2/scans_other" --exclude "$EXCL" --flip_chi --bs 6 --accum 11 --workers 6 --steps $STEPS --head_steps 300 --log_every 10 --save_every 250 --grad_ckpt >> "$OUT/stdout.log" 2>&1 & TP=$!
  ( while :; do av=$(avail); if [ "$av" -lt 6144 ]; then log "WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> killing the pretraining"; pkill -TERM -P $TP; kill $TP; fi; sleep 10; done ) & WD=$!
  wait $TP; rc=$?
  kill $SAMP $WD 2>/dev/null; wait $SAMP $WD 2>/dev/null
  log "END pretrain attempt $att rc=$rc (peak GPU $(sort -n "$OUT/gpu_mem.log" | tail -n 1) MiB; peak RAM used $(sort -n "$OUT/ram_used_MiB.log" | tail -n 1) MiB): $(grep -E '^step' "$OUT/stdout.log" | tail -n 1 | cut -c1-170)"
  if tail -n 40 "$OUT/stdout.log" | grep -q 'CUDA out of memory'; then log "FAILED: CUDA out of memory, stopping"; break; fi
done
if grep -q 'PRETRAIN DONE' "$OUT/train.log" 2>/dev/null; then log "DONE $OUT/backbone_teacher.pth"; else log "FAILED: pretraining incomplete: $(tail -n 3 "$OUT/stdout.log" | tr '\n' '|' | cut -c1-300)"; fi
log "QUEUE31 ALL DONE"
