#!/bin/bash
# colorbox1 queue 41 (user go 2026-11-16): latent map of the corpus. Waits for the shared GPU lock (after queue 40, frozen pretrained ViT-H+ seed 2).
# 1. embed_corpus.py: our pretrained ViT-H+ (vith16plus_3k_sk2), 5 frames per scan + all organic / 41 frames -> latent_map/embed_vith_sk2.npz
# 2. latent_map.py (CPU): regions for k = 6, 8, 10, coverage, maps and example sheets -> latent_map/vith_sk2/
# Status: hm_runs/_tools/queue41_status.txt (last line QUEUE41 ALL DONE).
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
OUT=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/latent_map
STATUS=$RUNS/_tools/queue41_status.txt
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
cd "$REPO" || exit 1
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
exec 9>"$RUNS/_tools/queue41.lock"
flock -n 9 || { log "another queue41 instance holds the lock -> exit"; exit 0; }
log "QUEUE41 START (pid $$)"
exec 8>"$RUNS/_tools/gpu.lock"
log "waiting for the GPU lock"; flock 8; log "GPU lock acquired"
while pgrep -f '^[^ ]*python[^ ]* -u heatmap/train.py' >/dev/null; do sleep 60; done
if [ ! -f "$OUT/embed_vith_sk2.npz" ]; then
  log "START embed"
  "$PY" -u heatmap/embed_corpus.py --out "$OUT/embed_vith_sk2.npz" > "$OUT/embed_vith_sk2.log" 2>&1
  log "END embed rc=$? ($(tail -n 1 "$OUT/embed_vith_sk2.log"))"
fi
flock -u 8          # the analysis is CPU only
log "START latent_map"
"$PY" -u heatmap/latent_map.py --emb "$OUT/embed_vith_sk2.npz" --out "$OUT/vith_sk2" > "$OUT/vith_sk2.log" 2>&1
log "END latent_map rc=$?"
log "QUEUE41 ALL DONE"
