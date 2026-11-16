#!/bin/bash
# Generate a realsim_v2 dataset on colorbox1 (data lives in the colorbox dir, code in the repo).
#   realsim_v2/gen.sh NAME N [WORKERS]      -> /mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/NAME  (config.json, shards/, failures/, gen.log)
# Resumable: finished shards are skipped; the config in <out>/config.json must match the generator settings (gen_v2.py: DEFAULT_CFG + iomin_sr 0.5).
# Started only on the user's go. Runs at nice 19; a watchdog stops the generator if MemAvailable < 6 GiB (GUI_project shares the RAM).
NAME=$1; N=$2; W=${3:-6}
[ -z "$NAME" ] || [ -z "$N" ] && { echo "usage: $0 NAME N [WORKERS]"; exit 1; }
REPO=$HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP
OUT=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/$NAME
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
mkdir -p "$OUT"
cd "$REPO/realsim_v2" || exit 1
nice -n 19 "$PY" -u gen_v2.py --out "$OUT" --n "$N" --workers "$W" >> "$OUT/gen.log" 2>&1 & GP=$!
( while kill -0 $GP 2>/dev/null; do av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$av" -lt 6144 ]; then echo "$(date '+%F %T') WATCHDOG: MemAvailable ${av} MiB < 6 GiB -> stopping the generator" >> "$OUT/gen.log"; pkill -TERM -P $GP; kill $GP; fi; sleep 10; done ) &
wait $GP; echo "$(date '+%F %T') generator exit $?" >> "$OUT/gen.log"
