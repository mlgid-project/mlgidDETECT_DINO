#!/bin/bash
# live progress bar of a shard generation:  watch -n 10 realsim_v2/progress.sh [NAME [TOTAL_SHARDS]]    (default v3_lowq 280)
NAME=${1:-v3_lowq}; TOT=${2:-280}
D=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/$NAME/shards
n=$(ls "$D"/shard_*.zip 2>/dev/null | wc -l)
partial=$(du -cb "$D"/*.tmp 2>/dev/null | awk 'END{print $1+0}')          # bytes of the shards still being written
awk -v n="$n" -v tot="$TOT" -v part="$partial" -v t="$(date +%T)" 'BEGIN{
  est = n + part/315e6; if (est > tot) est = tot                          # ~315 MB per finished shard of 250 images
  w = 40; f = int(est*w/tot + 0.5); bar = ""; for (i = 0; i < w; i++) bar = bar (i < f ? "#" : "-")
  printf "%s\n[%s]\nfinished %d/%d shards (%.1f%%), %d images | estimate incl. partial shards %.1f%%\n", t, bar, n, tot, n*100/tot, n*250, est*100/tot }'
