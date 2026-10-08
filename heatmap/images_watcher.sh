#!/bin/bash
# Companion to overnight.sh (which is already running without the images step): waits for each overnight run's
# final evaluation to be complete, then renders its images. Run detached: nohup bash heatmap/images_watcher.sh > .../images_watcher.log 2>&1 &
REPO=${REPO:-/home/nicolerch/Documents/DINO/mlgidDETECT_DINO_HEATMAP}
PY=${PY:-/home/nicolerch/miniconda3/envs/heatmap-smoke/bin/python}
RUNS=${RUNS:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs}
SIMMIM=${SIMMIM:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth}
BOXCONV=${BOXCONV:-/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/boxconv1_backbone.pth}
export HM_DATA_DIR=${HM_DATA_DIR:-$HOME/Documents/datasets}
STATUS=$RUNS/_tools/overnight_status.txt
cd "$REPO" || exit 1
for spec in "hm_ridge_lr1e-4_tf32_2.80_1.30:$SIMMIM" "hm_boxconv1_frozen_ridge_tf32_2.80_1.30:$BOXCONV" "hm_simmim_frozen_ridge_long_tf32_2.80_1.30:$SIMMIM"; do
  name=${spec%%:*}; bb=${spec#*:}
  for i in $(seq 1 1000); do   # up to ~16 h per run, polling every minute
    grep -q '+nms\] 41: ap_total' "$RUNS/$name/evaluate_final.txt" 2>/dev/null && [ -f "$RUNS/$name/final_checkpoint.pth" ] && break
    sleep 60
  done
  if [ -f "$RUNS/$name/final_checkpoint.pth" ]; then
    HM_BB_PATH="$bb" "$PY" -u heatmap/visualize.py --ckpt "$RUNS/$name/final_checkpoint.pth" --out "$RUNS/$name/images" \
        --sets organic 41 --thr 0.3 > "$RUNS/$name/visualize.log" 2>&1
    echo "$(date '+%F %T') IMAGES $name rc=$?" | tee -a "$STATUS"
  else
    echo "$(date '+%F %T') IMAGES $name skipped (no final checkpoint)" | tee -a "$STATUS"
  fi
done
