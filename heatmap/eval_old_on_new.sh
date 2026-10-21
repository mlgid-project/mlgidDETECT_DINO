#!/bin/bash
# Inference only (no training): the OLD-convention (box_coef 2.80,1.30) frozen checkpoints evaluated on the NEW visibility-fixed files (newbox_visfix; clamp patch in util/pygidloader.py).
# Same protocol as the newbox finals: evaluate.py (+nms, floor 0.1, top-225) and score_sweep.py (top-225/900, floor 0.2 number for the 41 set). Outputs: hm_runs/newbox/old_on_new/<run name>/.
RUNS=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs
PY=$HOME/miniconda3/envs/heatmap-smoke/bin/python
SIMMIM=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/swin_large_patch4_window12_384_22k.pth
export HM_DATA_DIR=$HOME/Documents/datasets/validationsSets/newbox_visfix
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH
cd $HOME/Documents/DINO/mlgidDETECT_DINO_HEATMAP || exit 1
exec 8>"$RUNS/_tools/gpu.lock"; flock 8
for n in plain_he_tf32_2.80_1.30:frozen_simmim_plain/hm_simmim_frozen_ridge_he_tf32_2.80_1.30 plain_seed1:frozen_simmim_plain/hm_simmim_frozen_ridge_he_tf32_seed1_2.80_1.30 plain_seed2:frozen_simmim_plain/hm_simmim_frozen_ridge_he_tf32_seed2_2.80_1.30 \
         runC_2.80_1.30:run_C_recipe/hm_simmim_frozen_ridge_long_tf32_2.80_1.30 runC_seed1:run_C_recipe/hm_simmim_frozen_ridge_long_tf32_seed1_2.80_1.30 runC_seed2:run_C_recipe/hm_simmim_frozen_ridge_long_tf32_seed2_2.80_1.30; do
  tag=${n%%:*}; src=$RUNS/${n#*:}; out=$RUNS/newbox/old_on_new/$(basename $src); mkdir -p $out/score_sweep
  echo "$(date '+%F %T') $tag"
  nice -n 10 $PY -u heatmap/evaluate.py hm=heatmap:$src/final_checkpoint.pth > $out/evaluate_final.txt 2>&1; echo " eval rc=$?"
  HM_BB_PATH=$SIMMIM nice -n 10 $PY -u heatmap/score_sweep.py "$tag=$src/final_checkpoint.pth" --out $out/score_sweep --decode nms --topk 225 900 > $out/score_sweep.log 2>&1; echo " sweep rc=$?"
done
echo "$(date '+%F %T') OLD-ON-NEW DONE"
