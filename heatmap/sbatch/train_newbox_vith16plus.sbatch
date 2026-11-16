#!/bin/bash
#SBATCH --job-name=hm_vith_unfrozen
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP_vit/heatmap/sbatch/%x-%j.out
# DINOv3 ViT-H+/16 (840M, public LVD-1689M weights) UNFROZEN on the heatmap detector, new-box files, box_coef 3.0,3.0, legacy sim.
# Recipe = the colorbox1 ConvNeXt-L unfrozen runs (queues 25/35/36), so the rows are comparable: ridge target, TF32, bs 4 x 250 steps (samples per epoch = 1000), head lr 3e-4,
# backbone lr 1e-5, 120 epochs, lr x0.1 at 90 and 112, eval every 2 epochs, grad checkpointing, HM_GT_BOX_COEF=auto (scaled GT is the only final scoring).
# Env switches (sbatch --export=ALL,...): SEED (default 42), LR_BB (default 1e-5), BS (default 4; steps per epoch becomes 1000/BS so the sample count stays equal), SMOKE=1 (20 steps, no eval, prints speed and peak GPU memory; use with --time=00:30:00).
# Resubmitting the same job resumes from <OUT>/checkpoint.pth. Memory with ViT-H+ on 40 GB is UNTESTED: run SMOKE=1 first.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP_vit
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
BASE=/mnt/lustre/work/schreiber/szb389/datasets/DINO_HEATMAP
W=${W:-/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth}   # other backbone weights: --export=ALL,W=<pth>,TAG=<name part>
TAG=${TAG:-}                                                    # e.g. TAG=_pretrain3k_sk2 -> hm_dinov3vith16plus_pretrain3k_sk2_UNFROZEN_...
SEED=${SEED:-42}; LR_BB=${LR_BB:-1e-5}; BS=${BS:-4}; SPE=$((1000 / BS))
OUT=$BASE/hm_dinov3vith16plus${TAG}_UNFROZEN_ridge_long_tf32_seed${SEED}_newbox_3.0_3.0_bblr${LR_BB}_galvani
[ "$BS" != 4 ] && OUT=${OUT%_galvani}_bs${BS}_galvani
[ "${SMOKE:-0}" = 1 ] && OUT=$BASE/_smoke_vith16plus_bs${BS}_${SLURM_JOB_ID:-x}
export PYTHONPATH=$REPO HM_DATA_DIR=/mnt/lustre/work/schreiber/szb389/datasets/newbox_visfix_links HM_GT_BOX_COEF=auto
export HM_DINOV3_REPO=/mnt/lustre/work/schreiber/szb389/dinov3/dinov3_repo
unset HM_SCORE_FLOOR HM_AMP HM_BB_PATH GIWAXS_BANK
cd $REPO
mkdir -p "$OUT"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  out: $OUT"; nvidia-smi --query-gpu=name,memory.total --format=csv
# 1. does the wrapper build and do the weights match strictly (no GPU needed)?
$PY - <<EOF
import importlib.util, torch
print('torch', torch.__version__)
spec = importlib.util.spec_from_file_location('dbb', 'models/dinov3_backbone.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bb = m.DinoV3Backbone('dinov3_vith16plus')
print('weights strict load:', bb.load_state_dict(torch.load('$W', map_location='cpu')))
print('params M', sum(p.numel() for p in bb.parameters()) / 1e6)
EOF
[ $? = 0 ] || { echo "backbone check FAILED -> abort"; exit 1; }
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$OUT/gpu_mem.log"; sleep 5; done ) &
MON=$!
COMMON="--bb dinov3_vith16plus --bb_path $W --ring_target ridge --tf32 --bs $BS --steps_per_epoch $SPE --lr 3e-4 --epochs 120 --lr_drops 90 112 --eval_interval 2 --box_coef 3.0,3.0 --sim legacy --seed $SEED --unfreeze --grad_ckpt --lr_backbone $LR_BB"
if [ "${SMOKE:-0}" = 1 ]; then
  $PY -u heatmap/train.py --out "$OUT" $COMMON --max_steps 20 > "$OUT/train.log" 2>&1; echo "smoke rc=$?"
  tail -15 "$OUT/train.log"
  echo "peak GPU memory used (MiB, nvidia-smi samples): $(awk -F, '{gsub(/ MiB| /,"",$2); print $2}' "$OUT/gpu_mem.log" | sort -n | tail -1)"
  kill $MON; exit 0
fi
$PY -u heatmap/train.py --out "$OUT" $COMMON >> "$OUT/train.log" 2>&1
RC=$?; echo "train rc=$RC"
if [ $RC = 0 ] && [ -f "$OUT/checkpoint.pth" ]; then
  cp "$OUT/checkpoint.pth" "$OUT/final_checkpoint.pth"
  $PY -u heatmap/evaluate.py hm=heatmap:"$OUT/final_checkpoint.pth" > "$OUT/evaluate_final_gtscaled.txt" 2>&1; echo "eval (scaled GT) rc=$?"
  grep -E '\[hm\+nms\] (organic|41): ap_total' "$OUT/evaluate_final_gtscaled.txt"
fi
kill $MON
