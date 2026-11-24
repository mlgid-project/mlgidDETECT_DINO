#!/bin/bash
#SBATCH --job-name=pre_vith
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn205,galvani-cn213,galvani-cn232
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP_vit/heatmap/sbatch/%x-%j.out
# Continued DINO-style pretraining of the public DINOv3 ViT-H+/16 (840M) on the 274k Moneta v2 corpus (both corpora, hold-out v2 excluded), one A100 40 GB.
# Recipe = colorbox1 queue 34 (convnextL_3k_sk2, the run without collapse): 3,000 optimizer steps x 66 frames (bs 6 x accum 11), heads alone for the first 300 steps,
# --sinkhorn 3 --t_warmup 0.04 0.07 1000 --koleo 0.1 --gram 10 --prototypes 1024 --proto_patch 1024 --patch_stats --patch_g1, collapse stop --min_div 0.3 from step 600 (exit 3; the first 30k run was stopped at step 6380 by --min_perp_frac 0.1, which only measured sharpening, see idea-flagship-dino-run).
# ViT taps: pooled image feature = last block (stride 32 resample), patch loss = stride-16 token grid after block 24 (the map the detector's stride-16 level uses).
# patch_stats.pt = stats of the PUBLIC ViT-H+ on corpus crops, made on colorbox1 (make_patch_stats.py --bb dinov3_vith16plus), copied to $PREP.
# Env switches (sbatch --export=ALL,...): SMOKE=1 (40 steps, 10 head steps, own output dir, prints speed and peak memory; use with --time=00:45:00), BS + ACCUM (default 6 x 11; keep BS*ACCUM = 66), STEPS (default 3000).
# Resubmitting the same job resumes from <OUT>/checkpoint.pth (saved every 250 steps). Result for the heatmap probes: <OUT>/backbone_teacher.pth.
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO_HEATMAP_vit
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
WORK=/mnt/lustre/work/schreiber/szb389
CORP=$WORK/datasets/corpus_v2
PREP=$WORK/pretrain_runs/vith16plus_prep
W=$WORK/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
BS=${BS:-6}; ACCUM=${ACCUM:-11}; STEPS=${STEPS:-3000}
OUT=${OUT:-$WORK/pretrain_runs/vith16plus_3k_sk2}      # OUT=... for another run (the 30k-step flagship pretraining uses vith16plus_30k_sk2)
[ "${SMOKE:-0}" = 1 ] && OUT=$WORK/pretrain_runs/_smoke_vith16plus_bs${BS}_${SLURM_JOB_ID:-x}
export PYTHONPATH=$REPO HM_DINOV3_REPO=$WORK/dinov3/dinov3_repo
cd $REPO; mkdir -p "$OUT"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  out: $OUT  bs $BS x accum $ACCUM  steps $STEPS"; nvidia-smi --query-gpu=name,memory.total --format=csv
# 1. inputs present?
for f in "$W" "$PREP/patch_stats.pt" "$CORP/pretrain_exclude_v2.tsv"; do [ -e "$f" ] || { echo "missing $f -> abort"; exit 1; }; done
[ -d "$CORP/scans" ] && [ -d "$CORP/scans_other" ] || { echo "corpus dirs missing -> abort"; exit 1; }
# 2. hold-out list with Galvani paths: every one of the 1046 excluded scans must exist here, else test/validation scans would leak into pretraining
EXCL=$PREP/pretrain_exclude_v2_galvani.tsv
sed "s#/mnt/DATA/mlgidDETECT_DINO_HEATMAP/corpus_v2/#$CORP/#" "$CORP/pretrain_exclude_v2.tsv" > "$EXCL"
NX=$(wc -l < "$EXCL"); NOK=$(cut -f2 "$EXCL" | while read f; do [ -e "$f" ] && echo y; done | wc -l)
echo "hold-out list: $NX entries, $NOK found on Galvani"
[ "$NX" = 1046 ] && [ "$NOK" = 1046 ] || { echo "HOLD-OUT CHECK FAILED -> abort"; exit 1; }
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> "$OUT/gpu_mem.log"; sleep 10; done ) &
MON=$!
ARGS="--out $OUT --corpus $CORP/scans $CORP/scans_other --exclude $EXCL --bb dinov3_vith16plus --weights $W --flip_chi --bs $BS --accum $ACCUM --workers 12 --grad_ckpt
  --sinkhorn 3 --t_warmup 0.04 0.07 1000 --koleo 0.1 --gram 10 --prototypes 1024 --proto_patch 1024 --patch_stats $PREP/patch_stats.pt --patch_g1"
if [ "${SMOKE:-0}" = 1 ]; then
  $PY -u heatmap/pretrain_dinov3.py $ARGS --steps 40 --head_steps 10 --log_every 5 --save_every 100000 > "$OUT/stdout.log" 2>&1; echo "smoke rc=$?"
  grep -E 'corpus:|exclude list|^step|Error|error' "$OUT/stdout.log" | tail -14
  echo "peak GPU memory (MiB, nvidia-smi): $(awk -F, '{gsub(/ MiB| /,"",$2); print $2}' "$OUT/gpu_mem.log" | sort -n | tail -1)  (card 40960)"
  kill $MON; exit 0
fi
$PY -u heatmap/pretrain_dinov3.py $ARGS --steps $STEPS --head_steps 300 --log_every 10 --save_every 250 --min_div 0.3 --check_from 600 >> "$OUT/stdout.log" 2>&1
RC=$?; echo "pretrain rc=$RC (0 done, 3 collapse stop)"; tail -3 "$OUT/stdout.log"
kill $MON
