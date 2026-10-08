#!/bin/bash
#SBATCH --job-name=dino_realbkg8
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn205        # allocates a GPU torch cannot init (jobs 2920327, 2920328)
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg8-%j.out
#
# RUN 8: run 7 with ONE change, realbkg_seg_on_ring_keep 0.12 -> 0.5 (config r8; evidence in its header
# and in memory realbkg7-run). Run 7's segments that sit on rings score 0.36 on 41 vs 0.58 isolated
# (run 6 / legacy: no such gap); the r7 sim had 10.7% of segments on rings vs 41's 23.3%.
#
# FRESH vs FINE-TUNE: with an empty OUT_DIR this starts from epoch 0. To continue from run 7 under the new
# sim instead, copy run 7's checkpoint.pth into OUT_DIR BEFORE submitting (main.py auto-resumes from it).
#
# WIRING CHECK at epoch 0 -- kill the run if wrong:
#     [sim] rings: segments kept on rings 0.5 | ring box w_coef 3.0 | ring sigma_q x0.7 |
#           ring min gap 8.0 px | ring chi texture sd 0.45 corr (1.0, 64.0) px
#     the rest of the [sim] counts lines as run 7; dn_number=100
# HOW TO JUDGE: 41 and organic vs run 7 at matched epochs 20-70, and rerun seg_context_probe.py:
# on-ring segment score on 41 should rise from 0.36 toward ~0.7 without ring boxes/ring (1.8-2.6) rising.
set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv8
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic_hkl.npz
DONORS=/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5
STATS=/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz
export PYTHONPATH=$REPO

for f in "$BB" "$BANK" "$DONORS" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
# Job 2920327 was allocated gres/gpu:a100=1 on galvani-cn205 and torch still raised
# "No CUDA GPUs are available" 19 s in. Fail here instead, with the node named.
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: GPU allocated but torch sees no CUDA device on $(hostname)"; exit 1; }

mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
  || echo "[run] fresh start (SSL backbone, 1-channel, real-background sim, batch 2, lr 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_realbkg_r8.py \
    --output_dir "$OUT_DIR"
