#!/bin/bash
#SBATCH --job-name=dino_structnoise1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn203
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_structnoise1-%j.out
#
# Step 2 (sim2real): structural noise injection. Fine-tunes the ssl1 checkpoint with real-level
# white pixel-grain injected into the synthetic training backgrounds (datasets/struct_noise.py;
# config DINO_4scale_swin_structnoise.py). Training-only transform, inference/ONNX byte-identical.
# RESUMABLE (auto-resume from checkpoint.pth). Per-epoch eval logs ap_total for 41 + organic.
# GATE = faint(vis=1)/high-q recall probe (diag_compare.py) vs ssl1, NOT AP alone.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_structnoise1
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
export PYTHONPATH=$REPO

if [ ! -f "$PRETRAIN" ]; then
    echo "ERROR: warm-start checkpoint not found at $PRETRAIN"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (ssl1 warm start)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_structnoise.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path "$PRETRAIN"
