#!/bin/bash
#SBATCH --job-name=dino_physics1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn203
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_physics1-%j.out
#
# Track B: physics-based CIF simulation dilution (50%). Fine-tunes the ssl1 checkpoint with half
# the training images built from CIF-derived peak configurations (config
# DINO_4scale_swin_physics.py; bank from physics_sim/generate_bank.py, eval structures excluded
# via mlgidMATCH). Training-only; inference/ONNX byte-identical. RESUMABLE (auto-resume from
# checkpoint.pth). GATE = organic/41 AP + faint/high-q recall probe vs ssl1.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics1
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library/bank/bank.npz
export PYTHONPATH=$REPO

if [ ! -f "$PRETRAIN" ]; then echo "ERROR: warm-start checkpoint not found at $PRETRAIN"; exit 1; fi
if [ ! -f "$BANK" ]; then echo "ERROR: physics bank not found at $BANK"; exit 1; fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (ssl1 warm start)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path "$PRETRAIN"
