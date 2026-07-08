#!/bin/bash
#SBATCH --job-name=dino_codino
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_codino-%j.out
#
# Co-DINO Phase-0 MVP (docs/CO_DINO_INVESTIGATION.md): fine-tune the ssl1 checkpoint with
# ONE training-only collaborative aux head (FCOS center-sampling one-to-many supervision
# on the encoder pyramid). The aux head is discarded at inference. Standard pipeline
# otherwise. RESUMABLE (auto-resume from checkpoint.pth). Per-epoch eval logs 41 + organic.
# READ-OUT: gate on organic AP AND the faint/high-q recall probe (diag_compare.py).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_codino1
PRETRAIN=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_ssl1/checkpoint.pth
export PYTHONPATH=$REPO

if [ ! -f "$PRETRAIN" ]; then
    echo "ERROR: warm-start checkpoint not found at $PRETRAIN"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (ssl1 warm start)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_codino.py \
    --output_dir "$OUT_DIR" \
    --amp \
    --pretrain_model_path "$PRETRAIN"
