#!/bin/bash
#SBATCH --job-name=dino_hires
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn203
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_hires-%j.out
#
# Higher-resolution FROM-SCRATCH run (512x2048). RESUMABLE, and driven past every 72h wall by a
# pre-submitted afterany dependency CHAIN (each link `--dependency=afterany` the previous). Every
# link auto-resumes from OUT_DIR/checkpoint.pth (written each epoch => <=1 epoch lost per hop) and
# is a NO-OP once organic eval reaches the final epoch (completion guard below) -- so any surplus
# links in the chain exit in ~1s. To stop early: scancel the pending links.

REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_hires1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
APLOG="$OUT_DIR/exp_ap_organic.txt"
TARGET_EPOCH=498                       # last eval epoch (epochs=500, eval_interval=2)
export PYTHONPATH=$REPO

# --- completion guard: a chain link that starts after training already finished is a no-op ---
if [ -f "$APLOG" ]; then
    LAST=$(awk '{print $1+0}' "$APLOG" 2>/dev/null | sort -n | tail -1); LAST=${LAST:--1}
    if [ "$LAST" -ge "$TARGET_EPOCH" ]; then
        echo "[chain] training already at epoch $LAST >= $TARGET_EPOCH — complete; nothing to do."; exit 0
    fi
    echo "[chain] last logged organic-eval epoch = $LAST (< $TARGET_EPOCH) — resuming."
fi

if [ ! -f "$BB" ]; then echo "ERROR: SSL backbone weights not found at $BB"; exit 1; fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 512x2048 from scratch)"

cd "$REPO"
srun $PY -u main.py -c config/DINO/DINO_4scale_swin_hires.py --output_dir "$OUT_DIR"
