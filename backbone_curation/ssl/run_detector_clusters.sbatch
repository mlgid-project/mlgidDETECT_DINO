#!/bin/bash
#SBATCH --job-name=dino_clusters
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=galvani-cn203
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_clusters-%j.out
#
# Phase U — azimuthal peak clusters, FROM-SCRATCH (SSL backbone + random head, ssl1 recipe).
# Single variable vs ssl1 = synthetic label geometry (several peaks sharing one q at different chi,
# with the real chi-gap distribution). Grid stays 512x1024 and postprocessing stays at the deployed
# settings, so this is a drop-in for the deployed inference path.
#
# RESUMABLE, and driven past every 72h wall by a pre-submitted afterany dependency CHAIN. Each link
# auto-resumes from OUT_DIR/checkpoint.pth (written every epoch => <=1 epoch lost per hop) and is a
# NO-OP once organic eval reaches the final epoch (completion guard below), so surplus links exit in
# ~1s. To stop early: scancel the pending links.
#
# GATE (pre-registered, MODIFICATIONS.md U): PRIMARY = same-q sibling recall by chi-gap bucket at a
# MATCHED operating point (diagnostics/nms_sweep_single.py reports exactly this table); SECONDARY =
# organic/41 ap_total must not regress. Control = ssl1 measured under identical settings
# (job 2755004): organic ap_total 0.5683 / 41 0.7441; tight-pair (<5 px) recall 0.352 / 0.449.

REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_clusters1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
APLOG="$OUT_DIR/exp_ap_organic.txt"
TARGET_EPOCH=498                       # epochs=500, eval_interval=2 => last eval epoch
export PYTHONPATH=$REPO

# --- completion guard: a chain link starting after training finished is a no-op ---
if [ -f "$APLOG" ]; then
    LAST=$(awk '{print $1+0}' "$APLOG" 2>/dev/null | sort -n | tail -1); LAST=${LAST:--1}
    if [ "$LAST" -ge "$TARGET_EPOCH" ]; then
        echo "[chain] training already at epoch $LAST >= $TARGET_EPOCH — complete; nothing to do."; exit 0
    fi
    echo "[chain] last logged organic-eval epoch = $LAST (< $TARGET_EPOCH) — resuming."
fi

if [ ! -f "$BB" ]; then echo "ERROR: SSL backbone weights not found at $BB"; exit 1; fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
                                 || echo "[run] fresh start (SSL backbone init, clusters ON, 512x1024)"

cd "$REPO"
srun $PY -u main.py -c config/DINO/DINO_4scale_swin_clusters.py --output_dir "$OUT_DIR"
