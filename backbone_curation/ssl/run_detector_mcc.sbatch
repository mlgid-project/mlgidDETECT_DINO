#!/bin/bash
#SBATCH --job-name=dino_mcc
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_mcc-%j.out
#
# Multi-CONTRAST detector: 4-channel input [log+HE, log+CLAHE4@16x16, log+gamma0.7, mask], i.e.
# three contrasts of the SAME image chosen from the 74-setting contrast sweep (ch0 = deployed
# default, ch1 = best organic, ch2 = best 41). Replaces the derived-channel run dino_mc1,
# which lost 0.022 organic against ssl1.
# ch0 carries the default contrast so the zero-padded patch embed still makes step 0 exactly
# a 1-channel model (diagnostics/mc_smoke.py).
#
# Gate: organic/41 ap_total vs ssl1's 0.5683 / 0.7441 (reproduced on this branch).
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same
# command across 72h sessions. Per-epoch eval logs `ap_total` for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_mcc1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 4-channel multi-contrast)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_mcc.py \
    --output_dir "$OUT_DIR"
