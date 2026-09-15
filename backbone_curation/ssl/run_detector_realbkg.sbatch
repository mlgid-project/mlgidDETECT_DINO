#!/bin/bash
#SBATCH --job-name=dino_realbkg
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_realbkg-%j.out
#
# pygidSIM peaks on REAL, feature-stripped GIWAXS backgrounds.
# Config: DINO_4scale_swin_realbkg.py -- read its header. Built in giwaxs_sim.ipynb.
#
# WHAT CHANGES: the image source, wholesale. The background is a real frame from
# ekaterina_aftermlgidFIT with its own diffraction features removed; only the peaks are simulated,
# and only their positions and intensities come from pygidSIM. Widths, shapes, brightness and
# counts are all taken from mlgidFIT's 2-D Gaussian fits of 1,926 real peaks.
#
# NOT A ONE-VARIABLE TEST. The image source changes completely, so this is not comparable to the
# dino_physics* arms as a single lever. Judge it on the absolute AP curve.
#
# WIRING CHECK, epoch 0: the log must show
#     [sim] real-background sim ON -- image source is .../sim_background_bank4.h5
#     [realbkg] 189 donors of 444 from 27 source files
#     [sim] box convention overridden: a_coef=2.8 w_coef=1.3
# If the [realbkg] line is absent the config key did not reach args -- kill it, the run is a
# duplicate of the SSL baseline.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20.
#
# KNOWN RISK, measured and open: selected donors score leftover-z ~15 against a detection
# threshold of 3, so a few frames carry a faint unremoved REAL feature with no box on it. The
# detector is penalised for finding it. Treat a weak result as possibly this rather than the idea.
#
# JUDGE POST-280 ONLY -- pre-drop ranking correlates with the plateau at rho = 0.49.
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz
DONORS=/mnt/lustre/work/schreiber/szb389/datasets/sim_background_bank4.h5
STATS=/mnt/lustre/work/schreiber/szb389/datasets/sim_real_stats.npz
export PYTHONPATH=$REPO

for f in "$BB" "$BANK" "$DONORS" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" \
  || echo "[run] fresh start (SSL backbone, 1-channel, real-background sim, batch 2, lr 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_realbkg.py \
    --output_dir "$OUT_DIR"
