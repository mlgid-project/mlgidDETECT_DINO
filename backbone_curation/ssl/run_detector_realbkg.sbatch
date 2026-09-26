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
#     [sim] real-background sim ON -- image source is fresh mosaics of reviewed peak-free frames
#     [realbkg] mosaic mode: 48 fresh backgrounds from 90 clean donor frames
#     [sim] labels: render-iff-labelled | contrast>=2.0 snr>=6.0 | IoU seg 0.3 ring 0.1 | max_peaks 200
#     [sim] counts: spots/entry (2, 200) rings/entry (1, 15) | oriented/frame (1, 3) | p_ring 0.3 ...
#     [sim] box convention overridden: a_coef=2.8 w_coef=1.3
# If the [realbkg] line is absent the config key did not reach args -- kill it, the run is a
# duplicate of the SSL baseline. If the labels line says THREE-TIER, or counts says (8, 60), the
# convention did not reach the simulator and the run is on the OLD labelling -- kill it too.
# diagnostics/realbkg_wiring_smoke.py asserts all twelve values on CPU before you submit.
#
# EARLY KILL GATE: loss_giou > 1.2 at epoch 40, or organic ap_total < 0.10 at epoch 20.
#
# THAT RISK IS GONE. This used to warn that the selected donors scored leftover-z ~15 against a
# detection threshold of 3, so some frames carried a faint unremoved REAL feature with no box on
# it. That was the OLD 189-donor bank. Backgrounds are now mosaics of the 90 reviewed bare-Si
# frames, which never had peaks to remove -- see donors_final.json and the donor-final-set note.
#
# DYNAMIC RANGE, accepted by the user 2026-09-26 rather than fixed. Measured over 120 frames,
# max/median runs 3.9 / p50 161 / max 128,520 against real's 96 / p50 3,917 / max 13,116, so 39%
# of frames are flatter than any real frame. The user's call: real images with e7 peaks sit on
# backgrounds of a few hundred counts, the raw frames show exactly that, and the ABSOLUTE levels
# are what matter -- max/median is the wrong thing to tune. Do not re-raise it as a blocker.
#
# KNOWN MISMATCHES, open and not blocking: segments/frame 94.6 against organic 63.6 and 41 16.5,
# so frames are denser than either gate; and corr(rings, segments) is -0.10 against real's -0.48
# (organic) / -0.25 (41), i.e. the real anti-correlation is essentially absent -- the sim makes
# frames with many segments AND many rings, which neither eval set contains.
#
# JUDGE POST-280 ONLY -- pre-drop ranking correlates with the plateau at rho = 0.49.
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_realbkg_conv1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic_hkl.npz
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
