#!/bin/bash
#SBATCH --job-name=dino_phys2
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00              # 72h = partition max
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/backbone_curation/ssl/dino_phys2-%j.out
#
# Physics-CIF peak configuration, second attempt. Config: DINO_4scale_swin_physics2.py.
# Full rationale, the four differences from the declined phase P, and the eval-cleanliness
# caveat are in MODIFICATIONS.md section I and in the config header -- read those first.
#
# vs the ssl1 recipe this changes exactly two things, both deliberately bundled:
#   use_physics_sim / physics_sim_fraction 0.25  -- a quarter of each epoch's images take their
#       peak positions and STRUCTURE-FACTOR INTENSITIES from 60,474 COD organic CIFs via
#       pygidsim, instead of the uniform (2,50)/(10,50) draws in gen_intensities. The
#       intensities are the point of the track; the standard sim has no intensity physics at all.
#   unify_contrast                               -- physics images run the REAL contrast order
#       (percentile clip -> log10(|x|+1e-7) -> HE, mask-aware, simulation.contrast_like_real)
#       rather than the sim's log -> HE -> clip with a synthetic decade range. Only meaningful
#       BECAUSE physics intensities have a real dynamic range.
# Everything else is inherited: 1-channel, SSL backbone init, 2.80/1.30 boxes, lr = lr_backbone
# = 4e-5, batch 2, 500 ep, lr_drop 280.
#
# GATES.
#   Wiring check, epoch 0: the startup banner must print "[sim] physics sim ON: 25% of images
#   from .../bank_organic.npz (N powder / M oriented entries), unify_contrast=True". If the bank
#   path is wrong main.py raises instead of silently training on 100% standard images.
#   Sanity, epoch ~5: loss_giou should track a normal run (batch8_1 was ~1.05 -> 0.74 over epochs
#   1-3). PhysicsSimulation raises loudly after 20 empty draws rather than hanging, but note
#   SimulationDataset.__getitem__ swallows exceptions in a bare except and retries forever, so a
#   stalled epoch 0 with no error means the physics generator is failing every draw -- check the
#   bank before assuming a GPU problem.
#   VERDICT: organic/41 AP judged ONLY on the post-280 plateau (MODIFICATIONS.md; pre-drop
#   ranking correlates only rho=0.49 with the settled one) against dino_batch8_1's 0.6081/0.7613.
#   Phase P's decisive probe applies here too: a fixed-score recall comparison is NOT admissible,
#   because dilution made the detector fire 5.5x more boxes and look better at score 0.3 while
#   losing at every matched precision. Compare at matched precision (~0.83) and matched FP/img.
#
# RESUMABLE: main.py auto-picks-up checkpoint.pth from --output_dir, so resubmit the same command
# across 72h sessions. Per-epoch eval logs ap_total for 41 + organic (eval_interval=2).

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
OUT_DIR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/detector_runs/dino_physics2_1
BB=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/ssl_runs/simmim1/backbone_export/swin_large_patch4_window12_384_22k.pth
BANK=/mnt/lustre/work/schreiber/szb389/datasets/cif_library_organic/bank/bank_organic.npz
export PYTHONPATH=$REPO

if [ ! -f "$BB" ]; then
    echo "ERROR: SSL backbone weights not found at $BB"; exit 1
fi
if [ ! -f "$BANK" ]; then
    echo "ERROR: physics peak bank not found at $BANK"
    echo "       build it first: sbatch physics_sim/run_bank_organic.sbatch"; exit 1
fi
mkdir -p "$OUT_DIR"
[ -f "$OUT_DIR/checkpoint.pth" ] && echo "[run] resuming from $OUT_DIR/checkpoint.pth" || echo "[run] fresh start (SSL backbone init, 1-channel, 25% physics-CIF images, unified contrast, batch 2, lr = lr_backbone = 4e-5)"

cd "$REPO"
srun $PY -u main.py \
    -c config/DINO/DINO_4scale_swin_physics2.py \
    --output_dir "$OUT_DIR"
