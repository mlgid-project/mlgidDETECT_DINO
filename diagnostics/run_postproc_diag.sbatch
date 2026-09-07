#!/bin/bash
#SBATCH --job-name=postproc_diag
#SBATCH --partition=2080-galvani
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO/diagnostics/postproc_diag-%j.out
#
# Is the rings+segments gap vs faster_rcnn a POSTPROCESSING problem or a training problem?
# Answers that before we spend another 3-day run on it. See diagnostics/postproc_diag.py.
#
# CPU-only (onnxruntime CPUExecutionProvider), so no --gres: it asks for no GPU and leaves both
# a100 nodes to dino_mcc / dino_edge. Env is mlgid-ens, the only one here with onnxruntime.

set -e
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/mlgid-ens/bin/python
ONNX=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/onnx
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=16
cd "$REPO"
srun $PY -u diagnostics/postproc_diag.py "$ONNX/dino_ssl1.onnx" "$ONNX/dino_lr4e5.onnx"
