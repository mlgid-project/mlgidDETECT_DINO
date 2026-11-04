#!/bin/bash
#SBATCH --job-name=rs_v3_tables
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=heatmap/sbatch/rs_v3_tables-%j.out
# realsim_v3 label tables only (no images), all GNoME structures (one per image). Resumable: finished shards are skipped,
# so after a timeout resubmit the same script. Do not run two jobs on the same OUT (startup deletes *.tmp shards).
W=/mnt/lustre/work/schreiber/szb389
PY=/home/schreiber/szb389/.conda/envs/mlgid_physics017/bin/python
OUT=$W/datasets/realsim_v3_tables
N=${N:-426359}
cd $W/realsim_v3 || exit 1
export GIWAXS_GNOME=$W/datasets/gnome OMP_NUM_THREADS=1
$PY -u generate_dataset_v3.py --out "$OUT" --n $N --workers 20 --tables_only
echo "generate rc=$?"
