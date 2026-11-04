#!/bin/bash
#SBATCH --job-name=rs_v3_test
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=heatmap/sbatch/rs_v3_test-%j.out
# One bundled test of realsim_v3 under pygidsim 0.1.7 (env mlgid_physics017): 8 images, tables only, compare with colorbox reference.
W=/mnt/lustre/work/schreiber/szb389
PY=/home/schreiber/szb389/.conda/envs/mlgid_physics017/bin/python
OUT=$W/realsim_v3_test_out
rm -rf "$OUT"
cd $W/realsim_v3 || exit 1
export GIWAXS_GNOME=$W/datasets/gnome OMP_NUM_THREADS=1
$PY -c "import pygidsim, numpy, scipy; from pygidsim import GIWAXSFromCif; print('pygidsim', pygidsim.__version__ if hasattr(pygidsim,'__version__') else '?', 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
t0=$(date +%s)
$PY -u generate_dataset_v3.py --out "$OUT" --n 8 --workers 2 --shard-size 4 --tables_only; echo "generate rc=$? wall $(( $(date +%s)-t0 )) s"
$PY -I - "$OUT" <<'PYEOF'
import sys, zipfile, hashlib, io, json, glob
import numpy as np
out = sys.argv[1]
ref = {0:("601db09192",15,"689773658b"),1:("8c622fb196",249,"b2acee8fa1"),2:("ff77757ff2",238,"c845401214"),3:("e7e11f9b36",183,"cf4295e66e"),
       4:("822bb07bb9",264,"642ca23346"),5:("e1aa631c13",206,"9b276ab94f"),6:("e853c5dd33",297,"0c9d55da96"),7:("2defb86220",154,"53f87eaa7a")}
keys = ["kind","q_c","chi_c","sigma_q","sigma_chi","intensity"]
for z in sorted(glob.glob(out + "/shards/shard_*.zip")):
    zf = zipfile.ZipFile(z)
    for name in sorted(zf.namelist()):
        d = np.load(io.BytesIO(zf.read(name)), allow_pickle=True)
        i = int(name.split(".")[0])
        info = json.loads(str(d["info"]))
        mid = info["MaterialId"]
        md5 = hashlib.md5(b"".join(d[k].tobytes() for k in keys)).hexdigest()[:10]
        n = len(d["kind"])
        r = ref[i]
        print(f"{i} mid {mid} (ref {r[0]}) {'MID_OK' if mid==r[0] else 'MID_DIFF'} n={n} (ref {r[1]}) {'N_OK' if n==r[1] else 'N_DIFF'} md5 {md5} (ref {r[2]}) {'BIT_IDENTICAL' if md5==r[2] else 'md5_differs'}")
PYEOF
