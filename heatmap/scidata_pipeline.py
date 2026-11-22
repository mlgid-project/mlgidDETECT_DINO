"""mlgidBASE detect -> fit -> match on the converted frames of the Das et al. 2026 GIWAXS benchmark (convert_scidata.py output), headless, the calls the mlgidGUI makes.

  ~/miniconda3/envs/mlgidgui/bin/python heatmap/scidata_pipeline.py --h5 <run1.h5> --stage detect fit match [--cifs <folder>] [--limit N] [--probability 0.5]

detect : mlgidBASE.run_detection (mlgidDETECT, the original detector)   -> data/analysis/frame00000/detected_peaks
fit    : mlgidBASE.run_fitting (pygidFIT)                                  -> fitted_peaks
match  : mlgidBASE.run_matching against the CIF folder (mlgidMATCH), segments and rings separately; the CifPattern is built once per file (the wavelength is the same for every entry of a run) -> matched_*
Every stage writes into the h5 file itself; run it on a COPY if the converted file must stay untouched. --limit works on the first N entries (copied into <h5>.limit<N>.h5)."""
import argparse, os, shutil, sys, time
from pathlib import Path
import h5py

p = argparse.ArgumentParser()
p.add_argument('--h5', required=True); p.add_argument('--stage', nargs='+', default=['detect', 'fit', 'match']); p.add_argument('--cifs', default=str(Path.home() / 'Documents/datasets/cif_giwaxs_scidata/match_set'))
p.add_argument('--limit', type=int, default=0); p.add_argument('--probability', type=float, default=0.5)
a = p.parse_args()
src = Path(a.h5)
if a.limit:
    dst = src.with_suffix(f'.limit{a.limit}.h5')
    with h5py.File(src, 'r') as fi, h5py.File(dst, 'w') as fo:
        for n in list(fi.keys())[:a.limit]:
            fi.copy(fi[n], fo, name=n)
    src = dst
with h5py.File(src, 'r') as f:
    entries = list(f.keys())
print(f'{src}: {len(entries)} entries, stages {a.stage}', flush=True)
from mlgidbase import mlgidBASE
analysis = mlgidBASE(filename=str(src))
t0 = time.time()
if 'detect' in a.stage:
    analysis.run_detection(); print(f'detection done {time.time() - t0:.0f} s', flush=True)
if 'fit' in a.stage:
    t0 = time.time()
    for k, n in enumerate(entries):
        try:
            analysis.run_fitting(entry=n)
        except Exception as e:
            print('fit failed', n, repr(e)[:150], flush=True)
        if (k + 1) % 100 == 0:
            print(f'fitting {k + 1}/{len(entries)} {time.time() - t0:.0f} s', flush=True)
    print(f'fitting done {time.time() - t0:.0f} s', flush=True)
if 'match' in a.stage:
    from mlgidgui.pipeline import parse_cif_input
    t0 = time.time()
    pattern = parse_cif_input(a.cifs, src, entries[0]); print(f'CIF pattern built from {a.cifs} {time.time() - t0:.0f} s', flush=True)
    t0 = time.time(); nfail = 0
    for k, n in enumerate(entries):
        for pt in ('segments', 'rings'):
            try:
                analysis.run_matching(entry=n, cif_prepr=pattern, peaks_type=pt, probability_threshold=a.probability)
            except Exception as e:
                nfail += 1
                if nfail <= 5:
                    print('match failed', n, pt, repr(e)[:200], flush=True)
        if (k + 1) % 100 == 0:
            print(f'matching {k + 1}/{len(entries)} {time.time() - t0:.0f} s', flush=True)
    print(f'matching done {time.time() - t0:.0f} s, {nfail} failures', flush=True)
