"""Structure check of the detect + fit results of the Das et al. 2026 GIWAXS benchmark against CIF reflections (scidata_pipeline.py output), threshold-free.

  ~/miniconda3/envs/mlgidgui/bin/python heatmap/scidata_eval.py --h5 <run1.h5> [--h5 <run3.h5> ...] --out <dir> [--tol 0.025] [--min_score 0.3]

mlgidMATCH itself found no solution for the perovskite frames (even at probability 0.01), so this is a plain position check: for each phase the CIF reflections (simulated by pygidsim at the run's wavelength, q 0.6 - 2.6 A^-1, relative intensity >= 5 %) are compared with the radii of the fitted peaks (score >= --min_score);
a reflection is 'hit' when a fitted peak lies within --tol A^-1 (any angle, ring or segment). Per frame and phase:
  hit_weighted = sum of the relative intensities of the hit reflections / sum of all   (0 - 1)
  excess       = hit_weighted - mean(hit_weighted at the reflection list shifted by +-0.05, +-0.08, +-0.12 A^-1)     (what the positions explain beyond chance; many peaks give chance hits)
Labels come from the curated class name: perovskite expected when it contains PVK, SnO2 expected when it contains SNO2 (case-insensitive). AUC (frame with the label vs without) of 'excess' per phase and run -> <out>/auc.md, per-frame table -> <out>/frames.csv.
Phases: perovskite = alpha-FAPbI3 cubic + MAPbI3 cubic + MAPbI3 tetragonal (max over the three), PbI2, delta-FAPbI3, SnO2 (max over the two CIFs)."""
import argparse, csv, json, os, sys
from pathlib import Path
import h5py, numpy as np

p = argparse.ArgumentParser()
p.add_argument('--h5', action='append', required=True); p.add_argument('--out', required=True); p.add_argument('--tol', type=float, default=0.025); p.add_argument('--min_score', type=float, default=0.3)
p.add_argument('--cifs', default=str(Path.home() / 'Documents/datasets/cif_giwaxs_scidata/match_set'))
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from mlgidgui.pipeline import parse_cif_input
PHASES = {'perovskite': ['alphaFAPbI3_cubic_COD7236563.cif', 'MAPbI3_cubic_COD4335634.cif', 'MAPbI3_tetra_COD4335638.cif'], 'PbI2': ['PbI2_2H_COD9009114.cif'], 'delta_FAPbI3': ['deltaFAPbI3_hex_COD4335640.cif'], 'SnO2': ['SnO2_39173.cif', 'SnO2_9163.cif']}
SHIFTS = np.array([-0.12, -0.08, -0.05, 0.05, 0.08, 0.12])


def reflections(pat, cif):
    i = pat.cifs.index(cif); q = np.asarray(pat.all_patterns_q1d[i], float); w = np.asarray(pat.all_patterns_int1d[i], float)
    w = w / w.max(); k = (q >= 0.6) & (q <= 2.6) & (w >= 0.05)
    return q[k], w[k]


def hit(q, w, radii, tol):
    if len(q) == 0:
        return np.nan
    h = np.array([np.any(np.abs(radii - x) <= tol) for x in q]); return float((w * h).sum() / w.sum())


rows = []
for h5 in a.h5:
    idx = json.load(open(str(h5).replace('.h5', '_index.json')))
    with h5py.File(h5, 'r') as f:
        entries = list(f.keys()); pat = parse_cif_input(a.cifs, Path(h5), entries[0])
        refs = {ph: [reflections(pat, c) for c in cs] for ph, cs in PHASES.items()}
        for n in entries:
            fr = f[n]['data/analysis']; fr = fr[list(fr.keys())[0]] if len(fr) else None
            if fr is None or 'fitted_peaks' not in fr:
                radii, nfit = np.array([]), 0
            else:
                d = fr['fitted_peaks'][()]; ok = d['score'] >= a.min_score; radii = np.asarray(d['radius'][ok], float); nfit = int(ok.sum())
            cls = idx[n]['cls']; row = dict(run=Path(h5).stem, entry=n, cls=cls, raw=idx[n]['raw'], ai=idx[n]['ai'], n_peaks=nfit, label_perovskite=int('pvk' in cls.lower()), label_SnO2=int('sno2' in cls.lower()))
            for ph, lst in refs.items():
                best = -9
                for q, w in lst:
                    v = hit(q, w, radii, a.tol)
                    nul = np.nanmean([hit(q + s, w, radii, a.tol) for s in SHIFTS]) if len(q) else np.nan
                    if np.isfinite(v) and v - nul > best:
                        best, bh, bn = v - nul, v, nul
                row[f'{ph}_hit'] = round(bh, 3) if best > -9 else ''; row[f'{ph}_excess'] = round(best, 3) if best > -9 else ''
            rows.append(row)
with open(f'{a.out}/frames.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def auc(pos, neg):
    pos, neg = np.array(pos, float), np.array(neg, float); pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return float('nan'), len(pos), len(neg)
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), len(pos), len(neg)


lines = ['# Structure check against CIF reflections', '', f'tolerance {a.tol} A^-1, fitted peaks with score >= {a.min_score}; {len(rows)} frames. AUC of the per-frame excess hit score, frames with the label vs without (0.5 = no information).', '',
         '| phase | label | run | frames with label | without | AUC |', '|---|---|---|---|---|---|']
for ph, lab in (('perovskite', 'label_perovskite'), ('PbI2', 'label_perovskite'), ('delta_FAPbI3', 'label_perovskite'), ('SnO2', 'label_SnO2')):
    for run in sorted({r['run'] for r in rows}) + ['all']:
        sel = [r for r in rows if run == 'all' or r['run'] == run]
        A, np_, nn = auc([float(r[f'{ph}_excess']) for r in sel if r[lab] == 1 and r[f'{ph}_excess'] != ''], [float(r[f'{ph}_excess']) for r in sel if r[lab] == 0 and r[f'{ph}_excess'] != ''])
        lines.append(f'| {ph} | {lab[6:]} | {run} | {np_} | {nn} | {A:.3f} |')
open(f'{a.out}/auc.md', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
