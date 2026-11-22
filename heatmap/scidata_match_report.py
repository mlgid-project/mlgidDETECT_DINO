"""Score the mlgidMATCH results (scidata_pipeline.py --stage match) of the SciData GIWAXS benchmark against the curated class names (mlgidgui env).

  python heatmap/scidata_match_report.py --h5 <run1.h5> [--h5 <run3.h5>] --out <dir>

Per frame: for every phase the highest matching probability over its CIFs and over the matched_rings_* / matched_segments_* groups (0 when nothing matched).
  perovskite = alpha-FAPbI3 cubic, MAPbI3 cubic, MAPbI3 tetragonal;  PbI2;  delta-FAPbI3;  SnO2 = both rutile CIFs.
Labels (from the class name): perovskite expected when 'pvk' is in it, SnO2 expected when 'sno2' is in it (that includes the SnO2 + PVK stacks), PbI2 / delta: no label (reported only as how often they are called).
Output: <out>/match_frames.csv, <out>/match_report.md (AUC with the probability as score; precision / recall at probability >= 0.5; call rates per class family)."""
import argparse, csv, json, os
from pathlib import Path
import h5py, numpy as np

p = argparse.ArgumentParser(); p.add_argument('--h5', action='append', required=True); p.add_argument('--out', required=True); p.add_argument('--thr', type=float, default=0.5)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
PH = {'perovskite': ('alphaFAPbI3', 'MAPbI3_cubic', 'MAPbI3_tetra'), 'PbI2': ('PbI2',), 'delta_FAPbI3': ('deltaFAPbI3',), 'SnO2': ('SnO2',)}
fam = lambda c: 'PVK' if 'pvk' in c.lower() else ('SnO2 (no PVK)' if 'sno2' in c.lower() else 'OPV/other')
rows = []
for h5 in a.h5:
    idx = json.load(open(h5.replace('.h5', '_index.json')))
    with h5py.File(h5, 'r') as f:
        for n in f:
            fr = f[n]['data/analysis']; fr = fr[list(fr.keys())[0]] if len(fr) else None
            best = {k: 0.0 for k in PH}; nm = 0
            if fr is not None:
                for k in fr.keys():
                    if k.startswith('matched'):
                        r = fr[k][()].reshape(-1)[0]; cif = r['CIF']; cif = cif.decode() if isinstance(cif, bytes) else str(cif); pr = float(r['probability']); nm += 1
                        for ph, names in PH.items():
                            if any(cif.startswith(x) for x in names):
                                best[ph] = max(best[ph], pr)
            c = idx[n]['cls']
            rows.append(dict(run=Path(h5).stem, entry=n, cls=c, family=fam(c), raw=idx[n]['raw'], ai=idx[n]['ai'], n_matched_groups=nm, label_perovskite=int('pvk' in c.lower()), label_SnO2=int('sno2' in c.lower()), **{f'p_{k}': round(v, 3) for k, v in best.items()}))
with open(f'{a.out}/match_frames.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def auc(pos, neg):
    pos, neg = np.array(pos, float), np.array(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


L = [f'# mlgidMATCH vs the class labels', '', f'{len(rows)} frames, runs {sorted({r["run"] for r in rows})}; a phase is "called" when its best matching probability is >= {a.thr}.', '', '| phase | label | run | with label | without | AUC (probability) | recall | precision | called in frames without label |', '|---|---|---|---|---|---|---|---|---|']
for ph, lab in (('perovskite', 'label_perovskite'), ('SnO2', 'label_SnO2')):
    for run in sorted({r['run'] for r in rows}) + ['all']:
        s = [r for r in rows if run == 'all' or r['run'] == run]
        pos = [r[f'p_{ph}'] for r in s if r[lab] == 1]; neg = [r[f'p_{ph}'] for r in s if r[lab] == 0]
        tp = sum(x >= a.thr for x in pos); fp = sum(x >= a.thr for x in neg)
        L.append(f'| {ph} | {lab[6:]} | {run} | {len(pos)} | {len(neg)} | {auc(pos, neg):.3f} | {tp / max(len(pos), 1):.2f} | {tp / max(tp + fp, 1):.2f} | {fp} |')
L += ['', 'Call rate per class family (probability >= threshold):', '', '| family | frames | perovskite | PbI2 | delta-FAPbI3 | SnO2 |', '|---|---|---|---|---|---|']
for fm in ('PVK', 'SnO2 (no PVK)', 'OPV/other'):
    s = [r for r in rows if r['family'] == fm]
    L.append(f'| {fm} | {len(s)} | ' + ' | '.join(f'{np.mean([r[f"p_{ph}"] >= a.thr for r in s]):.0%}' for ph in PH) + ' |')
open(f'{a.out}/match_report.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
