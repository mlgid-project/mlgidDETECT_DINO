"""One table of all finished heatmap runs, read from their evaluate_final.txt (no GPU, no torch, stdlib only).

  python heatmap/summarize_runs.py [--runs_dir /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs] [--out <file>]

Per run (+nms decode = the deployed pipeline; floor 0.1 + top-225): AP on organic / 41, and at score > 0.3: recall, precision,
ring recall, recall of the segment pairs closer than 5 px (NN Euclid <5). Runs with a ring head list their three decode modes
(coarse_rings = the headline decode fixed in advance). Writes a Markdown table; DINO reference numbers are appended."""
import os, re, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs_layout import run_dirs

HDR = re.compile(r'^\s*\[(?P<name>.+?)\] (?P<ds>organic|41): ap_total (?P<ap>[\d.]+)')
S03 = re.compile(r'score>0\.3: recall (?P<rec>[\d.]+) \(seg [\d.]+, ring (?P<ring>[\d.]+)\) precision (?P<prec>[\d.]+) preds (?P<n>\d+) FP (?P<fp>\d+)')
NN = re.compile(r'recall by NN Euclid dist\s+<5:(?P<c5>[\d.]+)\(n=(?P<n5>\d+)\)')
LABELS = [  # run-dir tag -> readable name, in reading order (unknown runs are appended at the end)
    ('hm_simmim_frozen', 'main (first run, legacy ring target)'), ('hm_random_frozen', 'control: random frozen backbone'),
    ('ridge', 'ridge baseline (fp32)'), ('hm_ridge_lr1e-4_tf32', 'A: lr 1e-4 (TF32)'), ('hm_boxconv1_frozen_ridge_tf32', 'B: boxconv1 backbone (TF32)'),
    ('ridge_long_tf32', 'C: long (bs 8, 120 ep, lr 4.2e-4)'),
    ('ridge_long_tf32_seed1', 'C replicate, seed 1'), ('ridge_long_tf32_seed2', 'C replicate, seed 2'), 
    ('ridge_coord_tf32_seed1', 'coord channels (HE, x, y), seed 1'), ('ridge_realsim_single_tf32_seed42', 'real-background sim (single structure, 25% rings), plain recipe, seed 42'), ('ridge_realsimE1_tf32_seed42', 'real-background sim, calibrated, E1 (rings + segments per frame), plain recipe, seed 42'), ('ridge_realsimE3_tf32_seed42', 'real-background sim, calibrated, E3 (one structure per image), plain recipe, seed 42'), ('hm_boxconv1_frozen_ridge_tf32_seed1', 'B replicate, seed 1'), ('hm_boxconv1_frozen_ridge_tf32_seed2', 'B replicate, seed 2'), ('hm_boxconv1_frozen_ridge_tf32_seed3', 'B replicate, seed 3'),
    ('hm_simmim_unfrozen_ridge_he_tf32_seed1', 'unfrozen swin (bs 4, 60 ep), seed 1'), ('hm_simmim_unfrozen_ridge_he_tf32_seed2', 'unfrozen swin (bs 4, 60 ep), seed 2'), ('ridge_chanfull_tf32', 'chanfull: HE, B1, B2, mask'),
    ('ridge_stride1_tf32', 'stride-1 output (bs 2)'), ('ridge_chancontrast_tf32', 'contrast channels: log+HE, log, log+CLAHE, mask'),
    ('ridge_he_tf32', 'TF32 plain control'), ('ridge_zeroinv_tf32', 'invalid pixels zeroed'), ('ridge_wide_tf32', 'wide head (4.8 M params)'),
    ('ridge_ringhead_tf32', 'ring head (stride 8)'),
    ('ridge_he_tf32_seed1', 'TF32 plain control, seed 1'), ('ridge_he_tf32_seed2', 'TF32 plain control, seed 2'),
    ('ridge_ringhead16_tf32', 'ring head (stride 16)'), ('ridge_he_tf32_lr5e-4', 'TF32 plain control, lr 5e-4'),
    ('ridge_he_tf32_long120', 'TF32 plain control, 120 epochs (lr drop 90)'),
    ('hm_simmim_UNFROZEN_ridge_long_tf32', 'unfrozen swin (bs 8, 120 ep, galvani), seed 42'), ('hm_simmim_UNFROZEN_ridge_long_tf32_seed1', 'unfrozen swin (bs 8, 120 ep, galvani), seed 1'),
    ('hm_simmim_UNFROZEN_ridge_long_tf32_seed2', 'unfrozen swin (bs 8, 120 ep, galvani), seed 2'),
    ('hm_simmim_UNFROZEN_ridge_long_tf32_seed1_bblr4e-5', 'unfrozen 120 ep, bblr 4e-5, seed 1'),
    ('hm_simmim_UNFROZEN_ridge_long_tf32_seed2_bblr4e-5', 'unfrozen 120 ep, bblr 4e-5, seed 2'),
    ('hm_simmim_UNFROZEN_ridge_long_tf32_seed3_bblr4e-5', 'unfrozen 120 ep, bblr 4e-5, seed 3'),
    ('ridge_realsimE1b_tf32_seed42', 'realsim E1b (E1 + correlated bkg noise, galvani)'),
    ('ridge_realsimE5_tf32_seed42', 'realsim E5 (E1b + 4 low-q masks + ring_x_share), seed 42, galvani'),
    ('ridge_realsimE6_tf32_seed42', 'realsim E6 (E5 + width grows with q + ~21 segments/frame), seed 42, galvani'),
    ('ridge_realsimE7_tf32_seed42', 'realsim E7 (E6 + flat background along q + ring brightness shuffle), seed 42, galvani'),
    ('ridge_realsimE8_tf32_seed42', 'realsim E8 (E6 appearance, sparse frames ~5 rings + ~5 segments), seed 42, galvani'),
    ('ridge_realsimE9_tf32_seed42', 'realsim E9 (E6 + 20 % wedge-only masks, no detector gaps), seed 42, galvani'),
    ('ridge_realsimE10_tf32_seed42', 'realsim E10 (E8 structure, rings_cap 6-16, spots_cap 10-40, bkg_noise_edge_fix), seed 42, galvani'),
    ('ridge_realsimE5_tf32_seed2', 'realsim E5, seed 2, colorbox'), ('ridge_realsimE5_tf32_seed3', 'realsim E5, seed 3, colorbox'),
    ('hm_boxconv1_unfrozen_ridge_tf32_seed1_bblr4e-5', 'unfrozen boxconv1 backbone (bs 4, 60 ep, colorbox), bblr 4e-5, seed 1'),
    ('hm_simmim_unfrozen_ridge_he_tf32_seed1_bblr4e-5', 'unfrozen swin (bs 4, 60 ep, colorbox), bblr 4e-5, seed 1'),
    ('hm_simmim_unfrozen_ridge_he_tf32_seed1_bblr8e-5', 'unfrozen swin (bs 4, 60 ep, colorbox), bblr 8e-5, seed 1')]
NAME = dict(LABELS)
REPLICATES = [  # recipes run more than once: (group label, run tags); only runs that exist and have an evaluation are used
    ('Plain recipe, frozen SimMIM (bs 4 x 250, 60 ep, TF32)', ['ridge_he_tf32', 'ridge_he_tf32_seed1', 'ridge_he_tf32_seed2']),
    ('Run C recipe, frozen SimMIM (bs 8 x 125, 120 ep, lr 4.2e-4)', ['ridge_long_tf32', 'ridge_long_tf32_seed1', 'ridge_long_tf32_seed2']),
    ('Run B, boxconv1 backbone frozen (bs 4 x 250, 60 ep)', ['hm_boxconv1_frozen_ridge_tf32', 'hm_boxconv1_frozen_ridge_tf32_seed1', 'hm_boxconv1_frozen_ridge_tf32_seed2', 'hm_boxconv1_frozen_ridge_tf32_seed3']),
    ('Unfrozen swin (bs 4 x 250, 60 ep, colorbox)', ['hm_simmim_unfrozen_ridge_he_tf32_seed1', 'hm_simmim_unfrozen_ridge_he_tf32_seed2']),
    ('Unfrozen swin (bs 8 x 125, 120 ep, galvani)', ['hm_simmim_UNFROZEN_ridge_long_tf32', 'hm_simmim_UNFROZEN_ridge_long_tf32_seed1', 'hm_simmim_UNFROZEN_ridge_long_tf32_seed2']),
    ('Unfrozen swin (bs 8 x 125, 120 ep, galvani), backbone lr 4e-5', ['hm_simmim_UNFROZEN_ridge_long_tf32_seed1_bblr4e-5', 'hm_simmim_UNFROZEN_ridge_long_tf32_seed2_bblr4e-5', 'hm_simmim_UNFROZEN_ridge_long_tf32_seed3_bblr4e-5']),
    ('Realsim E5 (E1b + 4 low-q masks + ring_x_share; plain recipe)', ['ridge_realsimE5_tf32_seed42', 'ridge_realsimE5_tf32_seed2', 'ridge_realsimE5_tf32_seed3'])]
NAME = dict(LABELS)
DINO = [('DINO ssl1', '0.568', '0.744', '0.372', '0.370'), ('DINO lr 4e-5', '0.622', '0.763', '0.388', '0.397'),
        ('DINO boxconv1', '0.588', '0.752', '0.339', '0.370')]


def parse(path):
    res, cur = {}, None
    for ln in open(path):
        m = HDR.match(ln)
        if m:
            cur = (m['name'], m['ds']); res[cur] = dict(ap=float(m['ap'])); got03 = False; continue
        if cur is None:
            continue
        m = S03.search(ln)
        if m:
            res[cur].update(rec=float(m['rec']), prec=float(m['prec']), ring=float(m['ring']), fp=int(m['fp'])); got03 = True; continue
        m = NN.search(ln)
        if m and got03 and 'c5' not in res[cur]:
            res[cur].update(c5=float(m['c5']), n5=int(m['n5']))
    return res


def _stats(vals):
    n = len(vals); m = sum(vals) / n
    sd = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else None
    return n, m, sd


def repeated_tables(byrun):
    """Recipes that were run more than once: mean, sample SD (n-1), standard error (SD / sqrt(n)) and range over the runs/seeds."""
    G = [(lab, [(t, byrun[t]) for t in tags if t in byrun]) for lab, tags in REPLICATES]
    G = [(lab, rs) for lab, rs in G if len(rs) >= 2]
    if not G:
        return []
    fmt = lambda x, d=3: ('-' if x is None else f'{x:.{d}f}')
    L = ['', '## Recipes run more than once (+nms decode, floor 0.1, top-225)', '',
         'Mean over the runs of one recipe (different seeds). **SD** = sample standard deviation (n-1), **SE** = SD / sqrt(n), **range** = lowest to highest run. '
         'With n = 2 or 3 the SD and SE are themselves very uncertain; read the range and the single values next to them. Only recipes with at least 2 finished runs are listed.', '',
         '### AP', '',
         '| Recipe | n | Organic mean | SD | SE | range | 41 mean | SD | SE | range |', '|:--|--:|--:|--:|--:|:--|--:|--:|--:|:--|']
    for lab, rs in G:
        cells = []
        for k in (0, 1):
            v = [r[1][k]['ap'] for r in rs if 'ap' in r[1][k]]
            n, m, sd = _stats(v)
            cells += [fmt(m), fmt(sd), fmt(sd / n ** 0.5 if sd is not None else None), f'{min(v):.3f}-{max(v):.3f}']
        L.append(f'| {lab} | {len(rs)} | ' + ' | '.join(cells) + ' |')
    L += ['', '### Score > 0.3 and close pairs (mean ± SD)', '',
          '| Recipe | n | Recall org | Prec org | Ring org | Recall 41 | Prec 41 | Ring 41 | NN<5px org | NN<5px 41 |', '|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for lab, rs in G:
        cells = []
        for k, key in ((0, 'rec'), (0, 'prec'), (0, 'ring'), (1, 'rec'), (1, 'prec'), (1, 'ring'), (0, 'c5'), (1, 'c5')):
            v = [r[1][k][key] for r in rs if key in r[1][k]]
            if not v:
                cells.append('-'); continue
            n, m, sd = _stats(v)
            cells.append(f'{m:.3f} ± {sd:.3f}' if sd is not None else f'{m:.3f}')
        L.append(f'| {lab} | {len(rs)} | ' + ' | '.join(cells) + ' |')
    L += ['', '### The single runs behind it (AP organic / AP 41)', '']
    for lab, rs in G:
        L.append(f'- **{lab}**: ' + '; '.join(f"{NAME.get(t, t)} {o.get('ap', float('nan')):.4f} / {v.get('ap', float('nan')):.4f}" for t, (o, v) in rs))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    f = lambda d, k: (f'{d[k]:.3f}' if k in d else '-')
    rows = []
    byrun = {}                                                   # tag -> (organic dict, 41 dict) of the +nms decode, for the repeated-runs table
    for run, rdir in run_dirs(a.runs_dir):                      # works with runs at the top level or in group subfolders
        p = os.path.join(rdir, 'evaluate_final.txt')
        if not (run.startswith('hm_') and os.path.exists(p)):
            continue
        r = parse(p)
        tag = run.replace('hm_simmim_frozen_', '').replace('_2.80_1.30', '')
        tag = tag if tag in NAME else (run if run in NAME else tag)
        for name in sorted({n for n, _ in r if '+nms' in n}):
            if 'ring-head' not in name and any('ring-head' in n for n, _ in r):
                continue                                           # ring-head runs: list the three explicit decode modes only
            o, v = r.get((name, 'organic'), {}), r.get((name, '41'), {})
            mode = name.split('ring-head decode=')[1] if 'ring-head' in name else ''
            if not mode:
                byrun[tag] = (o, v)
            lab = NAME.get(tag, tag) + (f' [{mode}]' if mode else '')
            order = [t for t, _ in LABELS].index(tag) if tag in NAME else 99
            rows.append((order, lab, o, v))
    rows.sort(key=lambda x: (x[0], x[1]))
    best_o = max([o['ap'] for _, _, o, _ in rows if 'ap' in o] or [0]); best_v = max([v['ap'] for _, _, _, v in rows if 'ap' in v] or [0])
    bold = lambda txt, d, b: (f'**{txt}**' if d.get('ap') == b else txt)
    L = ['# Heatmap detector: results of all finished runs', '',
         'Decode: top-225 peaks, class-aware NMS, score floor 0.1 (the headline convention). **AP** uses that whole set. '
         '**Recall / precision / ring recall** use only boxes with score > 0.3. **NN<5px** = recall of the segment peaks that have another peak closer than 5 px (score > 0.3). '
         'Best AP per set in bold. Single seeds: differences of about 0.02 AP are within noise.', '',
         '| Run | Decode | AP organic | AP 41 | Recall org | Prec org | Ring org | Recall 41 | Prec 41 | Ring 41 | NN<5px org | NN<5px 41 |',
         '|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for _, lab, o, v in rows:
        lab, _, mode = lab.partition(' [')
        L.append(f'| {lab} | {mode.rstrip("]") or "-"} | {bold(f(o, "ap"), o, best_o)} | {bold(f(v, "ap"), v, best_v)} | {f(o, "rec")} | {f(o, "prec")} | {f(o, "ring")} | '
                 f'{f(v, "rec")} | {f(v, "prec")} | {f(v, "ring")} | {f(o, "c5")} | {f(v, "c5")} |')
    L += repeated_tables(byrun)
    L += ['', '## Reference: DINO detectors (single models, same evaluation)', '',
          '| Run | AP organic | AP 41 | NN<5px org | NN<5px 41 |', '|:--|--:|--:|--:|--:|']
    L += [f'| {n} | {a1} | {a2} | {c1} | {c2} |' for n, a1, a2, c1, c2 in DINO]
    txt = '\n'.join(L); print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
