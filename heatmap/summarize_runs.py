"""One table of all finished heatmap runs, read from their evaluate_final.txt (no GPU, no torch, stdlib only).

  python heatmap/summarize_runs.py [--runs_dir /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs] [--out <file>]

Per run (+nms decode = the deployed pipeline; floor 0.1 + top-225): AP on organic / 41, and at score > 0.3: recall, precision,
ring recall, recall of the segment pairs closer than 5 px (NN Euclid <5). Runs with a ring head list their three decode modes
(coarse_rings = the headline decode fixed in advance). Writes a Markdown table; DINO reference numbers are appended."""
import os, re, argparse

HDR = re.compile(r'^\s*\[(?P<name>.+?)\] (?P<ds>organic|41): ap_total (?P<ap>[\d.]+)')
S03 = re.compile(r'score>0\.3: recall (?P<rec>[\d.]+) \(seg [\d.]+, ring (?P<ring>[\d.]+)\) precision (?P<prec>[\d.]+) preds (?P<n>\d+) FP (?P<fp>\d+)')
NN = re.compile(r'recall by NN Euclid dist\s+<5:(?P<c5>[\d.]+)\(n=(?P<n5>\d+)\)')
LABELS = [  # run-dir tag -> readable name, in reading order (unknown runs are appended at the end)
    ('hm_simmim_frozen', 'main (first run, legacy ring target)'), ('hm_random_frozen', 'control: random frozen backbone'),
    ('ridge', 'ridge baseline (fp32)'), ('hm_ridge_lr1e-4_tf32', 'A: lr 1e-4 (TF32)'), ('hm_boxconv1_frozen_ridge_tf32', 'B: boxconv1 backbone (TF32)'),
    ('ridge_long_tf32', 'C: long (bs 8, 120 ep, lr 4.2e-4)'), ('ridge_chanfull_tf32', 'chanfull: HE, B1, B2, mask'),
    ('ridge_stride1_tf32', 'stride-1 output (bs 2)'), ('ridge_chancontrast_tf32', 'contrast channels: log+HE, log, log+CLAHE, mask'),
    ('ridge_he_tf32', 'TF32 plain control'), ('ridge_zeroinv_tf32', 'invalid pixels zeroed'), ('ridge_wide_tf32', 'wide head (4.8 M params)'),
    ('ridge_ringhead_tf32', 'ring head (stride 8)'),
    ('ridge_he_tf32_seed1', 'TF32 plain control, seed 1'), ('ridge_he_tf32_seed2', 'TF32 plain control, seed 2'),
    ('ridge_ringhead16_tf32', 'ring head (stride 16)'), ('ridge_he_tf32_lr5e-4', 'TF32 plain control, lr 5e-4'),
    ('ridge_he_tf32_long120', 'TF32 plain control, 120 epochs (lr drop 90)')]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    f = lambda d, k: (f'{d[k]:.3f}' if k in d else '-')
    rows = []
    for run in sorted(os.listdir(a.runs_dir)):
        p = os.path.join(a.runs_dir, run, 'evaluate_final.txt')
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
    L += ['', '## Reference: DINO detectors (single models, same evaluation)', '',
          '| Run | AP organic | AP 41 | NN<5px org | NN<5px 41 |', '|:--|--:|--:|--:|--:|']
    L += [f'| {n} | {a1} | {a2} | {c1} | {c2} |' for n, a1, a2, c1, c2 in DINO]
    txt = '\n'.join(L); print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
