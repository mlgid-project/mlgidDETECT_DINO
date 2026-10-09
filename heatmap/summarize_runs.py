"""One table of all finished heatmap runs, read from their evaluate_final.txt (no GPU, no torch, stdlib only).

  python heatmap/summarize_runs.py [--runs_dir /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs] [--out <file>]

Per run (+nms decode = the deployed pipeline; floor 0.1 + top-225): AP on organic / 41, and at score > 0.3: recall, precision,
ring recall, recall of the segment pairs closer than 5 px (NN Euclid <5). Runs with a ring head list their three decode modes
(coarse_rings = the headline decode fixed in advance). Reference numbers for the earlier runs and DINO are appended."""
import os, re, argparse

HDR = re.compile(r'^\s*\[(?P<name>.+?)\] (?P<ds>organic|41): ap_total (?P<ap>[\d.]+)')
S03 = re.compile(r'score>0\.3: recall (?P<rec>[\d.]+) \(seg [\d.]+, ring (?P<ring>[\d.]+)\) precision (?P<prec>[\d.]+) preds (?P<n>\d+) FP (?P<fp>\d+)')
NN = re.compile(r'recall by NN Euclid dist\s+<5:(?P<c5>[\d.]+)\(n=(?P<n5>\d+)\)')
REFS = [('ridge 3e-4 (fp32, earlier run)', '0.6057', '0.7259', '', ''),
        ('run C long (bs 8, 120 ep, lr 4.2e-4)', '0.6237', '0.7178', 'NN<5 >0.3: 0.463', '0.342'),
        ('run B boxconv1 backbone', '0.5732', '0.7489', 'NN<5 >0.3: 0.339', '0.397'),
        ('chanfull (HE, B1, B2, mask)', '0.5792', '0.7048', 'NN<5 >0.3: 0.430', '0.397'),
        ('DINO ssl1', '0.568', '0.744', 'NN<5 >0.3: 0.372', '0.370'),
        ('DINO lr4e5', '0.622', '0.763', 'NN<5 >0.3: 0.388', '0.397'),
        ('DINO boxconv1', '0.588', '0.752', 'NN<5 >0.3: 0.339', '0.370')]


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
    f = lambda d, k: (f'{d[k]:.3f}' if k in d else '  -  ')
    rows = []
    for run in sorted(os.listdir(a.runs_dir)):
        p = os.path.join(a.runs_dir, run, 'evaluate_final.txt')
        if not (run.startswith('hm_') and os.path.exists(p)):
            continue
        r = parse(p)
        for name in sorted({n for n, _ in r if '+nms' in n}):
            o, v = r.get((name, 'organic'), {}), r.get((name, '41'), {})
            tag = run.replace('hm_simmim_frozen_', '').replace('_2.80_1.30', '')
            mode = name.split('ring-head decode=')[1] if 'ring-head' in name else '-'
            rows.append((tag, mode, f(o, 'ap'), f(v, 'ap'), f'{f(o,"rec")}/{f(o,"prec")}/{f(o,"ring")}', f'{f(v,"rec")}/{f(v,"prec")}/{f(v,"ring")}',
                         f'{f(o,"c5")} / {f(v,"c5")}'))
    hd = ('run', 'decode', 'AP organic', 'AP 41', 'organic rec/prec/ring', '41 rec/prec/ring', 'NN<5px recall org/41')
    w = [max(len(str(x[i])) for x in rows + [hd]) for i in range(len(hd))]
    L = ['Heatmap runs, +nms decode (floor 0.1, top-225). rec/prec/ring = recall / precision / ring recall at score > 0.3.', '']
    L.append('  '.join(h.ljust(w[i]) for i, h in enumerate(hd)))
    L += ['  '.join(str(x).ljust(w[i]) for i, x in enumerate(r)) for r in rows]
    L += ['', 'References (AP organic / 41; NN<5px at score>0.3 organic / 41 where known):']
    L += [f'  {n:42s} {a1} / {a2}   {c1} / {c2}'.rstrip(' /') for n, a1, a2, c1, c2 in REFS]
    txt = '\n'.join(L); print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
