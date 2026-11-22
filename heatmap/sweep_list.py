"""List the complete heatmap models for the region sweep: every run folder under hm_runs with args.json, box_coef 3.0,3.0 (the one convention all can be scored against with the same scaled labels),
a checkpoint, and a recorded scaled-GT score in evaluate_final_gtscaled.txt (so only finished runs).

  python heatmap/sweep_list.py --runs /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs --out <models.tsv>

models.tsv columns: run, dir, ckpt, family (run name without seed and box suffix), seed, bb, unfrozen, organic_ap, ap41 (recorded, +nms, scaled GT). The recorded APs are the check for the sweep: the AP recomputed from the
dumped detections on all 49 frames must reproduce them."""
import argparse, glob, json, os, re

p = argparse.ArgumentParser()
p.add_argument('--runs', default='/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'); p.add_argument('--out', required=True)
a = p.parse_args()
rows = []
for d in sorted(glob.glob(f'{a.runs}/*/runs/hm_*') + glob.glob(f'{a.runs}/*/hm_*') + glob.glob(f'{a.runs}/hm_*')):
    if not os.path.exists(f'{d}/args.json'):
        continue
    ar = json.load(open(f'{d}/args.json'))
    if ar.get('box_coef') != '3.0,3.0' or ar.get('chan', 'he') != 'he':
        continue
    ev = next((f for f in ('evaluate_final_gtscaled.txt', 'evaluate_final.txt') if os.path.exists(f'{d}/{f}')), None)
    ck = next((f for f in ('final_checkpoint.pth', 'checkpoint.pth') if os.path.exists(f'{d}/{f}')), None)
    if not ev or not ck:
        continue
    m = dict(re.findall(r'\[hm\+nms\] (organic|41): ap_total ([0-9.]+)', open(f'{d}/{ev}').read()))
    if len(m) < 2:
        continue
    run = os.path.basename(d)
    fam = re.sub(r'_newbox_3\.0_3\.0', '', re.sub(r'_seed\d+', '_seedN', run))
    seed = ar.get('seed', re.search(r'seed(\d+)', run).group(1) if re.search(r'seed(\d+)', run) else '')
    rows.append([run, d, ck, fam, seed, ar.get('bb'), int(bool(ar.get('unfreeze'))), m['organic'], m['41']])
with open(a.out, 'w') as f:
    f.write('run\tdir\tckpt\tfamily\tseed\tbb\tunfrozen\torganic_ap\tap41\n')
    for r in rows:
        f.write('\t'.join(str(x) for x in r) + '\n')
print(len(rows), 'models,', len({r[3] for r in rows}), 'families ->', a.out)
