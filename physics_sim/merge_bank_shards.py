"""Concatenate sharded bank npz files into one bank, identical to an unsharded build.

`entry_start` is an offset into the concatenated peak arrays, so it cannot simply be stacked --
each shard's offsets have to be shifted by the number of peaks already written. Everything else
appends. The merged manifest keeps every shard's entry records in order.

  python physics_sim/merge_bank_shards.py --parts DIR --out bank.npz
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', required=True, help='directory holding shard_*.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expect', type=int, default=None, help='fail unless this many shards found')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.parts, 'shard_*.npz')))
    if not files:
        raise SystemExit(f'[FATAL] no shard_*.npz under {a.parts}')
    if a.expect and len(files) != a.expect:
        raise SystemExit(f'[FATAL] found {len(files)} shards, expected {a.expect}')
    print(f'merging {len(files)} shards')

    q, chi, inten, starts, counts, kinds, cifs = [], [], [], [], [], [], []
    metas, errors, off = [], [], 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        n = len(z['q'])
        q.append(z['q']); chi.append(z['chi']); inten.append(z['intensity'])
        starts.append(z['entry_start'] + off)          # shift into the merged peak array
        counts.append(z['entry_count'])
        kinds.append(z['entry_kind']); cifs.append(z['entry_cif'])
        off += n
        m = os.path.splitext(f)[0] + '_manifest.json'
        if os.path.exists(m):
            d = json.load(open(m))
            metas.extend(d.get('entries', [])); errors.extend(d.get('errors', []))
            base = d
        print(f'  {os.path.basename(f)}: {len(z["entry_start"])} entries, {n} peaks')

    kinds = np.concatenate(kinds)
    out = dict(q=np.concatenate(q), chi=np.concatenate(chi),
               intensity=np.concatenate(inten),
               entry_start=np.concatenate(starts), entry_count=np.concatenate(counts),
               entry_kind=kinds, entry_cif=np.concatenate(cifs))
    assert int(out['entry_start'][-1] + out['entry_count'][-1]) == len(out['q']), \
        'offset bookkeeping is wrong'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)

    n_pow = int((kinds == 'powder').sum())
    mf = dict(base) if 'base' in dir() else {}
    mf.update(n_entries=len(out['entry_start']), n_powder=n_pow,
              n_oriented=len(out['entry_start']) - n_pow, shard=None,
              merged_from=[os.path.basename(f) for f in files],
              entries=metas, errors=errors[:2000])
    json.dump(mf, open(os.path.splitext(a.out)[0] + '_manifest.json', 'w'), indent=1)
    print(f'\nwrote {a.out}  {os.path.getsize(a.out)/1e9:.2f} GB')
    print(f'  {len(out["entry_start"])} entries ({n_pow} powder, '
          f'{len(out["entry_start"])-n_pow} oriented), {len(out["q"])} peaks')


if __name__ == '__main__':
    main()
