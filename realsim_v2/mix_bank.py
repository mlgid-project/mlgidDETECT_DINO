"""Fill a peak bank (build_peak_bank.py) with structures of the old organic bank up to N structures in total.

    python realsim_v2/mix_bank.py PEAKBANK.npz OUT.npz N_TOTAL [ORGANIC_BANK.npz] [SEED]

Each added organic CIF becomes an oriented-only (p 0.25), rings-only (0.25) or both (0.5) structure (the mode mix of the generator): its oriented entry is one random orientation of the CIF.
The added entries have NaN sigma_q / sigma_chi / eta: realsim `v2_widths` borrows the stored widths of the v2/v3 peaks of the same q for them (RealBkgSimulation._borrow).
CIFs that are already in the peak bank do not occur (MaterialIds of GNoME and CIF names of the organic bank differ).
"""
import sys, os, collections, numpy as np

ORG = os.path.join(os.environ.get('GIWAXS_WORK', '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data'), 'datasets/cif_library_organic/bank/bank_organic_hkl.npz')


def mix(peak_bank, out, n_total, organic=ORG, seed=0):
    b = dict(np.load(peak_bank, allow_pickle=False))
    o = {k: v for k, v in np.load(organic, allow_pickle=False).items()}       # read every array once
    n_have = len(set(b['entry_cif'].tolist()))
    n_add = int(n_total) - n_have
    if n_add <= 0:
        np.savez(out, **b); print('peak bank already has', n_have, 'structures'); return
    rng = np.random.default_rng(seed)
    ok, oc, ocif = np.asarray(o['entry_kind']), np.asarray(o['entry_count']), np.asarray(o['entry_cif'])
    by = collections.defaultdict(lambda: [[], None])          # cif -> [oriented entries, powder entry]
    for e in np.nonzero(oc > 0)[0]:
        if ok[e] == 'oriented':
            by[str(ocif[e])][0].append(int(e))
        else:
            by[str(ocif[e])][1] = int(e)
    cifs = sorted(by)
    cifs = [cifs[i] for i in rng.permutation(len(cifs))[:n_add]]
    q, chi, I, start, count, kind, cif = [], [], [], [], [], [], []
    pos = len(b['q'])
    for c in cifs:
        ors, pw = by[c]
        m = rng.random()
        want = ['oriented'] if m < 0.25 else ['powder'] if m < 0.5 else ['oriented', 'powder']
        for kd in want:
            e = (ors[int(rng.integers(len(ors)))] if ors else None) if kd == 'oriented' else pw
            if e is None:
                continue
            s0, c0 = int(o['entry_start'][e]), int(oc[e])
            q.append(o['q'][s0:s0 + c0]); chi.append(o['chi'][s0:s0 + c0]); I.append(o['intensity'][s0:s0 + c0])
            start.append(pos); count.append(c0); kind.append(kd); cif.append('org_' + c); pos += c0
    n = pos - len(b['q'])
    nan = np.full(n, np.nan, np.float32)
    res = dict(q=np.concatenate([b['q'], np.concatenate(q).astype(np.float32)]), chi=np.concatenate([b['chi'], np.concatenate(chi).astype(np.float32)]),
               intensity=np.concatenate([b['intensity'], np.concatenate(I).astype(np.float32)]),
               sigma_q=np.concatenate([b['sigma_q'], nan]), sigma_chi=np.concatenate([b['sigma_chi'], nan]), eta=np.concatenate([b['eta'], nan]),
               entry_start=np.concatenate([b['entry_start'], np.array(start, np.int64)]), entry_count=np.concatenate([b['entry_count'], np.array(count, np.int64)]),
               entry_kind=np.concatenate([b['entry_kind'], np.array(kind)]), entry_cif=np.concatenate([b['entry_cif'], np.array(cif)]))
    np.savez(out, **res)
    print(f'{n_have} peak-bank structures + {len(cifs)} organic = {n_have + len(cifs)}; entries {len(res["entry_start"])} (oriented {int((res["entry_kind"] == "oriented").sum())}, powder {int((res["entry_kind"] == "powder").sum())}); peaks {len(res["q"])}')


if __name__ == '__main__':
    mix(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else ORG, int(sys.argv[5]) if len(sys.argv) > 5 else 0)
