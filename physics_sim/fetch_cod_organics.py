"""Select organic-semiconductor-like structures from COD and fetch just those CIFs.

Phase 0a of the pygidSIM track. The bank used by the declined phase-P run was 98.5%
perovskite (26,341 of 26,734 entries) while the eval sets are organic thin films, so the
first job is a library whose chemistry actually matches organic/41.

COD is 500k CIFs / 26.6 GB as a tarball, but result.php will run the selection server-side
and hand back full metadata (cell constants, formula, volume) for the hits in one request.
We filter that locally, then rsync only the survivors -- the CIF tree is sharded
cif/<d1>/<d2d3>/<d4d5>/<id>.cif, so an --files-from list fetches them in one session.

Run in mlgid_physics (needs nothing beyond the stdlib here; pymatgen is used by the
downstream screen in generate_bank.py).
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter

COD_QUERY = 'https://www.crystallography.net/cod/result.php'
COD_RSYNC = 'rsync://www.crystallography.net/cif'

# Non-metals that make up organic semiconductors. C and H are required separately.
ORGANIC_ELEMENTS = {
    'C', 'H', 'N', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'As', 'D',
}
# Metal centres that appear in real OSC families -- the user's own library has CuPc and
# ZnPc, so a blanket "no metals" rule would throw away eval-family chemistry. At most one
# such element is allowed per structure (see _classify).
OSC_METALS = {
    'Cu', 'Zn', 'Ni', 'Co', 'Fe', 'Mn', 'Mg', 'Al', 'Pt', 'Pd', 'Ir', 'Ru', 'Sn', 'Pb', 'V', 'Ti',
}

_TOKEN = re.compile(r'([A-Z][a-z]?)([0-9.]*)')


def parse_formula(formula):
    """Element symbols in a COD formula string ('C6 H6', 'C24H12', ...)."""
    return {m.group(1) for m in _TOKEN.finditer(formula or '') if m.group(1)}


def _classify(elements):
    """None if acceptable, else the reason for rejection."""
    if 'C' not in elements or not ({'H', 'D'} & elements):
        return 'no C+H'
    foreign = elements - ORGANIC_ELEMENTS
    if not foreign:
        return None
    if foreign <= OSC_METALS and len(foreign) == 1:
        return None
    return 'non-organic element: ' + ','.join(sorted(foreign))


def query_cod(vmin, vmax, strictmax, out_csv, timeout):
    """Ask COD for candidates and cache the metadata CSV."""
    url = (f'{COD_QUERY}?vmin={vmin}&vmax={vmax}&el1=C&el2=H'
           f'&strictmax={strictmax}&format=csv')
    print(f'querying COD: vol {vmin}-{vmax} A^3, contains C+H, <= {strictmax} element types')
    subprocess.run(['curl', '-sS', '--max-time', str(timeout), url, '-o', out_csv], check=True)
    size = os.path.getsize(out_csv)
    if size < 10000:
        sys.exit(f'COD returned only {size} bytes -- query failed, see {out_csv}')
    print(f'  -> {out_csv} ({size / 1e6:.1f} MB)')
    return out_csv


def select(meta_csv, vmin, vmax, dedupe):
    """Filter the COD metadata down to organic candidates, optionally deduping cells."""
    rejects, seen, chosen = Counter(), set(), []
    with open(meta_csv, newline='') as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith('#'))
        for row in rows:
            cod_id = (row.get('file') or '').strip()
            if not cod_id.isdigit():
                continue
            reason = _classify(parse_formula(row.get('formula') or row.get('calcformula')))
            if reason:
                rejects[reason.split(':')[0]] += 1
                continue
            try:
                vol = float(row['vol'])
                cell = tuple(round(float(row[k]), 2)
                             for k in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'))
            except (TypeError, ValueError, KeyError):
                rejects['unparsable cell'] += 1
                continue
            if not vmin <= vol <= vmax:
                rejects['volume out of range'] += 1
                continue
            # COD carries many redeterminations of the same structure; near-identical cells
            # would give near-identical patterns and overweight those phases in the bank.
            if dedupe:
                if cell in seen:
                    rejects['duplicate cell'] += 1
                    continue
                seen.add(cell)
            chosen.append((cod_id, vol))
    return chosen, rejects


def rsync_cifs(ids, dest, dry_run):
    """Fetch the selected CIFs by their sharded COD paths in a single rsync session."""
    os.makedirs(dest, exist_ok=True)
    listing = os.path.join(dest, '_files_from.txt')
    with open(listing, 'w') as fh:
        for cod_id in ids:
            fh.write(f'{cod_id[0]}/{cod_id[1:3]}/{cod_id[3:5]}/{cod_id}.cif\n')
    cmd = ['rsync', '-a', '--no-relative', '--files-from', listing,
           f'{COD_RSYNC}/', dest + '/']
    if dry_run:
        cmd.insert(1, '--dry-run')
    print(f'rsync {len(ids)} CIFs -> {dest}')
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', default='/mnt/lustre/work/schreiber/szb389/datasets/'
                                      'cif_library_organic/cif')
    ap.add_argument('--vmin', type=float, default=500.0,
                    help='min cell volume, A^3 (organic semiconductor range)')
    ap.add_argument('--vmax', type=float, default=8000.0)
    ap.add_argument('--strictmax', type=int, default=6,
                    help='max distinct element types, passed to COD')
    ap.add_argument('--no-dedupe', action='store_true',
                    help='keep redeterminations of the same cell')
    ap.add_argument('--limit', type=int, default=0, help='cap the selection (0 = no cap)')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--dry-run', action='store_true', help='select and report, fetch nothing')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.dest.rstrip('/')), exist_ok=True)
    meta_csv = os.path.join(os.path.dirname(args.dest.rstrip('/')), 'cod_query.csv')
    if not os.path.exists(meta_csv):
        query_cod(args.vmin, args.vmax, args.strictmax, meta_csv, args.timeout)
    else:
        print(f'reusing cached query {meta_csv}')

    chosen, rejects = select(meta_csv, args.vmin, args.vmax, not args.no_dedupe)
    print(f'\nselected {len(chosen)} structures')
    for reason, n in rejects.most_common():
        print(f'  rejected {n:>7}  {reason}')
    if chosen:
        vols = sorted(v for _, v in chosen)
        print(f'  cell volume A^3: min {vols[0]:.0f}  median {vols[len(vols) // 2]:.0f}  '
              f'max {vols[-1]:.0f}')

    if args.limit:
        chosen = chosen[:args.limit]
        print(f'  capped to {len(chosen)}')
    if not chosen:
        sys.exit('nothing selected')

    ids = [c for c, _ in chosen]
    with open(os.path.join(os.path.dirname(args.dest.rstrip('/')), 'selected_ids.txt'), 'w') as fh:
        fh.write('\n'.join(ids) + '\n')
    rsync_cifs(ids, args.dest, args.dry_run)
    if not args.dry_run:
        n = len([f for f in os.listdir(args.dest) if f.endswith('.cif')])
        print(f'done: {n} CIFs in {args.dest}')


if __name__ == '__main__':
    main()
