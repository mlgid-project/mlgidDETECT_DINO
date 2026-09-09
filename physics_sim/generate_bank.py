"""Generate a physics peak bank from a CIF library with pygidsim.

Ported from branch `development` (git show e14f8e9:physics_sim/generate_bank.py) with three
changes: the CIF directory and output path are parameters instead of constants bound to the old
perovskite staging area; the per-CIF work is parallelised (60k CIFs at ~1.1 s each is 18
CPU-hours single-threaded); and screening is folded into the worker, since a CIF that cannot be
simulated simply reports its error into the audit.

For every usable CIF this simulates:
  * the POWDER pattern  (orientation=None)   -> rings:  (|q|, I),      chi = -1 sentinel
  * K ORIENTED patterns (explicit random fiber axes, recorded per entry) -> (|q|, chi_deg, I)

Everything is stored geometry-independently in polar physics coordinates (|q| in A^-1, chi in
degrees from the q_xy axis). The training-time renderer maps them to pixels per image via
x = q/q_max*1024, y = chi/90*512 with a sampled q_max.

EXCLUSIONS (user mandate; from eval_matched.json produced by build_exclusion_list.py):
  * a CIF matched in eval as powder/rings -> its POWDER entry is dropped;
  * a CIF matched in eval with orientation v -> oriented entries within ORIENT_MARGIN_DEG of v
    are dropped (other orientations of the same structure remain usable, per user).

Runs in the `mlgid_physics` env (CPU only). Usage:
  python physics_sim/generate_bank.py --cif-dir <dir> --out <bank.npz> [--limit N] [--jobs N]
"""
import argparse
import json
import multiprocessing as mp
import os
import warnings

import numpy as np

# hypot(3.5, 3.5) = 4.95 = the organic eval detector's q_max (every entry of
# organic_labeled.h5; util/pygidloader.py:187). The old bank used 3.0 -> 4.24, which structurally
# left the outer 14% of an organic frame empty of physics peaks. 41's q_max is 3.82, well inside.
Q_XY_MAX = 3.5
Q_Z_MAX = 3.5
ORIENT_MARGIN_DEG = 10.0
TOP_PEAKS = 200          # keep at most this many peaks per entry (by intensity)
MIN_PEAKS = 3            # drop entries with fewer visible peaks
MIN_Q = 0.05

_PARAMS = None           # per-worker ExpParameters; the form-factor table costs ~0.3 s to build


def to_cartesian(u, rec):
    """Map a crystal-basis (Miller) direction to Cartesian.

    CRITICAL: pygidsim interprets `orientation` in the CRYSTAL basis and maps it as
    `orientation @ rec`. Angles between such directions must be measured AFTER this map -- a dot
    product in Miller space equals the physical angle only for cubic cells, and most of these
    structures are triclinic/monoclinic. Getting this wrong silently under-excludes matched
    eval orientations and makes the "well separated" test below meaningless.
    """
    c = np.asarray(u, float) @ np.asarray(rec, float)
    return c / (np.linalg.norm(c) + 1e-12)


def ang_deg(u, v, rec):
    """Physical angle (deg) between two crystal-basis directions; axes are sign-invariant."""
    uc, vc = to_cartesian(u, rec), to_cartesian(v, rec)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(uc, vc))), -1.0, 1.0))))


def orientation_vectors(k, rng, rec):
    """k random fiber axes as crystal-basis directions, physically well separated."""
    vs, tries = [], 0
    while len(vs) < k and tries < k * 50:
        tries += 1
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n < 1e-8:
            continue
        v = v / n
        if to_cartesian(v, rec)[2] < 0:      # upper hemisphere in REAL space
            v = -v
        if all(ang_deg(v, u, rec) > 15.0 for u in vs):
            vs.append(v)
    return vs


def _init_worker():
    global _PARAMS
    from pygidsim.experiment import ExpParameters
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _PARAMS = ExpParameters(q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX)


def _simulate_cif(job):
    """One CIF -> (entries, errors). Runs in a worker; must not raise."""
    path, k_orient, seed, powder_banned, orient_banned = job
    name = os.path.basename(path)
    from pygidsim.giwaxs_sim import GIWAXSFromCif
    entries, errors = [], []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            el = GIWAXSFromCif(path, _PARAMS)
        rec = np.asarray(el.giwaxs.rec, float)
    except BaseException as e:                 # noqa: BLE001 - unparsable CIFs are expected
        return [], [[name, 'load: ' + repr(e)[:80]]]

    if not powder_banned:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                q1d, i1d = el.giwaxs.giwaxs_sim(orientation=None)
            q1d, i1d = np.asarray(q1d, float), np.asarray(i1d, float)
            keep = (q1d > MIN_Q) & np.isfinite(i1d) & (i1d > 0)
            q1d, i1d = q1d[keep], i1d[keep]
            if len(q1d) >= MIN_PEAKS:
                top = np.argsort(i1d)[::-1][:TOP_PEAKS]
                entries.append((name, 'powder', None, q1d[top],
                                np.full(len(top), -1.0), i1d[top]))
        except BaseException as e:             # noqa: BLE001
            errors.append([name, 'powder: ' + repr(e)[:80]])

    rng = np.random.default_rng(seed)
    for v in orientation_vectors(k_orient, rng, rec):
        if any(ang_deg(v, b, rec) < ORIENT_MARGIN_DEG for b in orient_banned):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                q2d, i2d = el.giwaxs.giwaxs_sim(orientation=np.asarray(v))
            q2d, i2d = np.asarray(q2d, float), np.asarray(i2d, float)
            if q2d.ndim != 2 or q2d.shape[0] != 2 or q2d.shape[1] < MIN_PEAKS:
                continue
            qxy, qz = np.abs(q2d[0]), np.abs(q2d[1])
            qabs = np.hypot(qxy, qz)
            chi = np.degrees(np.arctan2(qz, qxy))       # 0 = in-plane (q_xy axis)
            keep = (qabs > MIN_Q) & np.isfinite(i2d) & (i2d > 0)
            qabs, chi, ii = qabs[keep], chi[keep], i2d[keep]
            if len(qabs) < MIN_PEAKS:
                continue
            top = np.argsort(ii)[::-1][:TOP_PEAKS]
            entries.append((name, 'oriented', [round(float(x), 5) for x in v],
                            qabs[top], chi[top], ii[top]))
        except BaseException as e:              # noqa: BLE001
            errors.append([name, 'orient: ' + repr(e)[:80]])
    return entries, errors


def load_exclusions(path):
    """-> (powder_excluded: set[cif], oriented_excluded: {cif: [orientation vectors]})"""
    if not path or not os.path.isfile(path):
        return set(), {}
    with open(path) as f:
        d = json.load(f)
    powder, oriented = set(), {}
    for rows in d.get('matches', {}).values():
        for r in rows:
            cif = r['cif']
            if r.get('peaks_type') == 'rings' or r.get('orientation') in (None, 'powder'):
                powder.add(cif)
            o = r.get('orientation')
            if isinstance(o, (list, tuple)) and len(o) == 3:
                oriented.setdefault(cif, []).append([float(x) for x in o])
    return powder, oriented


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cif-dir', required=True)
    ap.add_argument('--out', required=True, help='output .npz (manifest written alongside)')
    ap.add_argument('--k-orient', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0,
                    help='use only N CIFs, sampled at random (smoke / distribution gate)')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed for --limit')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument('--exclusions', default='', help='eval_matched.json from build_exclusion_list')
    ap.add_argument('--no-exclusions', action='store_true',
                    help='build without exclusions (PRELIMINARY / statistics-only bank)')
    args = ap.parse_args()

    if not args.no_exclusions and not (args.exclusions and os.path.isfile(args.exclusions)):
        raise SystemExit('[FATAL] no exclusion list -- run build_exclusion_list.py, or pass '
                         '--no-exclusions for a PRELIMINARY bank not usable for training')

    names = sorted(f for f in os.listdir(args.cif_dir) if f.endswith('.cif'))
    if args.limit and args.limit < len(names):
        # sample rather than truncate: COD ids are ordered by deposition, so the first N would
        # be a biased slice of the library rather than a picture of its distribution
        idx = np.random.default_rng(args.seed).choice(len(names), args.limit, replace=False)
        names = [names[i] for i in sorted(idx)]
    powder_excl, orient_excl = load_exclusions(args.exclusions)
    print(f'[bank] {len(names)} CIFs from {args.cif_dir}, {args.jobs} workers')
    print(f'[bank] exclusions: {len(powder_excl)} powder, '
          f'{sum(len(v) for v in orient_excl.values())} oriented on {len(orient_excl)} CIFs')

    jobs = [(os.path.join(args.cif_dir, n), args.k_orient, i,
             n in powder_excl, orient_excl.get(n, [])) for i, n in enumerate(names)]

    q_all, chi_all, i_all = [], [], []
    starts, counts, kinds, cifs, metas = [], [], [], [], []
    errors, n_peaks = [], 0
    with mp.Pool(args.jobs, initializer=_init_worker) as pool:
        for done, (entries, errs) in enumerate(
                pool.imap_unordered(_simulate_cif, jobs, chunksize=8), 1):
            errors.extend(errs)
            for name, kind, orient, q, chi, inten in entries:
                starts.append(n_peaks)
                counts.append(len(q))
                kinds.append(kind)
                cifs.append(name)
                metas.append(dict(cif=name, kind=kind, orientation=orient))
                q_all.append(q); chi_all.append(chi); i_all.append(inten)
                n_peaks += len(q)
            if done % 500 == 0:
                print(f'[bank] {done}/{len(names)} CIFs -> {len(starts)} entries', flush=True)

    if not starts:
        raise SystemExit('[FATAL] no entries produced')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        q=np.concatenate(q_all).astype(np.float32),
        chi=np.concatenate(chi_all).astype(np.float32),     # -1 marks powder rings
        intensity=np.concatenate(i_all).astype(np.float32),
        entry_start=np.asarray(starts, np.int64),
        entry_count=np.asarray(counts, np.int64),
        entry_kind=np.asarray(kinds),
        entry_cif=np.asarray(cifs),
    )
    n_pow = kinds.count('powder')
    manifest = dict(cif_dir=args.cif_dir, n_cifs=len(names), n_entries=len(starts),
                    n_powder=n_pow, n_oriented=len(starts) - n_pow, k_orient=args.k_orient,
                    orient_margin_deg=ORIENT_MARGIN_DEG, q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX,
                    top_peaks=TOP_PEAKS, exclusions_applied=not args.no_exclusions,
                    n_sim_errors=len(errors), entries=metas, errors=errors[:2000])
    with open(os.path.splitext(args.out)[0] + '_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'[bank] wrote {args.out}: {len(starts)} entries '
          f'({n_pow} powder, {len(starts) - n_pow} oriented), {n_peaks} peaks, '
          f'{len(errors)} sim errors')


if __name__ == '__main__':
    main()
