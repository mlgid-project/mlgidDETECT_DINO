"""Generate the physics peak bank for the CIF-based training simulation (Track B, step B2).

For every usable CIF (staged + screened by build_exclusion_list.py) this simulates, with pygidsim:
  * the POWDER pattern  (orientation=None)      -> rings:  (|q|, I)
  * K ORIENTED patterns (explicit random orientation vectors, recorded per entry)
                                                -> spots:  (|q|, chi_deg, I)
and stores everything geometry-independently in polar physics coordinates (|q| in A^-1, chi in
degrees from the q_xy axis). The training-time renderer (physics_simulation.py) maps them to
pixels per image via x = q/q_max*1024, y = chi/90*512 with a sampled q_max.

EXCLUSIONS (user mandate; from eval_matched.json produced by build_exclusion_list.py):
  * a CIF matched in eval as powder/rings -> its POWDER entry is dropped;
  * a CIF matched in eval with orientation v -> oriented entries within ORIENT_MARGIN_DEG of v
    are dropped (other orientations of the same structure remain usable, per user).
An audit section in the manifest records exactly what was dropped and why, plus name-collisions
with the known eval material list (logged for review, NOT auto-excluded beyond the above).

Output: <cif_library>/bank/bank.npz + bank_manifest.json.
Runs in the `mlgid_physics` env (CPU is fine). Re-runnable; keyed on the screened CIF set.

Usage:
  python physics_sim/generate_bank.py [--k-orient 8] [--limit N] [--no-exclusions]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_exclusion_list import (STAGING, CACHE_DIR, USER_CIFS, Q_XY_MAX, Q_Z_MAX,
                                  stage_cifs, screen_cifs)

OUT_NPZ = os.path.join(CACHE_DIR, "bank.npz")
OUT_MANIFEST = os.path.join(CACHE_DIR, "bank_manifest.json")
EXCL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_matched.json")

ORIENT_MARGIN_DEG = 10.0
TOP_PEAKS = 200          # keep at most this many peaks per entry (by intensity)
MIN_PEAKS = 3            # drop entries with fewer visible peaks
# eval material name fragments (consolidated eval list) -- logged in the audit on filename hits
EVAL_NAME_FRAGMENTS = [
    "4p", "quaterphenyl", "c60", "fulleren", "dbttf", "dip", "diindenoperylene", "hatcn",
    "ptcdi", "pdic", "perylene", "pentacene", "znpc", "phthalocyanine",
    "mapbi", "mapbbr", "fapbi", "fapbbr", "pbi2", "pbbr2", "cuscn",
    "ba2pbi4", "ba2ma", "ba2fa", "pea2", "pdma", "ada", "adam",
]


def to_cartesian(u, rec):
    """Map a crystal-basis (Miller) direction to Cartesian.

    CRITICAL: pygidsim/mlgidmatch interpret `orientation` in the CRYSTAL basis and map it as
    `orientation @ rec` (mlgidmatch.preprocess.rotate.rotate_vect). mlgidMATCH therefore reports
    matched orientations as Miller directions (e.g. [0, 4, 3]). Angles between such directions
    must be measured AFTER this map -- a dot product in Miller space equals the physical angle
    only for cubic cells, and most eval structures (2D perovskites) are not cubic. Getting this
    wrong would silently under-exclude matched eval orientations.
    """
    c = np.asarray(u, float) @ np.asarray(rec, float)
    return c / (np.linalg.norm(c) + 1e-12)


def ang_deg(u, v, rec):
    """Physical angle (deg) between two crystal-basis directions; axes are sign-invariant."""
    uc, vc = to_cartesian(u, rec), to_cartesian(v, rec)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(uc, vc))), -1.0, 1.0))))


def orientation_vectors(k, rng, rec):
    """k random fiber/texture axes as crystal-basis directions, physically well separated
    (separation measured in Cartesian space via `rec`, same reason as ang_deg)."""
    vs = []
    tries = 0
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


def load_exclusions():
    """-> (powder_excluded: set[cif], oriented_excluded: {cif: [orientation vectors]})"""
    if not os.path.isfile(EXCL_JSON):
        return set(), {}
    with open(EXCL_JSON) as f:
        d = json.load(f)
    powder, oriented = set(), {}
    for sample, rows in d.get("matches", {}).items():
        for r in rows:
            cif = r["cif"]
            if r.get("peaks_type") == "rings" or r.get("orientation") in (None, "powder"):
                powder.add(cif)
            o = r.get("orientation")
            if isinstance(o, (list, tuple)) and len(o) == 3:
                oriented.setdefault(cif, []).append([float(x) for x in o])
    return powder, oriented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-orient", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N CIFs (smoke)")
    ap.add_argument("--no-exclusions", action="store_true",
                    help="build without eval_matched.json (PRELIMINARY bank only)")
    args = ap.parse_args()

    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif

    names, set_hash = stage_cifs()
    valid = screen_cifs(names, set_hash)
    if args.limit:
        valid = valid[: args.limit]
    print(f"[bank] {len(valid)} valid CIFs (set {set_hash})")

    if not args.no_exclusions and not os.path.isfile(EXCL_JSON):
        print(f"[FATAL] {EXCL_JSON} not found -- run build_exclusion_list.py first "
              f"(or pass --no-exclusions for a PRELIMINARY bank).")
        sys.exit(2)
    powder_excl, orient_excl = load_exclusions()
    print(f"[bank] exclusions: {len(powder_excl)} powder-excluded CIFs, "
          f"{sum(len(v) for v in orient_excl.values())} oriented exclusions on {len(orient_excl)} CIFs")

    params = ExpParameters(q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX)
    rng = np.random.default_rng(0)
    q_all, chi_all, i_all = [], [], []
    entry_index = []          # (start, count) into flat arrays
    entry_meta = []           # dicts: cif, kind, orientation
    audit = dict(powder_dropped=[], orient_dropped=[], sim_errors=[], name_flags=[])

    for ci, name in enumerate(valid):
        path = os.path.join(STAGING, name)
        lname = name.lower()
        if any(f in lname for f in EVAL_NAME_FRAGMENTS):
            audit["name_flags"].append(name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                el = GIWAXSFromCif(path, params)
        except BaseException as e:
            audit["sim_errors"].append([name, "load: " + repr(e)[:80]])
            continue

        rec = np.asarray(el.giwaxs.rec, float)     # Miller -> Cartesian map for this cell

        # ---- powder ----
        if name in powder_excl:
            audit["powder_dropped"].append(name)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    q1d, i1d = el.giwaxs.giwaxs_sim(orientation=None)
                q1d = np.asarray(q1d, float); i1d = np.asarray(i1d, float)
                keep = (q1d > 0.05) & np.isfinite(i1d) & (i1d > 0)
                q1d, i1d = q1d[keep], i1d[keep]
                if len(q1d) >= MIN_PEAKS:
                    top = np.argsort(i1d)[::-1][:TOP_PEAKS]
                    entry_index.append((len(q_all), len(top)))
                    entry_meta.append(dict(cif=name, kind="powder", orientation=None))
                    q_all.extend(q1d[top]); chi_all.extend([-1.0] * len(top)); i_all.extend(i1d[top])
            except BaseException as e:
                audit["sim_errors"].append([name, "powder: " + repr(e)[:80]])

        # ---- oriented ----
        banned = orient_excl.get(name, [])
        for v in orientation_vectors(args.k_orient, rng, rec):
            if any(ang_deg(v, b, rec) < ORIENT_MARGIN_DEG for b in banned):
                audit["orient_dropped"].append([name, [round(float(x), 4) for x in v]])
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    q2d, i2d = el.giwaxs.giwaxs_sim(orientation=np.asarray(v))
                q2d = np.asarray(q2d, float); i2d = np.asarray(i2d, float)
                if q2d.ndim != 2 or q2d.shape[0] != 2 or q2d.shape[1] < MIN_PEAKS:
                    continue
                qxy, qz = np.abs(q2d[0]), np.abs(q2d[1])
                qabs = np.hypot(qxy, qz)
                chi = np.degrees(np.arctan2(qz, qxy))          # 0 = in-plane (q_xy axis)
                keep = (qabs > 0.05) & np.isfinite(i2d) & (i2d > 0)
                qabs, chi, ii = qabs[keep], chi[keep], i2d[keep]
                if len(qabs) < MIN_PEAKS:
                    continue
                top = np.argsort(ii)[::-1][:TOP_PEAKS]
                entry_index.append((len(q_all), len(top)))
                entry_meta.append(dict(cif=name, kind="oriented",
                                       orientation=[round(float(x), 5) for x in v]))
                q_all.extend(qabs[top]); chi_all.extend(chi[top]); i_all.extend(ii[top])
            except BaseException as e:
                audit["sim_errors"].append([name, "orient: " + repr(e)[:80]])

        if (ci + 1) % 200 == 0:
            print(f"[bank] {ci+1}/{len(valid)} CIFs -> {len(entry_meta)} entries")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        q=np.asarray(q_all, np.float32),
        chi=np.asarray(chi_all, np.float32),          # -1 marks powder rings
        intensity=np.asarray(i_all, np.float32),
        entry_start=np.asarray([s for s, _ in entry_index], np.int64),
        entry_count=np.asarray([c for _, c in entry_index], np.int64),
        entry_kind=np.asarray([m["kind"] for m in entry_meta]),
        entry_cif=np.asarray([m["cif"] for m in entry_meta]),
    )
    n_pow = sum(1 for m in entry_meta if m["kind"] == "powder")
    manifest = dict(cif_set_hash=set_hash, n_cifs=len(valid), n_entries=len(entry_meta),
                    n_powder=n_pow, n_oriented=len(entry_meta) - n_pow,
                    k_orient=args.k_orient, orient_margin_deg=ORIENT_MARGIN_DEG,
                    q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX, top_peaks=TOP_PEAKS,
                    exclusions_applied=not args.no_exclusions,
                    n_powder_dropped=len(audit["powder_dropped"]),
                    n_orient_dropped=len(audit["orient_dropped"]),
                    n_sim_errors=len(audit["sim_errors"]),
                    n_name_flags=len(audit["name_flags"]),
                    entries=entry_meta, audit=audit)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[bank] wrote {OUT_NPZ}: {len(entry_meta)} entries "
          f"({n_pow} powder, {len(entry_meta) - n_pow} oriented), "
          f"{len(q_all)} peaks total")
    print(f"[bank] audit: {len(audit['powder_dropped'])} powder dropped, "
          f"{len(audit['orient_dropped'])} orientations dropped, "
          f"{len(audit['sim_errors'])} sim errors, {len(audit['name_flags'])} name flags")
    print(f"[bank] manifest -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
