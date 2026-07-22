"""Build the eval-exclusion list for the physics-based CIF simulation (Track B, step B1).

MANDATE (user): the physics simulation must NOT train on the specific oriented structures present
in the labeled evaluation datasets. Run mlgidMATCH over the eval sets with a LOWERED probability
threshold ("really find all candidates"), and record every candidate (CIF, orientation,
probability) per eval sample. Bank generation (generate_bank.py) then excludes:
  * matched powder phases -> that structure's powder mode entirely;
  * matched oriented phases -> (structure, orientation) pairs within an angular margin
    (other orientations of the same structure stay usable, per user).

Runs in the `mlgid_physics` conda env (pygidsim + mlgidmatch), CPU-only (cpu-galvani): the
dominant cost is mlgidmatch's per-orientation numpy simulation, not the small NN screen.
Re-runnable: staging + caches keyed on the CIF file set, so re-run after dropping more CIFs.

TRACTABILITY (measured 2026-07-21: uncapped match_all on ONE ~20-peak measurement at thr 0.10
took >2 h single-core -- infeasible for ~90 eval samples). Restructured:
  * SEGMENTS: library Match.match_all with audited caps (CappedMatch): top-K NN-screen
    candidates enter orientation matching per tree node; multi-phase recursion only into the
    top-B branches per node. Cap events are counted and reported (never silent). A full
    UNCAPPED NN screen per sample is stored in the output as an audit/belt-and-braces layer.
  * Parallelism: Slurm ARRAY SHARDS (--shard/--nshards), independent processes. A fork
    pool was tried and DEADLOCKED (torch/OpenMP + fork; job 2679842 burned 8 h, zero
    units done) -- do not reintroduce multiprocessing here.
  * Single threshold 0.05 (a 0.05 screen admits a superset of 0.10 -- the old two-pass union
    was redundant work; per-row NN probability is recorded anyway).
  * RINGS: the library rings path needs precomputed 1-D patterns (create_all=True) we don't
    have, and powder matching needs no orientation search -- done directly here: two-sided 1-D
    q-coverage (observed rings covered by sim + sim's strong in-window rings observed).

Pipeline:
  1. Stage a combined CIF folder (symlinks): user library + COD perovskite selection.
  2. CifPattern-preprocess (cached pickle keyed on the sorted file list hash).
  3. SELF-TEST (segments + rings): simulate peaks for known CIFs with pygidsim, require
     self-recovery through the same matching paths -- validates the (q_xy, q_z) conventions
     end-to-end before any eval matching is trusted.
  4. Extract labeled peaks from the eval sets:
       41.h5 (roi_data; 45.h5 excluded per user): q[A^-1] = radius_px*qz_max/image_shape0 (util/labeleddataset
       convention); chi = angle degrees from the q_xy axis -> (q_xy, q_z) = q*(cos chi, sin chi);
       type==1 -> ring (|q| only). Intensity = 'peak height'.
       organic_labeled.h5 (pygid): fitted_peaks radius already in A^-1 (util/pygidloader
       convention); is_ring field present.
  5. Write eval_matched.json {sample: [{cif, orientation, probability, ...}]} + screen audit
     + cap statistics.

Usage (sbatch, mlgid_physics env):
  python physics_sim/build_exclusion_list.py [--selftest-only]
                                             [--shard i --nshards N]  (Slurm array)
"""
import argparse
import hashlib
import re
import json
import os
import pickle
import sys

import numpy as np

USER_CIFS = "/mnt/lustre/work/schreiber/szb389/datasets/cif_library"
COD_CIFS = "/mnt/lustre/work/schreiber/szb082/CIFs/CIFs_cod_selection_perovskite"
STAGING = os.path.join(USER_CIFS, "_combined")
CACHE_DIR = os.path.join(USER_CIFS, "bank")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_matched.json")

# 45.h5 deliberately EXCLUDED: user 2026-07-20 "you can ignore 45, use the 41 set as has been
# done so far" -- the standing eval gates are organic + 41, so only those need exclusion.
EVAL_ROI = ["/mnt/lustre/work/schreiber/szb389/datasets/41.h5"]
EVAL_PYGID = ["/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"]

Q_XY_MAX = 3.0
Q_Z_MAX = 3.0
SEG_THRESHOLD = 0.05   # lowered NN threshold (user: "really find all candidates"); superset of 0.10
TOP_K = 64             # cap: NN-screen candidates entering orientation matching per tree node
BRANCH_CAP = 6         # cap: residual-phase recursions per tree node
RING_DQ = 0.05         # A^-1 tolerance for 1-D powder ring matching
RING_MIN_MATCHED = 3   # rings pass requires >=3 observed rings explained ...
RING_MIN_FRAC = 0.5    # ... and >=50% two-sided coverage


def patch_pygidsim_compat():
    """mlgidmatch 0.1.3 orient_experiment_match still calls GIWAXS.giwaxs_2d(q_range=...) but
    pygidsim 0.1.5 renamed that to q_xy_range/q_z_range (cif_preprocess.py is already migrated).
    q_range=(q_xy_max, q_z_max) elementwise upper bounds == ExpParameters(q_xy_max, q_z_max)
    semantics ((0, max) each), so the translation is exact."""
    from pygidsim.giwaxs_sim import GIWAXS
    orig = GIWAXS.giwaxs_2d

    def compat(*args, q_range=None, q_xy_range=None, q_z_range=None, **kw):
        if q_range is not None:
            q_xy_range = (0.0, float(q_range[0]))
            q_z_range = (0.0, float(q_range[1]))
        return orig(*args, q_xy_range=q_xy_range, q_z_range=q_z_range, **kw)

    GIWAXS.giwaxs_2d = staticmethod(compat)


def make_capped_match(cp, device):
    """Match subclass with audited tractability caps (see module docstring)."""
    from mlgidmatch.matching import Match

    class CappedMatch(Match):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.cap_events = 0
            self.branch_cap_events = 0

        def screen_uncapped(self, peaks, q_range):
            return Match.match_cifs(self, peaks, q_range, None)

        def match_cifs(self, peaks, q_range, candidates=None):
            probs = super().match_cifs(peaks, q_range, candidates)
            if int(np.sum(probs >= SEG_THRESHOLD)) > TOP_K:
                kth = np.sort(probs)[-TOP_K]
                probs = np.where(probs >= kth, probs, 0.0)
                self.cap_events += 1
            return probs

        def _build_tree(self, peaks_all, intens_real_all, q_range, peaks_indices, candidates,
                        threshold, save_metrics, depth):
            # library logic (mlgidmatch/matching.py) + branch cap on the recursion
            if depth >= 3 or len(peaks_indices) <= 3:
                return {}
            probs = self.match_cifs(peaks=peaks_all[peaks_indices], q_range=q_range,
                                    candidates=candidates)
            if np.sum(probs >= threshold) == 0:
                return {}
            if self.peaks_type == 'rings':
                peaks_input = np.linalg.norm(peaks_all, axis=-1)
            elif self.peaks_type == 'segments':
                peaks_input = peaks_all
            else:
                raise ValueError("peaks_type should be either 'rings' or 'segments'")
            data_matched = self.match_peaks(
                peaks_all=peaks_input, intens_real_all=intens_real_all, probs=probs,
                q_range=q_range, peaks_indices=peaks_indices, candidates=candidates,
                threshold=threshold, save_metrics=save_metrics)
            if not data_matched:
                return {}
            order = sorted(data_matched,
                           key=lambda k: -float(data_matched[k].get('probability') or 0.0))
            recurse = set(order[:BRANCH_CAP])
            if len(order) > BRANCH_CAP:
                self.branch_cap_events += 1
            for key, branch in data_matched.items():
                if key not in recurse or len(branch['indices_real_matched']) == 0:
                    continue
                mask = np.zeros(len(peaks_all), dtype=bool)
                mask[peaks_indices] = True
                mask[branch['indices_real_matched']] = False
                new_idx = np.arange(len(peaks_all))[mask]
                branch.update(self._build_tree(peaks_all, intens_real_all, q_range, new_idx,
                                               candidates, threshold, save_metrics, depth + 1))
            return data_matched

    return CappedMatch(cp, device=device)


# ---------------- segments unit ----------------
# NOTE: run SERIALLY within a process. An earlier multiprocessing.Pool(fork) version deadlocked
# (job 2679842: 8 h, zero units completed) -- forking a parent that has torch/OpenMP initialized
# is unsafe. Parallelism now comes from Slurm array shards (--shard/--nshards), i.e. independent
# processes, which is also restartable.


def seg_unit(name, samples, match):
    import time
    d = samples[name]
    match.cap_events = 0
    match.branch_cap_events = 0
    res = dict(unit=name, rows=[], screen=[], sec=0.0, cap_events=0,
               branch_cap_events=0, error=None)
    t0 = time.time()
    try:
        p = np.asarray(d["seg_qxyqz"], np.float32)
        i = np.asarray(d["seg_int"], np.float32)
        qr = tuple(d["q_range"])
        probs = match.screen_uncapped(p, qr)                       # full audit screen
        cifs = list(match.config.cif_prepr.cifs)
        res["screen"] = [dict(cif=str(cifs[j]), probability=round(float(probs[j]), 4))
                         for j in np.nonzero(probs >= SEG_THRESHOLD)[0]]
        raw = match.match_all([name], [p], [i], [qr],
                              peaks_type="segments", threshold=SEG_THRESHOLD)
        res["rows"] = flatten_solutions(raw, SEG_THRESHOLD, "segments")
    except Exception as e:
        res["error"] = repr(e)[:300]
    res["sec"] = round(time.time() - t0, 1)
    res["cap_events"] = match.cap_events
    res["branch_cap_events"] = match.branch_cap_events
    return res


def run_units(units, samples, match):
    out = []
    for k, name in enumerate(units):
        res = seg_unit(name, samples, match)
        out.append(res)
        print(f"[unit] {k + 1}/{len(units)} {res['unit']}: rows={len(res['rows'])} "
              f"screen={len(res['screen'])} caps={res['cap_events']}/{res['branch_cap_events']} "
              f"{res['sec']}s" + (f"  ERROR {res['error']}" if res["error"] else ""),
              flush=True)
    return out


# ---------------- rings (1-D powder coverage, no orientation search) ----------------

def powder_q_from_cp(cp):
    """Per-CIF (sorted |q|, intensities sorted alike) from the preprocessed 3-D patterns --
    the same peak source the segments matcher and the bank generator use."""
    out = []
    for q3d, inten in zip(cp.pattern_3d.q_3d, cp.pattern_3d.intensities):
        q = np.linalg.norm(np.asarray(q3d, np.float32), axis=-1)
        inten = np.asarray(inten, np.float32)
        keep = q > 1e-4
        q, inten = q[keep], inten[keep]
        order = np.argsort(q)
        out.append((q[order], inten[order]))
    return out


def match_rings_sample(name, d, powder_q, cifs):
    """Two-sided coverage: (a) observed rings explained by sim rings within RING_DQ;
    (b) the sim's strongest in-window rings actually observed (kills trivially-dense
    powders that would 'cover' anything). probability = min(frac_a, frac_b)."""
    obs = np.asarray(d["ring_q"], np.float32)
    obs = np.sort(obs[obs > 1e-3])
    rows = []
    if len(obs) < RING_MIN_MATCHED:
        return rows
    lo, hi = obs[0] - 0.1, obs[-1] + 0.1
    for i, (qs, ints) in enumerate(powder_q):
        if len(qs) < 2:
            continue
        j = np.clip(np.searchsorted(qs, obs), 1, len(qs) - 1)
        dmin = np.minimum(np.abs(obs - qs[j - 1]), np.abs(obs - qs[j]))
        matched = int(np.sum(dmin <= RING_DQ))
        frac_a = matched / len(obs)
        if matched < RING_MIN_MATCHED or frac_a < RING_MIN_FRAC:
            continue
        w = np.nonzero((qs >= lo) & (qs <= hi))[0]
        if len(w) == 0:
            continue
        top = w[np.argsort(ints[w])[::-1][:10]]
        k = np.clip(np.searchsorted(obs, qs[top]), 1, len(obs) - 1)
        d2 = np.minimum(np.abs(qs[top] - obs[k - 1]), np.abs(qs[top] - obs[k]))
        frac_b = float(np.mean(d2 <= RING_DQ))
        prob = min(frac_a, frac_b)
        if prob >= RING_MIN_FRAC:
            rows.append(dict(sample=name, cif=str(cifs[i]), orientation=None,
                             probability=round(prob, 4), peaks_type="rings",
                             threshold=RING_MIN_FRAC, depth=0))
    return rows


def stage_cifs():
    """Symlink user + COD CIFs into one folder; return sorted file names + content hash."""
    os.makedirs(STAGING, exist_ok=True)
    names = []
    for src_dir, prefix in [(USER_CIFS, "user__"), (COD_CIFS, "cod__")]:
        if not os.path.isdir(src_dir):
            continue
        for f in sorted(os.listdir(src_dir)):
            if not f.lower().endswith(".cif"):
                continue
            link = f"{prefix}{f}"
            dst = os.path.join(STAGING, link)
            if not os.path.islink(dst):
                os.symlink(os.path.join(src_dir, f), dst)
            names.append(link)
    names = sorted(set(names))
    h = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
    return names, h


def screen_cifs(names, set_hash):
    """Keep only CIFs the physics engine can actually simulate (COD files are heterogeneous:
    unknown elements like D, unparsable sections, bad occupancies). Ground truth = run the same
    code path as preprocessing (GIWAXSFromCif + powder sim) per file; cache the survivor list."""
    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"valid_cifs_{set_hash}.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            d = json.load(f)
        print(f"[screen] cached: {len(d['valid'])}/{len(names)} CIFs valid ({len(d['rejected'])} rejected)")
        return d["valid"]
    params = ExpParameters(q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX)
    valid, rejected = [], []
    import warnings
    for i, n in enumerate(names):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                el = GIWAXSFromCif(os.path.join(STAGING, n), params)
                q1d, i1d = el.giwaxs.giwaxs_sim(orientation=None)
            if len(np.asarray(q1d)) == 0:
                raise ValueError("no peaks in q range")
            valid.append(n)
        except BaseException as e:                       # some parsers raise SystemExit
            rejected.append([n, repr(e)[:120]])
        if (i + 1) % 250 == 0:
            print(f"[screen] {i+1}/{len(names)}  valid={len(valid)}")
    with open(cache, "w") as f:
        json.dump(dict(valid=valid, rejected=rejected), f, indent=1)
    print(f"[screen] {len(valid)}/{len(names)} CIFs valid; rejected list cached to {cache}")
    return valid


def build_cif_pattern(names, set_hash):
    from pygidsim.experiment import ExpParameters
    from mlgidmatch.preprocess.cif_preprocess import CifPattern
    os.makedirs(CACHE_DIR, exist_ok=True)
    names = screen_cifs(names, set_hash)
    vhash = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE_DIR, f"cif_prepr_{vhash}.pkl")
    params = ExpParameters(q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX)
    if os.path.isfile(cache):
        print(f"[cif] loading cached preprocessing {cache}")
        cp = CifPattern(params, STAGING, cifs=names, preprocessed_3d=cache)
    else:
        print(f"[cif] preprocessing {len(names)} valid CIFs (one-time; cached to {cache})")
        cp = CifPattern(params, STAGING, cifs=names)
        with open(cache, "wb") as f:
            pickle.dump(cp, f)
    return cp


# ---------------- eval peak extraction ----------------

def _arr(dset, dtype=float):
    """h5py cannot always build a direct-read conversion path for these files
    (TypeError: Operation not defined for data type class) -- materialize with [()] first."""
    return np.asarray(dset[()], dtype=dtype)


def _safe_qrange(qxy_max, qz_max):
    """One 41.h5 sample carries a degenerate (0, 0) range; matching would divide by it. Fall
    back to the library q window in that case."""
    qxy = qxy_max if qxy_max and qxy_max > 0.1 else Q_XY_MAX
    qz = qz_max if qz_max and qz_max > 0.1 else Q_Z_MAX
    return (min(Q_XY_MAX, qxy), min(Q_Z_MAX, qz))


def _q_range4(dset):
    """qz_qxy_range_[A-1] is heterogeneous across eval samples: usually a 4-float array, but
    sometimes a single (byte)string like b'(0, 3.2, 0, 3.2)'. Mirrors the deployed loader's
    workaround (util/labeleddataset.py:170-174) so eval geometry is read identically."""
    v = dset[()]
    vals = [v] if np.ndim(v) == 0 else list(v)
    if len(vals) == 1 or isinstance(vals[0], (bytes, str, np.bytes_, np.str_)):
        nums = re.findall(r"[-+]?(?:\d*\.*\d+)", str(vals[0] if len(vals) == 1 else vals))
        return np.array([float(x) for x in nums], dtype=float)
    return np.asarray(vals, dtype=float)


def peaks_from_roi_h5(path):
    """roi_data eval file -> {sample: dict(seg_qxyqz, seg_int, ring_q, ring_int)}."""
    import h5py
    out = {}
    with h5py.File(path, "r") as h:
        def walk(g, p=""):
            for k in g:
                obj = g[k]
                pp = f"{p}/{k}" if p else k
                if isinstance(obj, h5py.Group):
                    if "roi_data" in obj and "image" in obj:
                        yield pp, obj
                    else:
                        yield from walk(obj, pp)
        for name, grp in walk(h):
            rd = grp["roi_data"]
            radius_px = _arr(rd["radius"])
            angle = _arr(rd["angle"])                                 # deg from q_xy axis
            typ = _arr(rd["type"], int)                               # 1 = ring
            inten = _arr(rd["peak height"])
            shape0 = grp["image"].shape[0]
            qr = _q_range4(grp["metadata/qz_qxy_range_[A-1]"])
            qz_max = float(qr[1])                                     # [qz_min, qz_max, qxy_min, qxy_max]
            q = radius_px * qz_max / shape0                           # A^-1 (labeleddataset convention)
            chi = np.deg2rad(np.clip(angle, 0.0, 90.0))
            is_ring = typ == 1
            seg = ~is_ring
            out[f"{os.path.basename(path)}::{name}"] = dict(
                seg_qxyqz=np.stack([q[seg] * np.cos(chi[seg]), q[seg] * np.sin(chi[seg])], axis=-1),
                seg_int=inten[seg],
                ring_q=q[is_ring], ring_int=inten[is_ring],
                q_range=_safe_qrange(float(qr[3]) if len(qr) > 3 else 0.0, qz_max),
            )
    return out


def peaks_from_pygid_h5(path):
    """pygid eval file -> same dict structure; fitted_peaks radius already in A^-1."""
    import h5py
    out = {}
    with h5py.File(path, "r") as h:
        for entry in h:
            g = h[entry]
            if "data" not in g:
                continue
            try:
                qz_max = float(_arr(g["data/q_z"])[-1])
                qxy_max = float(_arr(g["data/q_xy"])[-1])
            except Exception:
                qz_max = qxy_max = 3.0
            ana = g.get("data/analysis")
            if ana is None:
                continue
            for frame in ana:
                fp = ana[frame].get("fitted_peaks")
                if fp is None or fp.dtype.names is None:
                    continue
                rec = fp[()]                      # COMPOUND dataset: structured array, not a group
                f = rec.dtype.names
                q = np.asarray(rec["radius"], float)                  # already A^-1 (pygidloader)
                chi = np.deg2rad(np.clip(np.asarray(rec["angle"], float), 0.0, 90.0))
                inten = np.ones_like(q)
                for cand in ("amplitude", "score"):                   # both can be all-zero
                    if cand in f and np.any(np.asarray(rec[cand], float) > 0):
                        inten = np.asarray(rec[cand], float)
                        break
                is_ring = (np.asarray(rec["is_ring"]).astype(bool) if "is_ring" in f
                           else np.zeros(len(q), bool))
                seg = ~is_ring
                out[f"{os.path.basename(path)}::{entry}/{frame}"] = dict(
                    seg_qxyqz=np.stack([q[seg] * np.cos(chi[seg]), q[seg] * np.sin(chi[seg])], axis=-1),
                    seg_int=inten[seg],
                    ring_q=q[is_ring], ring_int=inten[is_ring],
                    q_range=_safe_qrange(qxy_max, qz_max),
                )
    return out


# ---------------- matching output ----------------

def flatten_solutions(raw, threshold, peaks_type):
    """Recursively walk the match_all tree: branch dicts carry 'cif'/'orient'/'probability';
    residual-phase children are nested inside their parent branch under integer-string keys."""
    rows = []

    def walk(node, sample, depth):
        for v in node.values():
            if isinstance(v, dict) and "cif" in v:
                rows.append(dict(sample=sample, cif=str(v["cif"]),
                                 orientation=_jsonable(v.get("orient")),
                                 probability=_jsonable(v.get("probability")),
                                 peaks_type=peaks_type, threshold=threshold, depth=depth))
                walk(v, sample, depth + 1)

    if isinstance(raw, dict):
        for sample, tree in raw.items():
            if isinstance(tree, dict):
                walk(tree, sample, 0)
    return rows


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


# ---------------- self-test ----------------

def self_test(match, cp, powder_q):
    """Simulate peaks from known library CIFs with pygidsim; require self-recovery through the
    SAME code paths eval matching uses (capped+pooled segments; 1-D coverage rings)."""
    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif
    rng = np.random.default_rng(0)
    params = ExpParameters(q_xy_max=Q_XY_MAX, q_z_max=Q_Z_MAX)
    names = list(cp.cifs)
    picks = [names[i] for i in rng.choice(len(names), size=min(3, len(names)), replace=False)]

    st_samples = {}
    for cif_name in picks:
        try:
            el = GIWAXSFromCif(os.path.join(STAGING, cif_name), params)
            q2d, inten = el.giwaxs.giwaxs_sim(orientation="random")
            q2d = np.asarray(q2d); inten = np.asarray(inten)
            peaks = np.stack([q2d[0], q2d[1]], axis=-1)
            keep = np.argsort(inten)[::-1][:min(len(peaks), 40)]
            st_samples[f"selftest::{cif_name}"] = dict(
                seg_qxyqz=peaks[keep], seg_int=inten[keep],
                ring_q=np.array([]), ring_int=np.array([]),
                q_range=(Q_XY_MAX, Q_Z_MAX))
        except Exception as e:
            print(f"[selftest] {cif_name}: SIM ERROR {repr(e)[:200]}")
    ok_seg = 0
    for res in run_units(list(st_samples), st_samples, match):
        cif_name = res["unit"].split("::", 1)[1]
        hit = any(cif_name in r["cif"] or r["cif"] in cif_name for r in res["rows"])
        print(f"[selftest/seg] {cif_name}: {'RECOVERED' if hit else 'NOT RECOVERED'} "
              f"({len(res['rows'])} rows, {res['sec']}s)"
              + (f"  ERROR {res['error']}" if res["error"] else ""))
        ok_seg += int(hit)

    ring_picks = [names[i] for i in rng.choice(len(names), size=min(3, len(names)), replace=False)]
    ok_ring = 0
    for cif_name in ring_picks:
        try:
            el = GIWAXSFromCif(os.path.join(STAGING, cif_name), params)
            q1d, i1d = el.giwaxs.giwaxs_sim(orientation=None)
            q1d = np.asarray(q1d, np.float32); i1d = np.asarray(i1d, np.float32)
            keep = np.argsort(i1d)[::-1][:min(len(q1d), 20)]
            rows = match_rings_sample(f"ringtest::{cif_name}", dict(ring_q=q1d[keep]),
                                      powder_q, names)
            hit = any(r["cif"] == cif_name and r["probability"] >= 0.7 for r in rows)
            print(f"[selftest/ring] {cif_name}: {'RECOVERED' if hit else 'NOT RECOVERED'} "
                  f"({len(rows)} powder candidates)")
            ok_ring += int(hit)
        except Exception as e:
            print(f"[selftest/ring] {cif_name}: ERROR {repr(e)[:200]}")

    print(f"[selftest] segments {ok_seg}/{len(st_samples)}; rings {ok_ring}/{len(ring_picks)}")
    return ok_seg, ok_ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true",
                    help="ONLY for re-running a shard whose self-test already passed with the "
                         "same CIF set + code; recorded in the shard JSON and surfaced by "
                         "merge_shards.py so a skipped gate is never invisible.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    patch_pygidsim_compat()
    import torch
    torch.set_num_threads(4)
    names, set_hash = stage_cifs()
    print(f"[stage] {len(names)} CIFs staged (set hash {set_hash})")
    cp = build_cif_pattern(names, set_hash)
    match = make_capped_match(cp, device="cpu")
    powder_q = powder_q_from_cp(cp)
    print(f"[match] CappedMatch ready on cpu (top_k={TOP_K}, branch_cap={BRANCH_CAP}, "
          f"shard {args.shard + 1}/{args.nshards})")

    if args.skip_selftest:
        print("[selftest] SKIPPED by --skip-selftest (re-run of an already-validated shard)")
        ok_seg = ok_ring = -1
    else:
        ok_seg, ok_ring = self_test(match, cp, powder_q)
    if args.selftest_only:
        return
    if ok_seg == 0 or ok_ring == 0:
        print(f"[FATAL] self-test failed (segments {ok_seg}, rings {ok_ring}) -- "
              f"conventions wrong; not matching eval.")
        sys.exit(2)

    samples = {}
    for p in EVAL_ROI:
        samples.update(peaks_from_roi_h5(p))
    for p in EVAL_PYGID:
        samples.update(peaks_from_pygid_h5(p))
    print(f"[eval] {len(samples)} labeled eval samples")

    # shard deterministically over the SORTED sample list so array tasks partition exactly
    ordered = sorted(samples)
    mine = [n for k, n in enumerate(ordered) if k % args.nshards == args.shard]
    print(f"[shard] {len(mine)}/{len(ordered)} samples in shard {args.shard + 1}/{args.nshards}")

    cif_names = list(cp.cifs)
    all_rows = []
    for name in mine:
        all_rows.extend(match_rings_sample(name, samples[name], powder_q, cif_names))
    print(f"[rings] {len(all_rows)} powder candidate rows across {len(mine)} samples")

    seg_units = [n for n in mine if len(samples[n]["seg_qxyqz"]) >= 4]
    print(f"[segments] {len(seg_units)} measurements (serial)")
    results = run_units(seg_units, samples, match)
    screen_audit = {}
    n_err = cap_ev = bcap_ev = 0
    for res in results:
        all_rows.extend(res["rows"])
        screen_audit[res["unit"]] = res["screen"]
        cap_ev += res["cap_events"]; bcap_ev += res["branch_cap_events"]
        n_err += int(res["error"] is not None)

    # union by (sample, cif, orientation-string); keep max probability
    merged = {}
    for r in all_rows:
        key = (r["sample"], r["cif"], str(r["orientation"]))
        if key not in merged or (r["probability"] or 0) > (merged[key]["probability"] or 0):
            merged[key] = r
    result = {}
    for r in merged.values():
        result.setdefault(r["sample"], []).append(
            {k: r[k] for k in ("cif", "orientation", "probability", "peaks_type", "threshold")})
    out_path = OUT_JSON if args.nshards == 1 else \
        OUT_JSON.replace(".json", f".shard{args.shard:02d}of{args.nshards:02d}.json")
    with open(out_path, "w") as f:
        json.dump(dict(cif_set_hash=set_hash, seg_threshold=SEG_THRESHOLD,
                       shard=[args.shard, args.nshards],
                       selftest_skipped=bool(args.skip_selftest),
                       caps=dict(top_k=TOP_K, branch_cap=BRANCH_CAP,
                                 screen_cap_events=cap_ev, branch_cap_events=bcap_ev),
                       ring_rule=dict(dq=RING_DQ, min_matched=RING_MIN_MATCHED,
                                      min_frac=RING_MIN_FRAC),
                       n_samples=len(mine), n_unit_errors=n_err,
                       matches=result, screen_audit=screen_audit), f, indent=1)
    ncif = len({r["cif"] for r in merged.values()})
    print(f"[out] {out_path}: {sum(len(v) for v in result.values())} matches, "
          f"{ncif} distinct CIFs across {len(result)} samples; "
          f"cap events screen={cap_ev} branch={bcap_ev}; unit errors={n_err}")
    if n_err:
        print("[WARN] some measurements errored -- their matches are MISSING; inspect before "
              "trusting the exclusion list.")
        sys.exit(3)


if __name__ == "__main__":
    main()
