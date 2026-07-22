"""Merge eval_matched.shardXXofYY.json files (Slurm array output) into eval_matched.json.

Refuses to merge an incomplete set -- a missing shard would silently drop eval samples from the
exclusion list, i.e. train the physics sim on structures we promised to exclude.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_matched.json")


def main():
    parts = sorted(glob.glob(os.path.join(HERE, "eval_matched.shard*of*.json")))
    if not parts:
        print("[FATAL] no shard files found"); sys.exit(2)
    nshards = json.load(open(parts[0]))["shard"][1]
    have = set()
    matches, screen, caps = {}, {}, dict(screen_cap_events=0, branch_cap_events=0)
    n_samples = n_err = 0
    skipped = []
    set_hash = None
    for p in parts:
        d = json.load(open(p))
        have.add(d["shard"][0])
        matches.update(d["matches"])
        screen.update(d.get("screen_audit", {}))
        caps["screen_cap_events"] += d["caps"]["screen_cap_events"]
        caps["branch_cap_events"] += d["caps"]["branch_cap_events"]
        n_samples += d["n_samples"]
        n_err += d.get("n_unit_errors", 0)
        set_hash = set_hash or d["cif_set_hash"]
        if d.get("selftest_skipped"):
            skipped.append(d["shard"][0])
    missing = set(range(nshards)) - have
    if missing:
        print(f"[FATAL] missing shards {sorted(missing)} of {nshards} -- refusing to write an "
              f"incomplete exclusion list (would leak eval structures into the bank).")
        sys.exit(2)

    ref = json.load(open(parts[0]))
    with open(OUT, "w") as f:
        json.dump(dict(cif_set_hash=set_hash, seg_threshold=ref["seg_threshold"],
                       caps=dict(ref["caps"], **caps), ring_rule=ref["ring_rule"],
                       n_shards=nshards, n_samples=n_samples, n_unit_errors=n_err,
                       matches=matches, screen_audit=screen), f, indent=1)
    if skipped:
        print(f"[note] shards {sorted(skipped)} ran with --skip-selftest (self-test had already "
              f"passed for this CIF set + code).")
    ncif = len({r["cif"] for rows in matches.values() for r in rows})
    nrow = sum(len(v) for v in matches.values())
    print(f"[merge] {len(parts)} shards, {n_samples} samples -> {OUT}")
    print(f"[merge] {nrow} matches, {ncif} distinct CIFs across {len(matches)} matched samples; "
          f"caps {caps}; unit errors {n_err}")
    if n_err:
        print("[WARN] shard(s) reported unit errors -- inspect before trusting the list.")


if __name__ == "__main__":
    main()
