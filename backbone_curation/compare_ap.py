"""
Compare detector ap_total curves between runs (e.g. SSL-backbone vs from-scratch baseline).
Reads the per-epoch eval files `exp_ap_41.txt` / `exp_ap_organic.txt` (format: 'epoch<TAB>ap')
that main.py writes in each run dir.

Usage:
  python compare_ap.py <baseline_run_dir> [<ssl_run_dir>]
  - one dir  : print that run's curve + best/final
  - two dirs : side-by-side at matched epochs with deltas
"""
import sys, os


def load_ap(run_dir, name):
    p = os.path.join(run_dir, f"exp_ap_{name}.txt")
    d = {}
    if os.path.exists(p):
        for line in open(p):
            t = line.split()
            if len(t) == 2:
                try:
                    d[int(float(t[0]))] = float(t[1])
                except ValueError:
                    pass
    return d


def summary(d):
    if not d:
        return "(no data)"
    be = max(d, key=lambda e: d[e]); fe = max(d)
    return f"best {d[be]:.3f}@ep{be}  final {d[fe]:.3f}@ep{fe}"


def main():
    base = sys.argv[1]
    ssl = sys.argv[2] if len(sys.argv) > 2 else None
    for name in ["41", "organic"]:
        b = load_ap(base, name)
        s = load_ap(ssl, name) if ssl else {}
        print(f"\n===== {name}  ap_total =====")
        print(f"  baseline: {summary(b)}")
        if ssl:
            print(f"  ssl     : {summary(s)}")
        if ssl and s:
            print(f"\n  {'epoch':>5} {'baseline':>9} {'ssl':>9} {'delta':>8}")
            for e in sorted(s):                       # matched at the SSL run's evaluated epochs
                bb = b.get(e)
                bs = f"{bb:.3f}" if bb is not None else "  -  "
                dd = f"{s[e]-bb:+.3f}" if bb is not None else "   -  "
                print(f"  {e:5d} {bs:>9} {s[e]:9.3f} {dd:>8}")
        else:
            # baseline-only reference at a sampling of epochs
            es = sorted(b)
            pick = [e for e in [0,10,20,30,40,50,75,100,150,200,250,300,350] if e in b]
            if es and es[-1] not in pick: pick.append(es[-1])
            print(f"  {'epoch':>5} {'baseline ap':>12}")
            for e in pick:
                print(f"  {e:5d} {b[e]:12.3f}")


if __name__ == "__main__":
    main()
