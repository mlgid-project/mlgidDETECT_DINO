"""
Focused comparison figure: from-scratch baseline vs the round-1 SSL-backbone detector
(dino_ssl1), on both labeled eval sets (41 + organic). Two panels, full AP curves with
best-epoch markers + matched-epoch deltas in the title.

Baseline AP: exp_ap_{41,organic}.txt in the baseline run dir.
SSL AP: parsed from the newest dino_ssl-*.out ('[epoch N] {41,organic} ap_total = X').

Usage:  python plot_compare.py   ->  figures/baseline_vs_ssl1.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE     = os.path.dirname(__file__)
DET_OUT  = os.path.join(HERE, "ssl")
BASELINE = "/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434"
OUT      = os.path.join(HERE, "figures", "baseline_vs_ssl1.png")


def load_det(out_dir):
    outs = [f for f in os.listdir(out_dir) if f.startswith("dino_ssl-") and f.endswith(".out")]
    d = {"41": {}, "organic": {}}
    if not outs:
        return d
    newest = max(outs, key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    pat = re.compile(r"\[epoch (\d+)\]\s+(41|organic)\s+ap_total\s*=\s*([0-9.]+)")
    for line in open(os.path.join(out_dir, newest)):
        m = pat.search(line)
        if m:
            d[m.group(2)][int(m.group(1))] = float(m.group(3))
    return d


def load_base(run_dir, name):
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


def sortxy(d):
    xs = sorted(d)
    return xs, [d[x] for x in xs]


det = load_det(DET_OUT)
base = {n: load_base(BASELINE, n) for n in ("41", "organic")}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
for ax, name, title in [(axes[0], "41", "41.h5"),
                        (axes[1], "organic", "organic_labeled.h5")]:
    bx, by = sortxy(base[name])
    sx, sy = sortxy(det[name])
    # baseline
    ax.plot(bx, by, color="#7f7f7f", lw=1.6, label="from-scratch baseline")
    bbest_e = max(base[name], key=lambda e: base[name][e]); bbest = base[name][bbest_e]
    ax.axhline(bbest, color="#7f7f7f", ls=":", lw=1.2)
    ax.scatter([bbest_e], [bbest], color="#7f7f7f", zorder=5)
    ax.annotate(f"baseline best {bbest:.3f}@{bbest_e}", (bbest_e, bbest),
                textcoords="offset points", xytext=(-4, 8), fontsize=8, color="#555",
                ha="right")
    # SSL
    ax.plot(sx, sy, color="#2ca02c", lw=1.8, label="SSL backbone (dino_ssl1)")
    sbest_e = max(det[name], key=lambda e: det[name][e]); sbest = det[name][sbest_e]
    ax.axhline(sbest, color="#2ca02c", ls=":", lw=1.2)
    ax.scatter([sbest_e], [sbest], color="#2ca02c", zorder=6)
    ax.annotate(f"SSL best {sbest:.3f}@{sbest_e}", (sbest_e, sbest),
                textcoords="offset points", xytext=(-4, 8), fontsize=8.5,
                color="#1a7d1a", ha="right", fontweight="bold")
    delta = sbest - bbest
    sign = "+" if delta >= 0 else ""
    verdict = "WIN" if delta > 0.01 else ("tie" if delta > -0.01 else "behind")
    ax.set_title(f"{title}   best Δ = {sign}{delta:.3f}  ({verdict})", fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("ap_total"); ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, max(bbest, sbest) * 1.15)

fig.suptitle("Detector AP: from-scratch baseline vs round-1 SSL backbone", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
for n in ("41", "organic"):
    be = max(base[n], key=lambda e: base[n][e]); se = max(det[n], key=lambda e: det[n][e])
    print(f"  {n:8s} baseline {base[n][be]:.3f}@{be}  SSL {det[n][se]:.3f}@{se}  "
          f"Δ {det[n][se]-base[n][be]:+.3f}")
