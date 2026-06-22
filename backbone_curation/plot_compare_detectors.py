"""
Compare the two SSL-backbone detector runs: dino_ssl1 (round-1 backbone, COMPLETE) vs
dino_ssl2 (round-2 RECIPE_v2 backbone, IN PROGRESS). Two panels (41 + organic), full AP
curves with best markers, and a matched-epoch delta printed (only fair up to ssl2's last ep).

AP parsed from the slurm .out logs in ssl/  ('[epoch N] {41,organic} ap_total = X').

Usage:  python plot_compare_detectors.py  ->  figures/ssl1_vs_ssl2.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DET  = os.path.join(HERE, "ssl")
OUT  = os.path.join(HERE, "figures", "ssl1_vs_ssl2.png")


def load(prefix):
    """newest <prefix>*.out -> {'41':{ep:ap}, 'organic':{ep:ap}}"""
    outs = [f for f in os.listdir(DET) if f.startswith(prefix) and f.endswith(".out")]
    d = {"41": {}, "organic": {}}
    if not outs:
        return d
    newest = max(outs, key=lambda f: os.path.getmtime(os.path.join(DET, f)))
    pat = re.compile(r"\[epoch (\d+)\]\s+(41|organic)\s+ap_total\s*=\s*([0-9.]+)")
    for line in open(os.path.join(DET, newest)):
        m = pat.search(line)
        if m:
            d[m.group(2)][int(m.group(1))] = float(m.group(3))
    return d


def sortxy(d):
    xs = sorted(d); return xs, [d[x] for x in xs]


# ssl1 .out is 'dino_ssl-...'; ssl2 is 'dino_ssl2-...'. startswith('dino_ssl-') excludes ssl2.
ssl1 = load("dino_ssl-")
ssl2 = load("dino_ssl2-")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
for ax, name, title in [(axes[0], "41", "41.h5"),
                        (axes[1], "organic", "organic_labeled.h5")]:
    x1, y1 = sortxy(ssl1[name]); x2, y2 = sortxy(ssl2[name])
    ax.plot(x1, y1, color="#2ca02c", lw=1.6, label="dino_ssl1 (round-1, done)")
    ax.plot(x2, y2, color="#d62728", lw=1.8, label="dino_ssl2 (round-2, running)")
    # best markers
    for d, c, tag, dy in [(ssl1[name], "#1a7d1a", "ssl1", 8), (ssl2[name], "#a01818", "ssl2", -14)]:
        if d:
            be = max(d, key=lambda e: d[e])
            ax.scatter([be], [d[be]], color=c, zorder=6)
            ax.annotate(f"{tag} best {d[be]:.3f}@{be}", (be, d[be]),
                        textcoords="offset points", xytext=(0, dy), fontsize=8.5,
                        color=c, ha="center")
    # mark ssl2's current frontier
    if x2:
        ax.axvline(x2[-1], color="#d62728", ls=":", lw=0.8, alpha=.5)
        ax.annotate(f"ssl2 @ep{x2[-1]}", (x2[-1], 0.02), fontsize=7.5, color="#d62728",
                    rotation=90, va="bottom", ha="right")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("ap_total"); ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="lower right"); ax.set_ylim(0, None)

fig.suptitle("Detector AP — round-1 vs round-2 SSL backbone  (ssl2 still training)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)

# matched-epoch comparison up to ssl2's frontier (the only fair window)
for name in ("41", "organic"):
    if not ssl2[name]:
        continue
    cap = max(ssl2[name])
    common = [e for e in ssl2[name] if e in ssl1[name] and e <= cap]
    b1 = max(ssl1[name], key=lambda e: ssl1[name][e]); b2 = max(ssl2[name], key=lambda e: ssl2[name][e])
    # best of each within the matched window
    m1 = max((ssl1[name][e] for e in common), default=float("nan"))
    m2 = max((ssl2[name][e] for e in common), default=float("nan"))
    print(f"  {name:8s} | ssl1 best(all) {ssl1[name][b1]:.3f}@{b1}  ssl2 best(@ep<={cap}) "
          f"{ssl2[name][b2]:.3f}@{b2}  | matched<={cap}: ssl1 {m1:.3f} vs ssl2 {m2:.3f} "
          f"(Δ {m2-m1:+.3f})")
