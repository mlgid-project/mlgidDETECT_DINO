"""
Snapshot the current training state into one figure:
  (1) SimMIM SSL train/val loss curve  (from ssl_runs/.../log.csv)
  (2) detector AP on 41      : SSL-backbone vs from-scratch baseline
  (3) detector AP on organic : SSL-backbone vs from-scratch baseline

Detector AP is parsed straight from the slurm .out log lines
  "[epoch N] {41,organic} ap_total = X".
Baseline AP comes from exp_ap_{41,organic}.txt in the baseline run dir.

Usage:
  python plot_state.py            # writes figures/training_state.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR       = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation"
SSL_LOG   = f"{CUR}/ssl_runs/simmim1/log.csv"   # round-1 (finished)
SSL2_LOG  = f"{CUR}/ssl_runs/simmim2/log.csv"   # round-2 (running, harder mask + aug v2)
DET_OUT  = os.path.join(os.path.dirname(__file__), "ssl")  # contains dino_ssl-*.out
BASELINE = "/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434"
OUT      = os.path.join(os.path.dirname(__file__), "figures", "training_state.png")


def load_ssl(path):
    ep, tr, va = [], [], []
    if os.path.exists(path):
        for line in open(path):
            t = line.strip().split(",")
            if len(t) >= 3:
                try:
                    ep.append(int(t[0])); tr.append(float(t[1])); va.append(float(t[2]))
                except ValueError:
                    pass
    return ep, tr, va


def load_det_from_out(out_dir):
    """parse '[epoch N] 41 ap_total = X' / '... organic ...' from newest dino_ssl-*.out"""
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


def load_baseline_txt(run_dir, name):
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


ssl_ep, ssl_tr, ssl_va = load_ssl(SSL_LOG)
det = load_det_from_out(DET_OUT)
base = {n: load_baseline_txt(BASELINE, n) for n in ("41", "organic")}
ssl2_ep, ssl2_tr, ssl2_va = load_ssl(SSL2_LOG)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# ---- (1) SSL loss: simmim1 (finished) + simmim2 (running)
ax = axes[0]
if ssl_ep:
    ax.plot(ssl_ep, ssl_tr, color="#1f77b4", lw=1.4, label="v1 train")
    ax.plot(ssl_ep, ssl_va, color="#d62728", lw=1.4, label="v1 val")
    be = min(range(len(ssl_va)), key=lambda i: ssl_va[i])
    ax.scatter([ssl_ep[be]], [ssl_va[be]], color="#d62728", zorder=5)
    ax.annotate(f"v1 best {ssl_va[be]:.4f}@ep{ssl_ep[be]}", (ssl_ep[be], ssl_va[be]),
                textcoords="offset points", xytext=(8, 16), fontsize=8, color="#d62728")
if ssl2_ep:
    ax.plot(ssl2_ep, ssl2_tr, color="#1f77b4", lw=1.4, ls="--", label="v2 train")
    ax.plot(ssl2_ep, ssl2_va, color="#ff7f0e", lw=1.4, ls="--", label="v2 val (mask .70+aug)")
    ax.scatter([ssl2_ep[-1]], [ssl2_va[-1]], color="#ff7f0e", zorder=5)
    ax.annotate(f"v2 ep{ssl2_ep[-1]}: {ssl2_va[-1]:.4f}", (ssl2_ep[-1], ssl2_va[-1]),
                textcoords="offset points", xytext=(6, -14), fontsize=8, color="#ff7f0e")
ax.set_title("SimMIM SSL pretraining (L1 recon) — v1 done, v2 running", fontsize=11)
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=.3); ax.legend(fontsize=7.5)

# ---- (2)&(3) detector AP vs baseline
for ax, name, title in [(axes[1], "41", "Detector AP — 41.h5"),
                        (axes[2], "organic", "Detector AP — organic_labeled.h5")]:
    bx, by = sortxy(base[name])
    sx, sy = sortxy(det[name])
    if bx:
        ax.plot(bx, by, color="#7f7f7f", lw=1.5, label="from-scratch baseline")
        bbest = max(base[name], key=lambda e: base[name][e])
        ax.axhline(base[name][bbest], color="#7f7f7f", ls=":", lw=1,
                   label=f"baseline best {base[name][bbest]:.3f}")
    if sx:
        ax.plot(sx, sy, color="#2ca02c", lw=1.8, marker="o", ms=3,
                label="SSL backbone")
        ax.scatter([sx[-1]], [sy[-1]], color="#2ca02c", zorder=5)
        ax.annotate(f"ep{sx[-1]}: {sy[-1]:.3f}", (sx[-1], sy[-1]),
                    textcoords="offset points", xytext=(8, -12), fontsize=9,
                    color="#2ca02c")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("ap_total"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("mlgidDETECT_DINO — SSL backbone training state  (snapshot)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
print(f"SSL: ep{ssl_ep[-1] if ssl_ep else '-'}  "
      f"det 41 ep{max(det['41']) if det['41'] else '-'}  "
      f"det organic ep{max(det['organic']) if det['organic'] else '-'}")
