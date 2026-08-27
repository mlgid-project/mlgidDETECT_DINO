"""Phase AD — is there a sim/real gap in FEATURE space, where, and would CORAL be aimed at it?

The proposal is a CORAL term: align the second-order feature statistics of simulated and real frames
during training. Unsupervised on the real side, training-only, no inference change. Before building
it, this measures the three things that decide whether it can work.

WHY IT MIGHT NOT. CORAL corrects COVARIATE shift -- same scenes, different appearance. Our gap is
partly LABEL shift, and the two gates disagree about how much:

  per frame          simulator   organic       41
  segments               38.3      98.6      24.0
  rings (SPAN)            7.9       3.5      16.9
  ring:segment          0.206     0.035     0.704
  same-radius nbr       0.532     0.889     0.565
  peak aspect            3.18      0.67      3.13

The simulator is BRACKETED by the gates, and on peak aspect and same-radius structure it sits
essentially on top of 41. So "label shift breaks CORAL" is a strong objection for organic and a weak
one for 41, and any verdict that quotes only organic is wrong. Both gates are carried throughout.

THREE MEASUREMENTS, per feature level and split into PEAK and BACKGROUND tokens:

  CORAL DISTANCE  ||C_A - C_B||_F^2 / (4 d^2) -- literally the quantity the loss would minimise, so
                  the numbers here are what the gradient would see, not a proxy for it.
  MEAN DISTANCE   ||mu_A - mu_B||^2 / d -- first order, which CORAL does NOT touch. If the gap is
                  mostly first-order, CORAL is aimed at the wrong moment.
  SEPARABILITY    held-out AUC of a Fisher discriminant trained to tell the two sources apart. Says
                  whether a gap that exists is actually usable, which a raw distance does not.

The PEAK/BACKGROUND split is the crux. Aligning statistics dominated by peak tokens means asking the
network to make a 38-peak frame look like a 99-peak one, and the cheapest way to do that is to blur
peaks into background -- destroying the signal. Aligning BACKGROUND tokens is genuine covariate
shift and is safe. So:

  gap mostly in BACKGROUND tokens  ->  background-restricted CORAL is well aimed; this says at which
                                       level to put it.
  gap mostly in PEAK tokens        ->  it is composition, CORAL would fight the detector; spend the
                                       effort on the simulator instead.
  near chance everywhere           ->  no appearance gap worth a loss term.

CONTROL, and it is the one that makes the numbers mean anything: **organic vs 41**. Two real sets
from different instruments and samples. If sim-vs-41 is SMALLER than organic-vs-41, the simulator is
already closer to one gate than the two gates are to each other, and no domain-adaptation term is
going to be the lever.

Features are the `input_proj` outputs -- 256-d at all four levels, the tensors that actually enter
the transformer, and the natural place a CORAL term would sit. Single model ssl1, so the features
are the ones a CORAL run would be aligning. No sklearn in this env, so the discriminant is closed
form (regularised Fisher) and AUC is rank-based.

GPU, ~15 min. See tmp_diag/run_domaingap.sbatch.
"""
import os, sys, json, random

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from simulation import FastSimulation
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset, detect_dataset_type
from util.labeleddataset import H5GIWAXSDataset
from diagnostics.label_completeness import build_model_from_ckpt, CKPT_A, CONFIG

DSETS = [("organic", "/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5"),
         ("41", "/mnt/lustre/work/schreiber/szb389/datasets/41.h5")]
N_SIM = 60
NL = 4                  # feature levels
TOK_PER_FRAME = 400     # subsample per frame / level / class for the discriminant
RIDGE = 1e-3
PAIRS = [("sim", "organic"), ("sim", "41"), ("organic", "41")]
CLASSES = ["peak", "background"]


class Acc:
    """Streaming mean and covariance, plus a token reservoir for the discriminant."""
    def __init__(self, d=256):
        self.d = d; self.n = 0
        self.s = torch.zeros(d, dtype=torch.float64)
        self.ss = torch.zeros(d, d, dtype=torch.float64)
        self.res = []

    def add(self, X):                      # X: [N, d] float32 cpu
        if not len(X):
            return
        Xd = X.double()
        self.n += len(Xd); self.s += Xd.sum(0); self.ss += Xd.T @ Xd
        k = min(TOK_PER_FRAME, len(X))
        idx = torch.randperm(len(X))[:k]
        self.res.append(X[idx].clone())

    def mean(self):
        return self.s / max(self.n, 1)

    def cov(self):
        m = self.mean()
        return self.ss / max(self.n, 1) - torch.outer(m, m)

    def tokens(self):
        return torch.cat(self.res) if self.res else torch.zeros(0, self.d)


def coral(a, b):
    """Deep-CORAL distance between two Acc: ||C_a - C_b||_F^2 / (4 d^2)."""
    d = a.d
    return float(((a.cov() - b.cov()) ** 2).sum() / (4.0 * d * d))


def meandist(a, b):
    return float(((a.mean() - b.mean()) ** 2).sum() / a.d)


def auc_fisher(A, B):
    """Held-out AUC of a regularised Fisher discriminant separating token sets A and B."""
    if len(A) < 50 or len(B) < 50:
        return float('nan')
    A = A.double(); B = B.double()
    ia, ib = torch.randperm(len(A)), torch.randperm(len(B))
    Atr, Ate = A[ia[: len(A) // 2]], A[ia[len(A) // 2:]]
    Btr, Bte = B[ib[: len(B) // 2]], B[ib[len(B) // 2:]]
    ma, mb = Atr.mean(0), Btr.mean(0)
    Ca = torch.cov(Atr.T); Cb = torch.cov(Btr.T)
    S = (Ca + Cb) / 2
    S += RIDGE * torch.diag(S).mean() * torch.eye(S.shape[0], dtype=S.dtype)
    try:
        w = torch.linalg.solve(S, (ma - mb))
    except Exception:
        return float('nan')
    sa, sb = Ate @ w, Bte @ w
    s = torch.cat([sa, sb]); y = torch.cat([torch.ones(len(sa)), torch.zeros(len(sb))])
    r = torch.argsort(torch.argsort(s)).double() + 1
    na, nb = len(sa), len(sb)
    a_ = (r[y == 1].sum() - na * (na + 1) / 2) / (na * nb)
    return float(max(a_, 1 - a_))          # orientation-free


def token_mask(boxes, H, W, imH, imW):
    """[H*W] bool: does this token's centre fall inside any GT box?"""
    if not len(boxes):
        return torch.zeros(H * W, dtype=torch.bool)
    ys = (torch.arange(H, dtype=torch.float32) + 0.5) * (imH / H)
    xs = (torch.arange(W, dtype=torch.float32) + 0.5) * (imW / W)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Y = Y.reshape(-1); X = X.reshape(-1)
    b = torch.as_tensor(boxes, dtype=torch.float32)
    m = torch.zeros(H * W, dtype=torch.bool)
    for k in range(len(b)):
        x0, y0, x1, y1 = min(b[k, 0], b[k, 2]), min(b[k, 1], b[k, 3]), \
                         max(b[k, 0], b[k, 2]), max(b[k, 1], b[k, 3])
        m |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return m


def make_hooks(model):
    feats = {}

    def mk(i):
        def h(_m, _inp, out):
            feats[i] = out.detach()
        return h
    hs = [model.input_proj[i].register_forward_hook(mk(i)) for i in range(len(model.input_proj))]
    return feats, hs


def run_source(name, frames, model, feats, dev, acc):
    n = 0
    with torch.no_grad():
        for img, boxes in frames():
            _ = model(img)
            n += 1
            imH, imW = img.shape[-2], img.shape[-1]
            for l in range(min(NL, len(feats))):
                f = feats[l]                                   # [1, 256, H, W]
                _, d, H, W = f.shape
                X = f[0].permute(1, 2, 0).reshape(-1, d).float().cpu()
                m = token_mask(boxes, H, W, imH, imW)
                acc[(name, l, 'peak')].add(X[m])
                acc[(name, l, 'background')].add(X[~m])
    print(f"  {name}: {n} frames", flush=True)


def real_frames(path, model_ch, dev):
    def gen():
        cfg = Config()
        cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]
        cfg.INPUT_DATASET = path
        ds = (PyGIDDataset(cfg, path=path, preprocess_func=standard_preprocessing, buffer_size=5,
                           load_labels=True) if detect_dataset_type(path) == 'pygid'
              else H5GIWAXSDataset(cfg, path=path, preprocess_func=standard_preprocessing,
                                   buffer_size=5))
        for gc in ds.iter_images():
            L = gc.polar_labels
            b = np.array(L.boxes, dtype=np.float64) if len(L.boxes) else np.zeros((0, 4))
            img = torch.tensor(gc.converted_polar_image[:, 0, :, :]).unsqueeze(0).to(dev) \
                       .repeat(1, model_ch, 1, 1)
            yield img, b
        if hasattr(ds, 'close'):
            ds.close()
    return gen


def sim_frames(n, model_ch, dev):
    def gen():
        sim = FastSimulation(device=dev)
        sim.sim_config.use_peak_clusters = False        # ssl1's training configuration
        for k in range(n):
            sd = 90000 + k
            random.seed(sd); torch.manual_seed(sd); np.random.seed(sd)
            try:
                im, bx, _m, _r = sim.simulate_img()
            except Exception:
                continue
            t = im if torch.is_tensor(im) else torch.tensor(im)
            while t.dim() > 2:
                t = t[0]
            img = t.float().unsqueeze(0).unsqueeze(0).to(dev).repeat(1, model_ch, 1, 1)
            yield img, bx.detach().cpu().numpy().astype(np.float64)
    return gen


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  SINGLE MODEL ssl1  N_SIM={N_SIM}", flush=True)
    model, a = build_model_from_ckpt(CONFIG, CKPT_A, dev)
    model.eval()
    feats, hooks = make_hooks(model)

    names = ["sim"] + [n for n, _ in DSETS]
    acc = {(n, l, c): Acc() for n in names for l in range(NL) for c in CLASSES}
    run_source("sim", sim_frames(N_SIM, a.num_channels, dev), model, feats, dev, acc)
    for nm, pth in DSETS:
        run_source(nm, real_frames(pth, a.num_channels, dev), model, feats, dev, acc)
    for h in hooks:
        h.remove()

    print("\n  tokens collected (peak / background), per level")
    for n in names:
        print(f"  {n:<10s}" + "  ".join(
            f"L{l}: {acc[(n, l, 'peak')].n:>7d}/{acc[(n, l, 'background')].n:>7d}"
            for l in range(NL)))

    out = {}
    for title, fn in [("CORAL DISTANCE  ||C_A-C_B||_F^2 / 4d^2  — what the loss would minimise", coral),
                      ("MEAN DISTANCE  ||mu_A-mu_B||^2 / d  — first order, CORAL does NOT touch this",
                       meandist)]:
        print("\n" + "=" * 100); print(f"  {title}"); print("=" * 100)
        for c in CLASSES:
            print(f"\n  {c} tokens")
            print(f"  {'pair':<20s}" + "".join(f"{('level ' + str(l)):>14s}" for l in range(NL)))
            for A, B in PAIRS:
                vals = [fn(acc[(A, l, c)], acc[(B, l, c)]) for l in range(NL)]
                out[f"{fn.__name__}|{c}|{A}-{B}"] = vals
                print(f"  {A + ' vs ' + B:<20s}" + "".join(f"{v:14.4g}" for v in vals))

    print("\n" + "=" * 100)
    print("  SEPARABILITY — held-out AUC, Fisher discriminant.  0.5 = indistinguishable")
    print("=" * 100)
    for c in CLASSES:
        print(f"\n  {c} tokens")
        print(f"  {'pair':<20s}" + "".join(f"{('level ' + str(l)):>14s}" for l in range(NL)))
        for A, B in PAIRS:
            vals = [auc_fisher(acc[(A, l, c)].tokens(), acc[(B, l, c)].tokens()) for l in range(NL)]
            out[f"auc|{c}|{A}-{B}"] = vals
            print(f"  {A + ' vs ' + B:<20s}" + "".join(f"{v:14.3f}" for v in vals))

    print("\n  READ IT AS: compare each sim-vs-<gate> row against the organic-vs-41 row directly")
    print("  below it. That control is two REAL sets; anything smaller than it is not a sim2real gap.")
    json.dump(out, open('/mnt/lustre/work/schreiber/szb389/tmp_diag/domain_gap.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
