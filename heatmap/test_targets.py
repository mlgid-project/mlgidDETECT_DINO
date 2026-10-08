"""Equality test: vectorised build_targets == the reference per-ring-loop version (exit 1 on any mismatch).
  python heatmap/test_targets.py [n_trials]      (CPU is fine; run on colorbox1, not the cluster login node)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from heatmap import targets_loss as new, targets_loss_ref as ref

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
g = torch.Generator().manual_seed(0); bad = 0; tn = tr = 0.0
for t in range(n_trials):
    n = int(torch.randint(1, 120, (1,), generator=g))
    x0 = torch.rand(n, generator=g) * 1000; y0 = torch.rand(n, generator=g) * 480
    w = 1 + torch.rand(n, generator=g) * 30; hh = 3 + torch.rand(n, generator=g) * 500
    ringm = torch.rand(n, generator=g) < 0.2
    hh = torch.where(ringm, 40 + torch.rand(n, generator=g) * 470, hh)
    b = torch.stack([x0, y0, (x0 + w).clamp(max=1024), (y0 + hh).clamp(max=512)], 1)
    lab = ringm.long()
    for mode in ('ridge', 'legacy'):
        t0 = time.time(); a = new.build_targets(b, lab, 512, 1024, 2, mode); tn += time.time() - t0
        t0 = time.time(); c = ref.build_targets(b, lab, 512, 1024, 2, mode); tr += time.time() - t0
        # heat target: equal up to float rounding (the vectorised exp differs from the loop's by ~1 ulp = 6e-8);
        # regression targets and weights must match EXACTLY
        ok = torch.allclose(a[0], c[0], atol=1e-6, rtol=0) and all(torch.equal(p, q) for p, q in zip(a[1:], c[1:]))
        if not ok:
            bad += 1
            if bad < 4:
                print('MISMATCH trial', t, mode, [float((p - q).abs().max()) for p, q in zip(a, c)])
print(f'mismatches {bad} of {2 * n_trials} | time new {tn:.2f}s ref {tr:.2f}s')
sys.exit(1 if bad else 0)
