"""Phase AF — where does the 9.06 min/epoch actually go?

Motivation: a proposal to cut compute by reducing/merging encoder tokens (superpixels, or the
grid-preserving Sparse-DETR / Lite-DETR variants). Before choosing an instrument, measure the
premise. Nothing in this project has ever profiled a training step.

What is already known from reading the code, and is the reason this is worth measuring:
  main.py:494-499   the training DataLoader runs with num_workers=0
  main.py:122-129   SimulationDataset.__getitem__ calls simulate_img() ON THE GPU
so data generation is SERIALISED with the training step in the same CUDA stream -- no overlap at
all. With __len__=1000 and batch_size=2 that is 500 iterations/epoch, and the observed 9.06
min/epoch is 1.087 s/iteration INCLUDING simulation.

FOUR blocks:
  A  wall-clock split of one training iteration: data (= the simulator) / H2D / forward / loss
     (incl. Hungarian matching) / backward / optimizer. Every boundary torch.cuda.synchronize()d.
     This is the decision-maker.
  B  inside the model, by forward hooks: backbone (swin-L) / input_proj / encoder (6 deformable
     layers over 10880 tokens) / decoder (6 layers, 900 queries). Two-stage proposal generation and
     top-900 selection is recovered as transformer - encoder - decoder. Hooks synchronise, so the
     hooked total is inflated; block A's unhooked forward time is the honest denominator and both
     are printed.
  C  the token arithmetic the proposal turns on: tokens per feature level, and what a given encoder
     cut would be worth END TO END. Deformable attention and the FFN are both O(N) in tokens, so a
     linear projection from the measured encoder time is sound -- but it IS a projection and is
     labelled as one, not a measurement.
  D  peak memory at batch_size=2, i.e. whether there is headroom to raise it. NOTE this is not a
     free lever: changing batch size changes the optimisation and would need its own gated run.

Also splits the data cost into simulate_img() vs the rest of __getitem__ (preprocessing), because
the two have completely different fixes.

READ-ONLY: builds the model and dataset, runs the loop, writes nothing to any run directory.
No checkpoint is loaded -- timing does not depend on the weights, only on the shapes.

GPU, ~15 min. See tmp_diag/run_stepprof.sbatch.
"""
import os, sys, json, time, argparse

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

import main as M
from util.slconfig import SLConfig

CONFIG = os.environ.get('PROF_CONFIG', os.path.join(_REPO, 'config/DINO/DINO_4scale_swin_ssl.py'))
OUT = os.environ.get('PROF_OUT', '/mnt/lustre/work/schreiber/szb389/tmp_diag/step_profile')
WARMUP = int(os.environ.get('PROF_WARMUP', 10))
ITERS = int(os.environ.get('PROF_ITERS', 60))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build():
    parser = argparse.ArgumentParser('prof', parents=[M.get_args_parser()], add_help=False)
    args = parser.parse_args(['-c', CONFIG, '--output_dir', OUT])
    args.rank = 0
    os.makedirs(OUT, exist_ok=True)
    cfg = SLConfig.fromfile(args.config_file)
    for k, v in cfg._cfg_dict.to_dict().items():
        if k not in vars(args):
            setattr(args, k, v)
    for k, d in [('use_ema', False), ('debug', False), ('amp', False)]:
        if not getattr(args, k, None):
            setattr(args, k, d)
    # main.py sets this at :615, after the arg parser, right before main(args); the backbone
    # builder reads it (models/dino/backbone.py:177) so it has to exist here too.
    args.export = False
    # AF.1 levers, measured rather than projected. Defaults leave the config untouched.
    if os.environ.get('PROF_NOCKPT'):
        args.use_checkpoint = False
    if os.environ.get('PROF_AMP'):
        args.amp = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main():
    args = build()
    dev = torch.device(args.device)
    print(f"device={dev}  config={os.path.basename(CONFIG)}  amp={args.amp}  use_dn={args.use_dn}"
          f"  use_checkpoint={getattr(args, 'use_checkpoint', None)}", flush=True)

    model, criterion, _post = M.build_model_main(args)
    model.to(dev); model.train()
    criterion.to(dev); criterion.train()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_par / 1e6:.1f} M params", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ds = M.SimulationDataset(args)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                                     collate_fn=M.collate_fn)
    print(f"dataset len={len(ds)}  batch_size=2  num_workers=0"
          f"  -> {len(ds) // 2} iterations/epoch", flush=True)

    # ---------- block B wiring: forward hooks on the real module boundaries ----------
    hook_t, hook_on, _mark = {}, [False], {}

    def mk(name, mod):
        def pre(m, i):
            if hook_on[0]:
                sync(); _mark[name] = time.perf_counter()
        def post(m, i, o):
            if hook_on[0]:
                sync(); hook_t.setdefault(name, []).append(time.perf_counter() - _mark[name])
        mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)

    tr = getattr(model, 'transformer', None)
    targets_h = [('backbone', getattr(model, 'backbone', None)),
                 ('transformer', tr),
                 ('transformer.encoder', getattr(tr, 'encoder', None) if tr else None),
                 ('transformer.decoder', getattr(tr, 'decoder', None) if tr else None)]
    ip = getattr(model, 'input_proj', None)
    if ip is not None:
        for j, m in enumerate(ip):
            targets_h.append((f'input_proj[{j}]', m))
    for nm, mod in targets_h:
        if mod is not None:
            mk(nm, mod)
        else:
            print(f"  (no module {nm}; skipped)", flush=True)

    # ---------- blocks A + B ----------
    stages = ['data', 'h2d', 'forward', 'loss', 'backward', 'optim']
    T = {s: [] for s in stages}
    shapes_seen = set()
    it = iter(dl)
    torch.cuda.reset_peak_memory_stats()

    for k in range(WARMUP + ITERS):
        rec = k >= WARMUP
        hook_on[0] = rec and (k >= WARMUP + ITERS // 2)   # hooks ON for the 2nd half only

        sync(); t0 = time.perf_counter()
        try:
            samples, targets = next(it)
        except StopIteration:
            it = iter(dl); samples, targets = next(it)
        sync(); t1 = time.perf_counter()

        samples = samples.to(dev)
        targets = [{kk: v.to(dev) for kk, v in t.items()} for t in targets]
        sync(); t2 = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(samples, targets) if args.use_dn else model(samples)
            sync(); t3 = time.perf_counter()
            ld = criterion(out, targets)
            losses = sum(ld[kk] * criterion.weight_dict[kk]
                         for kk in ld.keys() if kk in criterion.weight_dict)
        sync(); t4 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        if args.amp:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        sync(); t5 = time.perf_counter()

        if args.amp:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        sync(); t6 = time.perf_counter()

        if rec:
            for s, dt in zip(stages, [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]):
                T[s].append(dt)
        if hasattr(samples, 'tensors'):
            shapes_seen.add(tuple(samples.tensors.shape))
        elif torch.is_tensor(samples):
            shapes_seen.add(tuple(samples.shape))
        if k == 0:
            print(f"  first iter ok; input shape(s) {shapes_seen}", flush=True)
        if rec and (k - WARMUP) % 20 == 0:
            print(f"  iter {k - WARMUP + 1}/{ITERS}", flush=True)

    hook_on[0] = False
    peak = torch.cuda.max_memory_allocated() / 2**30

    # ---------- block A report ----------
    tot = float(np.mean([sum(T[s][i] for s in stages) for i in range(len(T['data']))]))
    print("\n" + "=" * 92)
    print("  A  WALL-CLOCK SPLIT OF ONE TRAINING ITERATION  (batch_size=2, every stage synced)")
    print("=" * 92)
    print(f"  {'stage':<14s}{'mean s':>10s}{'p50 s':>10s}{'std':>9s}{'% of iter':>11s}")
    for s in stages:
        v = np.asarray(T[s])
        print(f"  {s:<14s}{v.mean():10.4f}{np.median(v):10.4f}{v.std():9.4f}"
              f"{100 * v.mean() / tot:11.1f}")
    print(f"  {'TOTAL':<14s}{tot:10.4f}{'':10s}{'':9s}{100.0:11.1f}")
    print(f"\n  measured {tot:.4f} s/iter  ->  {tot * (len(ds) // 2) / 60:.2f} min/epoch"
          f"   (observed in dino_qaspect1: 9.06 min/epoch)")

    # ---------- data cost, split ----------
    print("\n" + "=" * 92)
    print("  A.1  WHAT THE 'data' STAGE IS  (simulate_img vs the rest of __getitem__)")
    print("=" * 92)
    n = 30
    sync(); a = time.perf_counter()
    for i in range(n):
        _ = ds[i]
    sync(); b = time.perf_counter()
    sim = getattr(ds, 'simulation', None)
    tsim = float('nan')
    if sim is not None:
        sync(); c = time.perf_counter()
        for i in range(n):
            _ = sim.simulate_img()
        sync(); d = time.perf_counter()
        tsim = (d - c) / n
    tget = (b - a) / n
    print(f"  __getitem__            {tget:.4f} s/image  ->  {2 * tget:.4f} s/batch of 2")
    print(f"  simulate_img alone     {tsim:.4f} s/image  ->  {2 * tsim:.4f} s/batch of 2")
    print(f"  rest of __getitem__    {tget - tsim:.4f} s/image  (preprocessing/augmentation)")
    print(f"  block A 'data' stage   {np.mean(T['data']):.4f} s/batch")

    # ---------- block B report ----------
    print("\n" + "=" * 92)
    print("  B  INSIDE THE MODEL  (forward hooks, synced -> inflated; see note)")
    print("=" * 92)
    if hook_t:
        hb = {k2: float(np.mean(v)) for k2, v in hook_t.items()}
        ipk = [k2 for k2 in hb if k2.startswith('input_proj')]
        if ipk:
            hb['input_proj (all)'] = sum(hb[k2] for k2 in ipk)
        enc, dec, trf = hb.get('transformer.encoder'), hb.get('transformer.decoder'), hb.get('transformer')
        if None not in (enc, dec, trf):
            hb['transformer: two-stage + rest'] = trf - enc - dec
        fwd = float(np.mean(T['forward']))
        print(f"  {'module':<34s}{'mean s':>10s}{'% of fwd':>11s}{'% of iter':>11s}")
        for k2 in ['backbone', 'input_proj (all)', 'transformer', 'transformer.encoder',
                   'transformer.decoder', 'transformer: two-stage + rest']:
            if k2 in hb:
                print(f"  {k2:<34s}{hb[k2]:10.4f}{100 * hb[k2] / fwd:11.1f}"
                      f"{100 * hb[k2] / tot:11.1f}")
        print(f"\n  NOTE hooks synchronise at every boundary, so these sum to MORE than block A's")
        print(f"  unhooked forward ({fwd:.4f} s). Percentages use the unhooked forward/iter as the")
        print(f"  denominator, so they are conservative upper bounds on each module's real share.")
    else:
        print("  no hook data collected")

    # ---------- block C ----------
    print("\n" + "=" * 92)
    print("  C  TOKEN ARITHMETIC  — what an encoder cut would be worth END TO END")
    print("=" * 92)
    H, W = 512, 1024
    lv = [(H // s, W // s, s) for s in (8, 16, 32, 64)]
    N = sum(h * w for h, w, _ in lv)
    print(f"  {'level':<10s}{'stride':>8s}{'H x W':>14s}{'tokens':>10s}{'share':>9s}")
    for i, (h, w, s) in enumerate(lv):
        print(f"  {'L' + str(i):<10s}{s:8d}{f'{h} x {w}':>14s}{h * w:10d}{h * w / N:9.3f}")
    print(f"  {'TOTAL':<10s}{'':8s}{'':14s}{N:10d}{1.0:9.3f}")
    enc = (hook_t and float(np.mean(hook_t.get('transformer.encoder', [float('nan')]))))
    if enc == enc:
        print(f"\n  measured encoder {enc:.4f} s = {100 * enc / tot:.1f}% of the iteration.")
        print("  Deformable attention and the FFN are both O(N) in tokens, so a token cut scales")
        print("  the encoder linearly.  PROJECTED end-to-end saving (a projection, not a measurement):")
        print(f"  {'encoder tokens cut':<24s}{'encoder s':>12s}{'iter s':>10s}{'min/epoch':>12s}{'speedup':>10s}")
        for cut in (0.0, 0.25, 0.50, 0.75, 1.00):
            it_s = tot - enc * cut
            print(f"  {cut:<24.0%}{enc * (1 - cut):12.4f}{it_s:10.4f}"
                  f"{it_s * (len(ds) // 2) / 60:12.2f}{tot / it_s:10.3f}x")
        print("  (the 100% row is the unreachable limit: a free encoder. It bounds the whole axis.)")

    # ---------- block D ----------
    print("\n" + "=" * 92)
    print("  D  MEMORY")
    print("=" * 92)
    print(f"  peak allocated at batch_size=2: {peak:.2f} GiB")
    if torch.cuda.is_available():
        tot_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"  device total: {tot_mem:.1f} GiB   -> naive headroom factor ~{tot_mem / peak:.1f}x")
    print("  CAVEAT: raising batch_size changes the optimisation (effective LR, BN/DN statistics),")
    print("  so it is a SCIENTIFIC variable needing its own gated run, not a free speedup.")

    json.dump(dict(iter_s=tot, stages={s: float(np.mean(T[s])) for s in stages},
                   hooks={k2: float(np.mean(v)) for k2, v in hook_t.items()},
                   getitem_s=tget, sim_s=tsim, peak_gib=peak, tokens=N),
              open('/mnt/lustre/work/schreiber/szb389/tmp_diag/step_profile.json', 'w'),
              indent=2, default=str)
    print("\nPROBE DONE")


if __name__ == '__main__':
    main()
