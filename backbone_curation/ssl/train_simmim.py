"""
SimMIM pretraining loop for the swin-L 48x6 backbone on backbone_ssl_corpus.h5.
Single-GPU (tuned for 1x A100 40GB): AMP + gradient checkpointing + grad accumulation.

Pretrains the EXACT detector backbone from scratch (ImageNet weights are window-incompatible).
Validates on held-out WHOLE scans. Saves best/last checkpoints; export_backbone.py converts
the best one into detector-loadable backbone weights.

Example:
  PYTHONPATH=<repo> python train_simmim.py --epochs 200 --batch 8 --accum 4 --out_dir runs/simmim1
"""
import os, sys, math, time, csv, argparse
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from simmim_model import SimMIM
from ssl_dataset import build_loaders, H5_DEFAULT


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, m, v in loader:
        x, m, v = x.to(device, non_blocking=True), m.to(device, non_blocking=True), v.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            loss, _ = model(x, m, v)
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs", "simmim1"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)          # effective batch = batch*accum
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--mask_patch", type=int, default=32)
    ap.add_argument("--drop_path", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--aug", default="v1", choices=["v1", "v2"],
                    help="train-time augmentation recipe (v2 = ssl/RECIPE_v2.md)")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  effective_batch={args.batch*args.accum}", flush=True)

    train_loader, val_loader = build_loaders(
        args.h5, args.batch, args.workers, args.val_frac, args.mask_patch, args.mask_ratio,
        aug=args.aug)

    model = SimMIM(mask_patch=args.mask_patch, drop_path_rate=args.drop_path,
                   use_checkpoint=True).to(device)
    # no weight decay on norms / biases / mask token
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "mask_token" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    start_epoch, gstep, best = 0, 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start_epoch = ck["epoch"] + 1
        gstep = ck.get("gstep", 0); best = ck.get("best", float("inf"))
        print(f"resumed from {args.resume} @ epoch {start_epoch}", flush=True)

    logf = open(os.path.join(args.out_dir, "log.csv"), "a", newline="")
    logw = csv.writer(logf)
    if start_epoch == 0:
        logw.writerow(["epoch", "train_loss", "val_loss", "lr", "sec"]); logf.flush()

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, m, v) in enumerate(train_loader):
            x, m, v = x.to(device, non_blocking=True), m.to(device, non_blocking=True), v.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss, _ = model(x, m, v)
            scaler.scale(loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                lr = cosine_lr(gstep, total_steps, warmup_steps, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                gstep += 1
            run += loss.item(); nb += 1
            if it % 50 == 0:
                print(f"  ep{epoch} it{it}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        tr = run / max(1, nb)
        val = validate(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch}] train {tr:.4f}  val {val:.4f}  lr {cur_lr:.2e}  {dt:.0f}s", flush=True)
        logw.writerow([epoch, f"{tr:.5f}", f"{val:.5f}", f"{cur_lr:.2e}", f"{dt:.0f}"]); logf.flush()

        ck = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                  epoch=epoch, gstep=gstep, best=best, args=vars(args))
        torch.save(ck, os.path.join(args.out_dir, "last.pth"))
        if val < best:
            best = val; ck["best"] = best
            torch.save(ck, os.path.join(args.out_dir, "best.pth"))
            print(f"   * new best val {best:.4f}", flush=True)
    logf.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
