"""
Convert a SimMIM checkpoint into detector-loadable swin-L backbone weights, and verify
it loads into a freshly-built detector backbone.

The detector loads a pretrained backbone in models/dino/backbone.py via
  checkpoint = torch.load(pretrainedpath)['model']
  backbone.load_state_dict(clean_state_dict(checkpoint) filtered, strict=False)
so we save {'model': <swin encoder state_dict>}. The encoder inside SimMIM IS the
SwinTransformer, so its keys (patch_embed.*, layers.*, norm*.*) match exactly.

Usage:
  python export_backbone.py --ckpt runs/simmim1/best.pth --out swin_simmim_giwaxs.pth
Then point the detector at it: place it in --backbone_dir and add to PTDICT
('swin_L_384_22k': 'swin_simmim_giwaxs.pth'), OR load directly (see printed instructions).
"""
import os, sys, argparse
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
from models.dino.swin_transformer import build_swin_transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SimMIM best.pth")
    ap.add_argument("--out", default=os.path.join(_HERE, "swin_simmim_giwaxs.pth"))
    ap.add_argument("--window_h", type=int, default=48)
    ap.add_argument("--window_w", type=int, default=6)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    # extract encoder.* -> swin state_dict (strip prefix), drop decoder/mask_token
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    print(f"extracted {len(enc)} encoder tensors (dropped {len(sd)-len(enc)} decoder/mask keys)")
    torch.save({"model": enc}, args.out)
    print(f"saved -> {args.out}")

    # ---- load test against a freshly-built detector backbone ----
    bb = build_swin_transformer('swin_L_384_22k', pretrain_img_size=384,
                                out_indices=(1, 2, 3), dilation=False, use_checkpoint=False,
                                window_size_h=args.window_h, window_size_w=args.window_w,
                                patch_size_h=4, patch_size_w=4, in_chans=1)
    res = bb.load_state_dict(enc, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("norm")]   # norm0 unused w/ out_indices=(1,2,3)
    print(f"\nload test: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.unexpected_keys:
        print("  unexpected (should be empty):", res.unexpected_keys[:8])
    if missing:
        print("  non-norm missing (should be empty):", missing[:8])
    ok = len(res.unexpected_keys) == 0 and len(missing) == 0
    print("  ==> " + ("OK: backbone weights transfer cleanly" if ok
                       else "CHECK: unexpected mismatch — inspect keys above"))
    print("\nTo use in the detector:")
    print(f"  1) cp {args.out} <backbone_dir>/swin_simmim_giwaxs.pth")
    print("  2) in models/dino/backbone.py PTDICT set 'swin_L_384_22k':'swin_simmim_giwaxs.pth'")
    print("  3) train the detector as usual (it will init the backbone from these SSL weights).")
    print("  NB: ensure --window_size_h/w and num_channels match (48/6, in_chans=1).")


if __name__ == "__main__":
    main()
