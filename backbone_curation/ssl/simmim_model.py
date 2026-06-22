"""
SimMIM masked-image-modeling model for the mlgidDETECT_DINO swin-L backbone.

Why SimMIM (not vanilla MAE): the detector backbone is hierarchical/windowed Swin, which
needs an intact spatial grid (window_partition), so we cannot drop masked tokens ViT-style.
SimMIM keeps the full grid, replaces masked patches with a learnable mask token, and
reconstructs raw pixels of masked patches with a lightweight head -> the pretrained encoder
is the *same* SwinTransformer the detector uses, so weights transfer directly.

Encoder is built with the EXACT detector config (swin_L_384_22k, window 48x6, in_chans=1,
patch 4, out_indices=(1,2,3), dilation=False) so its state_dict keys/shapes match.
Final encoder feature: (B, 1536, H/32, W/32) = (B,1536,16,32) for 512x1024 input.
Decoder (SimMIM): 1x1 conv + PixelShuffle(32) -> per-pixel prediction. L1 loss on
masked AND valid (non-no-data) pixels only.
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo root on path so we reuse the detector's exact swin implementation
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from models.dino.swin_transformer import build_swin_transformer


class SimMIM(nn.Module):
    def __init__(self, img_h=512, img_w=1024, patch_size=4, mask_patch=32,
                 window_h=48, window_w=6, in_chans=1, drop_path_rate=0.2,
                 use_checkpoint=True, encoder_stride=32):
        super().__init__()
        self.patch_size = patch_size
        self.mask_patch = mask_patch
        self.encoder_stride = encoder_stride
        self.in_chans = in_chans
        # token-resolution upsample factor for the mask grid, and pixel-resolution factor
        self.tok_factor = mask_patch // patch_size          # 32//4 = 8
        self.pix_factor = mask_patch                        # 32

        # EXACT detector backbone config -> weights transfer
        self.encoder = build_swin_transformer(
            'swin_L_384_22k', pretrain_img_size=384,
            out_indices=(1, 2, 3), dilation=False, use_checkpoint=use_checkpoint,
            window_size_h=window_h, window_size_w=window_w,
            patch_size_h=patch_size, patch_size_w=patch_size,
            in_chans=in_chans, drop_path_rate=drop_path_rate)
        self.embed_dim = self.encoder.embed_dim              # 192
        self.enc_out_dim = self.encoder.num_features[-1]     # 1536

        # learnable mask token (injected at patch-embed resolution)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # SimMIM decoder: feature map (B,1536,h,w) -> pixels (B,in_chans,H,W)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.enc_out_dim, encoder_stride ** 2 * in_chans, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def encode(self, x, mask_grid):
        """x: (B,in_chans,H,W); mask_grid: (B,gh,gw) bool -> final feature (B,1536,h,w)."""
        enc = self.encoder
        x = enc.patch_embed(x)                               # (B, C0, Wh, Ww)
        B, C, Wh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)                     # (B, Wh*Ww, C)
        # inject mask token where masked
        tok_mask = mask_grid.repeat_interleave(self.tok_factor, 1).repeat_interleave(self.tok_factor, 2)
        tok_mask = tok_mask[:, :Wh, :Ww].reshape(B, Wh * Ww, 1).type_as(x)
        mt = self.mask_token.expand(B, Wh * Ww, -1)
        x = x * (1.0 - tok_mask) + mt * tok_mask
        x = enc.pos_drop(x)
        H = W = None
        for i in range(enc.num_layers):
            x_out, H, W, x, Wh, Ww = enc.layers[i](x, Wh, Ww)
        x_out = enc.norm3(x_out)                             # final-stage norm (out_indices has 3)
        feat = x_out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x, mask_grid, valid=None):
        """x:(B,1,H,W) float [0,1]; mask_grid:(B,gh,gw) bool; valid:(B,1,H,W) bool or None.
        Returns (loss, recon)."""
        feat = self.encode(x, mask_grid)
        recon = self.decoder(feat)                           # (B,in_chans,H,W)
        pix_mask = mask_grid.repeat_interleave(self.pix_factor, 1).repeat_interleave(self.pix_factor, 2)
        pix_mask = pix_mask[:, :x.shape[2], :x.shape[3]].unsqueeze(1).type_as(x)
        if valid is not None:
            pix_mask = pix_mask * valid.type_as(x)           # never score the no-data corner
        denom = pix_mask.sum().clamp(min=1.0)
        loss = (F.l1_loss(recon, x, reduction='none') * pix_mask).sum() / denom
        return loss, recon


if __name__ == "__main__":
    # construction + shape smoke test (forward only if CUDA is present; Swin-L@512x1024 is heavy on CPU)
    m = SimMIM(use_checkpoint=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"SimMIM built: encoder embed_dim={m.embed_dim} enc_out_dim={m.enc_out_dim}; params={n:.1f}M")
    if torch.cuda.is_available():
        m = m.cuda()
        x = torch.rand(2, 1, 512, 1024, device='cuda')
        mg = (torch.rand(2, 16, 32, device='cuda') < 0.6)
        v = (x > 0)
        with torch.cuda.amp.autocast():
            loss, rec = m(x, mg, v)
        print(f"forward ok: recon {tuple(rec.shape)} loss {loss.item():.4f}")
    else:
        print("(no CUDA here; run the smoke forward on the A100)")
