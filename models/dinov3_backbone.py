"""DINOv3 backbones (public weights, DINOv3 License; frozen, or trained with --unfreeze) behind the interface of the swin in HeatmapNet: forward(NestedTensor) -> 4 feature maps at strides 4, 8, 16, 32.

The gray HE image is repeated to 3 channels and normalised with the ImageNet statistics the weights were trained with.
convnext_large: a pyramid like the swin (192, 384, 768, 1536 channels); stages 0-2 have no norm of their own in the release, they get a parameter-free channel LayerNorm.
vitl16 / vith16plus: one stride-16 grid; the tokens after blocks 6, 12, 18, 24 (final norm) are resampled to strides 4, 8, 16, 32, 1024 channels each.
Model code: the facebookresearch/dinov3 clone in env HM_DINOV3_REPO. load_state_dict is strict, so a wrong file fails loudly (the heatmap loaders call it with strict=False)."""
import os, sys
from functools import partial
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO = os.environ.get('HM_DINOV3_REPO', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dinov3'))   # git clone of facebookresearch/dinov3 (env HM_DINOV3_REPO, default <repo>/dinov3)
VIT_BLOCKS = {'dinov3_vitl16': (5, 11, 17, 23), 'dinov3_vith16plus': (7, 15, 23, 31)}   # tokens after blocks 6/12/18/24 (L, 24 blocks) or 8/16/24/32 (H+, 32 blocks)
VIT_DIM = {'dinov3_vitl16': 1024, 'dinov3_vith16plus': 1280}


class DinoV3Backbone(nn.Module):
    def __init__(self, kind, grad_ckpt=False):
        super().__init__()
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from dinov3.hub.backbones import dinov3_convnext_large, dinov3_vitl16, dinov3_vith16plus
        self.kind = kind
        self.net = {'dinov3_convnext_large': dinov3_convnext_large, 'dinov3_vitl16': dinov3_vitl16, 'dinov3_vith16plus': dinov3_vith16plus}[kind](pretrained=False)
        self.num_features = [192, 384, 768, 1536] if kind == 'dinov3_convnext_large' else [VIT_DIM[kind]] * 4
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if grad_ckpt:                               # per block, on the instance: the state-dict keys stay the same
            for b in ([b for s in self.net.stages for b in s] if kind == 'dinov3_convnext_large' else self.net.blocks):
                b.forward = partial(checkpoint, b.forward, use_reentrant=False)

    def train(self, mode=True):
        super().train(mode)
        if self.kind in VIT_BLOCKS:                 # train mode would rescale the RoPE coordinates at random (up to 2x), evaluation does not
            self.net.rope_embed.eval()
        return self

    def load_state_dict(self, sd, strict=True):
        return self.net.load_state_dict(sd, strict=True)

    def forward(self, nt):
        x = (nt.tensors[:, :1].expand(-1, 3, -1, -1) - self.mean) / self.std
        if self.kind == 'dinov3_convnext_large':
            out = []
            for i in range(4):
                x = self.net.stages[i](self.net.downsample_layers[i](x))
                y = x.permute(0, 2, 3, 1)
                y = self.net.norm(y) if i == 3 else F.layer_norm(y, y.shape[-1:], eps=1e-6)
                out.append(y.permute(0, 3, 1, 2).contiguous())
        else:
            H, W = x.shape[-2:]
            f = self.net.get_intermediate_layers(x, n=list(VIT_BLOCKS[self.kind]), reshape=True, norm=True)
            out = [F.interpolate(t, size=(H // s, W // s), mode='bilinear', align_corners=False) for t, s in zip(f, (4, 8, 16, 32))]
        return [SimpleNamespace(tensors=t) for t in out]
