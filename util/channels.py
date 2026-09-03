import torch

def he_channel(img, mask): 
    #Channel 0 (B0): img, 0 for invalid pixels.
    return img.masked_fill(~mask, 0.)

def mask_channel(mask):
    #Channel 3 (B3): mask, 1 for valid pixels, 0 for invalid pixels.
    return mask.masked_fill(mask, 1.).to(torch.float32)

def column_profile(img, mask):
    #"Channel 2 (B2): per q-column masked median over chi, broadcast to (H, W)."
    H, W = img.shape
    x = img.masked_fill(~mask, float('inf'))        # invalid -> +inf, sinks to the bottom
    s, _ = torch.sort(x, dim=0)                     # per column, ascending
    n = mask.sum(dim=0)                             # valid count per column, (W,)
    idx = ((n - 1) // 2).clamp(min=0)               # median row of the valid values
    med = torch.gather(s, 0, idx[None, :])          # (1, W)
    med = med.masked_fill((n == 0)[None, :], 0.)    # empty columns -> 0
    return med.expand(H, W).masked_fill(~mask, 0.)

def ring_subtracted(img, profile, mask): 
    #Channel 1 (B1): img - profile on valid pixels, 0 elsewhere. Range [-1, 1].
    return (img - profile).masked_fill(~mask, 0.)

def build_channels(img, mask):
    #"Build the 4-channel image from the input image and mask."
    profile = column_profile(img, mask)
    return torch.stack([he_channel(img,mask), 
                        ring_subtracted(img, profile, mask),
                        profile,
                        mask_channel(mask)], dim = 0)