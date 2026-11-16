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


#----------------------------------------------------------------------------------------
# multi-CONTRAST channels (channel_mode='contrast')
#----------------------------------------------------------------------------------------
# Three contrasts of the SAME raw polar image instead of three quantities derived from one
# contrast. Picked from the 74-setting sweep in diagnostics/sweep_contrast.py, scored with
# dino_ssl1 on the labeled sets (ap_total 41 / organic):
#
#   ch0  clip 5/99.5 + log + HE            0.7441 / 0.5683   deployed default, "combined"
#   ch1  clip 5/99.5 + log + CLAHE 4@16x16 0.6621 / 0.5883   best organic of all 74
#   ch2  clip 5/99.5 + log + gamma 0.7     0.7541 / 0.5380   41-facing channel
#   ch3  mask
#
# ch2 keeps the LOG. The no-log variant scores marginally better on 41 (0.7586, same ap_high
# 0.891) but collapses to 0.4652 on organic, and diagnostics/dump_contrast_images.py shows
# why: without the log the organic frames go nearly black and most labeled peaks sit on empty
# background. Keeping the log costs 0.0045 on 41 and buys 0.073 on organic.
#
# ch0 MUST stay first: backbone.py zero-pads the pretrained single-channel patch embed into
# channel 0, so step 0 reproduces the 1-channel model exactly (diagnostics/mc_smoke.py [4]).
# All three share clip 5/99.5, which is why the simulator applies the clip once and branches.
CONTRAST_CHANNELS = [
    {'name': 'log_he',            'clip': (5.0, 99.5), 'log': True,  'gamma': None, 'he': True,  'clahe': None},
    {'name': 'log_clahe4_16x16',  'clip': (5.0, 99.5), 'log': True,  'gamma': None, 'he': False, 'clahe': (4.0, 16, 16)},
    {'name': 'log_gamma0.7',      'clip': (5.0, 99.5), 'log': True,  'gamma': 0.7,  'he': False, 'clahe': None},
]


def build_contrast_channels(raw_polar, mask, specs=None):
    """Build the 4-channel image from the RAW (pre-contrast) polar image and its mask.

    Note the input: unlike build_channels, which takes the already-contrasted image, this
    takes `raw_polar_image` -- contrasting an already-contrasted image would apply the
    percentile clip and the log twice.

    numpy in (H, W), torch (4, H, W) float32 out on `device`.
    """
    import numpy as np
    from .exp_preprocess import apply_contrast

    specs = CONTRAST_CHANNELS if specs is None else specs
    raw = np.asarray(raw_polar, dtype=np.float32)
    m = np.asarray(mask).reshape(raw.shape).astype(bool)
    chans = [apply_contrast(raw, m, s) for s in specs]
    chans.append(m.astype(np.float32))
    return torch.from_numpy(np.stack(chans))