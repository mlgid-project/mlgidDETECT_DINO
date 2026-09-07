#Multi-CONTRAST DINO config: the 4 input channels are three CONTRASTS of the same polar
#image plus the mask, instead of three quantities derived from a single contrast (which is
#what DINO_4scale_swin_mc.py does, and what lost 0.022 organic ap_total against ssl1).
#
#The three contrasts and the sweep numbers that picked them are in util/channels.py
#(CONTRAST_CHANNELS); the matching simulator side is FastSimulation.contrast_stack.
_base_ = ['DINO_4scale_swin_ssl.py']

num_channels = 4
channel_mode = 'contrast'
