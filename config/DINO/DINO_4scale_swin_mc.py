#Multi-channel DINO config for 4-channel input (B0, B1, B2, B3), SSL based
_base_ = ['DINO_4scale_swin_ssl.py']

num_channels = 4