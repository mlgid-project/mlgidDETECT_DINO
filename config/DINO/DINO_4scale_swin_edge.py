#Edge-peak config: identical to the ssl1 recipe (1 channel, SSL backbone, same lr/schedule);
#the ONLY change is on the simulator side -- peaks cut off by the dark wedge or a detector gap
#keep their FULL box instead of being clamped to the mask edge or deleted.
#
#Motivation, measured 2026-09-06 on the labeled sets: 34.8% of 41's GT boxes and 20.1% of
#organic's overlap an invalid pixel, in 100% of images, while the simulator produced ~none --
#so the model was trained to stop its boxes at the mask and never learns the cut-off case.
_base_ = ['DINO_4scale_swin_ssl.py']

edge_peaks = True
