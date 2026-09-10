#Learning-rate config: identical to the ssl1 recipe in every respect except the two learning
#rates. 1 channel, SSL backbone init, legacy simulation, 2.80/1.30 boxes, same schedule.
#
#Why these two numbers (measured 2026-09-07):
#  lr 4e-5  with lr_backbone 4e-5  -> dino_lr4e5_1, organic 0.6102 plateau. Best so far.
#  lr 1.6e-4 with lr_backbone 1.6e-4 -> dino_lrsweep_1, NEVER learned (organic 0.008-0.018 for
#  84 epochs) and was killed. A checkpoint autopsy found nothing broken: no NaNs, backbone
#  mean|W| 13.9 vs ssl1's 14.0, backbone drift only 0.106. It did not diverge, it just never
#  converged -- consistent with clip_max_norm=0.1 normalising the gradient so the step size is
#  essentially lr times a unit vector, with no magnitude feedback to settle near a minimum.
#
#So this run probes the untested range between 4e-5 and 1.6e-4 at the pair upstream DINO ships
#(lr 1e-4, lr_backbone lr/10), which this fork had abandoned -- every previous run here set
#lr_backbone == lr. Decoupling protects the SimMIM backbone, which is the part worth keeping.
#
#VERDICT IS CHEAP: ~9.3 min/epoch, and ssl1 was already at organic 0.203 by epoch 6 while the
#failed run never left 0.018. If organic ap_total is still flat by epoch 10, stop it -- that
#isolates step size rather than the backbone ratio, and the next probes are 8e-5/8e-5 or warmup
#(which this codebase has nowhere).
_base_ = ['DINO_4scale_swin_ssl.py']

lr = 1e-4
lr_backbone = 1e-5
