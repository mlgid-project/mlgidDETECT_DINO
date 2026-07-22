# Physics-based CIF simulation, FROM-SCRATCH detector training (docs/PHYSICS_SIM_INVESTIGATION.md).
#
# The FAITHFUL test of the physics lever. The warm-start variant (DINO_4scale_swin_physics.py)
# fine-tunes a converged ssl1 detector at 1e-5; that under-tests THIS lever specifically, because
# physics changes WHERE PEAKS ARE (the label/configuration distribution), not just appearance --
# and a converged model's query/box priors are already adapted to the standard sim's peak
# statistics. Empirically every warm-start data lever (M style-match, N struct-noise) peaked at
# ~epoch 10 and decayed, i.e. the warm start could mostly only lose.
#
# Recipe = ssl1's exactly (SSL backbone via backbone_dir, random detector head, uniform LR,
# 500 ep / lr-drop 280, NO amp) with the physics dilution present FROM EPOCH 0. So the
# apples-to-apples baseline is ssl1 itself (organic 0.586 / 41 0.762) = THIS recipe minus the
# physics dilution -> single variable.
#
# COST NOTE (measured 2026-07-22): from-scratch runs are amp=False, ~9.4 min/epoch -> lr-drop
# (ep280) at ~1.8 d, decisive plateau probe (~ep300) at ~2.0 d, full 500 ep at ~3.3 d. The
# a100 partition caps walltime at 3 d, so this needs ONE resubmit (the sbatch auto-resumes).
# amp=True would halve it but would add a confound vs the amp=False 0.586 control.
#
# TRAINING-ONLY: applied inside SimulationDataset.__getitem__ (main.py). Eval preprocessing,
# postprocessing and ONNX export are byte-identical to baseline; simulation.py is NOT modified.
#
# GATE = organic/41 AP + faint(vis=1)/high-q recall probe vs ssl1.
_base_ = ['DINO_4scale_swin_ssl.py']       # SSL backbone init (backbone_dir) + ssl1 schedule/LR

use_physics_sim = True
physics_sim_fraction = 0.5
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library/bank/bank.npz'
