# Physics-based CIF simulation dilution (Track B; docs/PHYSICS_SIM_INVESTIGATION.md).
#
# A fraction of training images get their PEAK CONFIGURATION from real crystallography:
# structures, orientations, and structure-factor intensities simulated from CIF files with
# pygidsim (offline bank: physics_sim/generate_bank.py -> bank.npz), composed per image as
# 0-1 powder + 1-2 oriented patterns (user spec). The appearance pipeline (backgrounds, noise,
# masks, log/HE) is REUSED unmodified from simulation.py, so physics images differ from standard
# ones only in where peaks are and their relative intensities. Eval-set structures/orientations
# are excluded via mlgidMATCH (physics_sim/build_exclusion_list.py, thresholds 0.10+0.05).
#
# Motivation: five detector/data levers null/negative; measured intensity-vs-q shows the synth
# joint distribution is unphysical in a structured way (backwards vs real labeled peaks at both
# ends) -- physics structure factors give the real joint distribution automatically.
#
# TRAINING-ONLY: applied inside SimulationDataset.__getitem__ (main.py); eval path and ONNX
# byte-identical to baseline. simulation.py is NOT modified (physics_simulation.py is a sibling).
#
# First run at fraction 0.5 (user decision: maximize the measurable impact; tune afterwards).
# Warm-start screen from ssl1. GATE = organic/41 AP + faint(vis=1)/high-q recall probe vs ssl1.
_base_ = ['DINO_4scale_swin.py']

use_physics_sim = True
physics_sim_fraction = 0.5
physics_bank_path = '/mnt/lustre/work/schreiber/szb389/datasets/cif_library/bank/bank.npz'
