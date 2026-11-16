# realsim_v2

Simple realistic GIWAXS simulator for the heatmap detector: GNoME structures -> pygidSIM peaks / powder rings -> pseudo-Voigt rendering, absolute counts, background,
detector noise, gaps, beam stop, missing wedge -> polar image 512 x 1024 (|q| 0..7.07 A^-1 on x, chi 0..90 deg on y) with box labels. Written by the user in
`/home/nicolerch/Documents/mlgid/pygidSIM/realsim/` (`realsim_tests.ipynb` = viewer and label studies).

## Code (repo, branch heatmap) vs data (colorbox dir)

| what | where |
|---|---|
| generator code (`pipeline.py`, `polar_labels.py`, `generate_dataset.py`) | `realsim_v2/` in this repo; byte-identical copies of the user's files, update with `realsim_v2/sync_from_pygidsim.sh` (checksums in `SOURCE_SHA256.txt`); nothing in pygidSIM is modified |
| repo-side logic (reader, contrast, boxes for the training convention) | `realsim_v2/dataset.py`, `heatmap/train.py --sim realsim_v2` |
| GNoME structures (`by_id.zip`, `stable_materials_summary.csv`) | `/mnt/DATA/gnome/` (input data, read only) |
| generated datasets (shards, config.json, failures, gen.log) | `/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/<name>/` |
| training runs of the heatmap model on them | `/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs/newbox/runs/` (as for all runs) |

`/mnt/DATA/giwaxs_sim/test100` is the user's own 100-image test output (kept where it is, used for smoke tests only).

## Use

    realsim_v2/gen.sh NAME N [WORKERS]                               # generate (only on the user's go; resumable)
    python heatmap/train.py ... --box_coef 3.0,3.0 --sim realsim_v2 --v2_data /mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/NAME [--v2_flip 0|1]

The shards store per-label sigmas, so `train.py --box_coef a,w` (full width in sigma, a angular, w radial) builds the boxes for any convention; the stored
`boxes` of the generation config are not used. The image goes through the deployed contrast (clip 5/99.5 percentile, log10, histogram equalisation, masked = 0),
the same input as the validation frames. Evaluation is unchanged (real validation sets, scaled GT).
