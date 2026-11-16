# realsim - the real-background simulator, ported from `multi-channel-analysis`

Source: branch `origin/multi-channel-analysis` (commit 308a921), copied unchanged except for the import lines. Nothing here is imported by the
default heatmap path: `heatmap/train.py` only loads this package with `--sim realbkg`. The legacy `simulation.py` and `util/` are untouched;
the branch's modified `simulation.py` / `exp_preprocess.py` are vendored as `simulation_mc.py` / `exp_preprocess_mc.py`.

| file | origin on `multi-channel-analysis` |
|---|---|
| `simulation_mc.py` | `simulation.py` (adds `contrast_like_real`, raw intensity, edge peaks used by the physics code) |
| `exp_preprocess_mc.py` | `util/exp_preprocess.py` (adds `apply_contrast`) |
| `physics_simulation.py` | `physics_simulation.py` (hkl-bank peaks) |
| `realbkg_simulation.py` | `realbkg_simulation.py` |
| `realbkg_sim/{mosaic_background,detector_masks,background_v2,review_raw_backgrounds}.py` | `realbkg_sim/` (the donor-bank build scripts were not copied; `review_raw_backgrounds.score` would need `build_donor_bank`, training does not call it) |
| `factory.py` | new: the settings of the last DINO realbkg run (config r6), overridable |

`realbkg_sim/README.md` is the original description of what the donors are and why.

## Use
    python heatmap/bench_sim.py --sim realbkg --n 100 --save 12 --out <dir>     # speed, boxes per frame, PNGs with boxes (GPU: colorbox1)
    python heatmap/train.py --out <dir> --sim realbkg [--realsim_opt p_ring=0.4 contrast_min=2.5] <the usual flags>

## Data (env `GIWAXS_WORK`, default `/mnt/lustre/work/schreiber/szb389`; mosaic mode needs only these)
    $GIWAXS_WORK/datasets/cif_library_organic/bank/bank_organic_hkl.npz   5.1 GB  (read whole into RAM: ~7 GB resident PER PROCESS)
    $GIWAXS_WORK/datasets/sim_real_stats.npz                              52 KB
    $GIWAXS_WORK/datasets/realbkg_donors_mm/mask.npy, qmax.npy            95 MB
    $GIWAXS_WORK/datasets/detector_masks/work/                            81 MB
    $GIWAXS_WORK/tmp_diag/sim2/donors_final.json, donor_cache_repaired.npz   276 MB (the repaired 90 donor frames; if the cache is missing or does not match the json the code tries to read the raw frames, which are not copied)

colorbox1:

    DEST=/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data
    W=szb389@<galvani login address>:/mnt/lustre/work/schreiber/szb389
    mkdir -p $DEST/datasets/cif_library_organic/bank $DEST/datasets/realbkg_donors_mm $DEST/datasets/detector_masks $DEST/tmp_diag/sim2
    scp $W/datasets/cif_library_organic/bank/bank_organic_hkl.npz $DEST/datasets/cif_library_organic/bank/
    scp $W/datasets/sim_real_stats.npz $DEST/datasets/
    scp $W/datasets/realbkg_donors_mm/{mask.npy,qmax.npy} $DEST/datasets/realbkg_donors_mm/
    scp -r $W/datasets/detector_masks/work $DEST/datasets/detector_masks/
    scp $W/tmp_diag/sim2/{donors_final.json,donor_cache_repaired.npz} $DEST/tmp_diag/sim2/
    export GIWAXS_WORK=$DEST

## Known limits (inherited, measured on the other branch)
- ~0.3 s per frame (v2 background), the heatmap step needs ~0.12 s per image: the sim will be the bottleneck unless frames are produced by extra processes.
- Open gaps listed in `realbkg_sim/README.md`: unremoved real features in some donors (unlabelled positives), background autocorrelation 0.11 against 0.31 on real organic.
- DINO realbkg runs: organic about equal to the legacy sim, 41 far below (best 0.470 against 0.761). Not yet tested with the heatmap detector.
