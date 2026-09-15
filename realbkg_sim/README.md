# realbkg_sim — building the real-background donor bank

Everything here exists to produce the two files `realbkg_simulation.py` loads:

| file | what it is |
|---|---|
| `$DATA/sim_background_bank4.h5` | 444 real GIWAXS frames, polar, with every diffraction feature removed |
| `$DATA/sim_real_stats.npz` | peak widths, brightness and counts measured from real 2-D peak fits |

The simulator itself is `realbkg_simulation.py` in the repo root; the notebook it was developed in
is `giwaxs_sim.ipynb`.

## Why a donor bank at all

Every previous simulator modelled the whole frame — background field, Poisson, read noise, a
correlated-noise term — with each piece fitted to a real statistic, and the frames still did not
look real. A real GIWAXS frame carries detector panel seams, a low-q halo, radial streaks,
beam-stop geometry, gain structure and a sample-and-beamline-specific q-falloff that a small
parameter set does not reproduce. So the background is not modelled: it is a real frame with its
own peaks taken out. Only the peaks are simulated, and from pygidSIM only their raw positions and
raw intensities are used.

Ground truth is sound because the donor's own peaks are removed before use: every feature the
detector is asked to find is one we put there and know the position of.

## Order to run

```bash
python realbkg_sim/inventory.py          # corpus inventory + eval leak check -> inventory.json
python realbkg_sim/peak_fits.py          # what real peaks look like (prints only)
python realbkg_sim/width_stats.py        # -> widthstats.npz
python realbkg_sim/amp_calibration.py    # -> ampcal.npy   (needs build_donor_bank.suppress)
python realbkg_sim/make_stats.py         # -> $DATA/sim_real_stats.npz
BANK_OUT=$DATA/sim_background_bank4.h5 \
  python realbkg_sim/build_donor_bank.py 12     # -> the donor bank  (~25 min)
python realbkg_sim/check_bank.py         # eyeball the result
python realbkg_sim/autocorr_check.py     # run after ANY change to suppress()
```

Paths come from `GIWAXS_WORK` / `REALBKG_CACHE` / `DONOR_SRC`, defaulting to the cluster layout.

## What was measured, and what it changed

Peak parameters are mlgidFIT's own 2-D Gaussian fits of 1,926 real peaks in 57 real frames
(`fitted_peaks/parameters_peak`), not estimates from 1-D cuts. Two results drove the design:

* **Real peaks are arcs, not blobs.** Median σ_q 3.9 px against σ_χ 20.3 px, and the two are
  uncorrelated per peak (r = 0.00). The old simulator drew both from one narrow range.
* **Real peaks are faint.** Median amplitude 3.1× the local noise, p10 1.5×.

Widths also vary *within* a frame — sd(log σ_q) = 0.53, sd(log σ_χ) = 0.83 — so one frame holds
both sharp and broad peaks.

## Leakage

Donors are `ekaterina_aftermlgidFIT` (4,040 linear frames, 46 entries, organic + perovskite).
Max azimuthal-I(q) cosine against any `organic_labeled.h5` frame is **0.888**, below the 0.949
same-material baseline and far below the 0.999 same-frame value. No evaluation frame is in the
donor pool. The matcher is the one validated for the SSL corpus check — q-calibrated,
χ-averaged I(q) — because 2-D pixel cosine is unreliable across conversion pipelines.

## Two things that are wrong on purpose to find out about

**Donor leftovers — the one that matters.** Selected donors score leftover-z ≈ 15 against a
detection threshold of 3, so a few frames still carry a faint unremoved *real* feature with no box
on it: an unlabelled positive the detector is penalised for finding. The candidate fix is to stop
relying on the hand-built q-narrow detector here and run the trained detector over the donor
corpus, inpainting what *it* finds.

**Background autocorrelation 0.11 against real organic's 0.31.** Not the data's fault —
unprocessed donors measure 0.53. Part of it was the fill (see `autocorr_check.py`), now fixed; the
rest is a structural tension in donor selection, and it is the same tension as above: `leftover` is
amplitude-over-noise, so a frame with a *white* background hides its own residual features and
scores as clean. Selecting clean donors therefore selects uncorrelated ones. `realbkg_simulation.py`
ranks on both (plus background/noise ratio and replaced fraction) to trade them off; ranking does
not remove the conflict. Fixing the leftovers properly is what would let the selection stop trading.
