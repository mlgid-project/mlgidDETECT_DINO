# Physics-based CIF simulation (Track B) — design & investigation record

**Status: build in progress (2026-07-20).** User-proposed direction: dilute the synthetic training
stream with images whose peak configurations come from *real crystallography* — CIF structures,
random orientations, structure-factor intensities — to close the synthetic/real gap at its source.

## 1. Motivation

- Five single-variable levers closed as documented negatives (Semi-DETR, Boost, CCTM, Co-DINO,
  style-match); struct-noise (phase N) in flight. The consistent read: the bottleneck is the
  DATA, and the 1-D histogram is not the gap that matters.
- The standard sim (`simulation.py`) places peaks at **random q with random intensities**. Two
  measured consequences (tmp_diag/qprofile.py, peak_contrast_vs_q.py, 2026-07-20):
  - column-level radial profiles are similar (log+HE flattens physical q-decay in both domains) —
    this killed the cheap "q-decay envelope" pre-test (Track A, declined before build);
  - but the **peak-level intensity-vs-q joint distribution is structurally wrong**: real labeled
    peaks are MOST contrasty at high q (organic high/low median ratio **14.3**, 41 ratio **2.0**)
    while synth is backwards (**0.70**) — synthetic low-q peaks too bright, high-q too faint,
    relative to what the detector must find at eval.
- No hand-tuned envelope fixes a joint distribution. Physics does it automatically: structure
  factors + form factors + real lattices give the correct marginal AND joint statistics of
  (q, chi, intensity), plus real morphology (powder rings vs textured arc/spot patterns).

## 2. Infrastructure discovered & used

- **`pygidsim`** (pip, mlgid-project): CIF → GIWAXS forward sim. `ExpParameters(q_xy_max, q_z_max,
  ai, en)`; `GIWAXSFromCif(path, params).giwaxs.giwaxs_sim(orientation)` with
  `None`=powder / explicit vector = textured. Validated end-to-end on user + COD CIFs.
- **`mlgidmatch`** (pip, mlgid-project): peak-to-structure matching (`Match.match_all`,
  `threshold` default 0.5; returns CIF + orientation + probability per phase). Used ONLY for the
  eval-exclusion list (user mandate), at lowered thresholds **0.10 ∪ 0.05**.
- New conda env **`mlgid_physics`** (py3.12; pygidsim, mlgidmatch, CUDA torch) — the training env
  `DINO_GIWAXS` is untouched; training-time consumption is npz-only.

## 3. CIF library

- User CIFs: `/mnt/lustre/work/schreiber/szb389/datasets/cif_library/` (51 files, incl. the
  group's 2D-perovskite/organic systems — the eval-family materials, handled by exclusion).
- COD perovskite selection: `/mnt/lustre/work/schreiber/szb082/CIFs/CIFs_cod_selection_perovskite/`
  (3,217 files; user approved use).
- Staged together (symlinks `user__*/cod__*` in `cif_library/_combined/`), then **screened**: each
  CIF must survive the actual physics code path (GIWAXSFromCif + powder sim). First screen:
  3,149/3,217 COD files valid (rejects: unknown elements e.g. deuterium, unparsable sections).

## 4. Eval-exclusion (user mandate)

`physics_sim/build_exclusion_list.py`; output `physics_sim/eval_matched.json`.

- Labeled peaks extracted from **41.h5, 45.h5** (roi_data: q = radius_px·qz_max/shape0, chi from
  the q_xy axis; type==1 → ring) and **organic_labeled.h5** (pygid: fitted_peaks in Å⁻¹).
- `Match.match_all` at thresholds **0.10 AND 0.05, union** ("really find all candidates"),
  segments + rings passes.
- **Built-in self-test** (runs before any eval matching is trusted): peaks simulated from known
  library CIFs with pygidsim must be re-matched by mlgidMATCH — validates the q-space conventions
  end-to-end; the full run aborts if self-recovery fails.
- **Exclusion policy** applied at bank generation:
  - matched powder/rings phase → that CIF's **powder mode excluded entirely** (no orientation
    freedom in a powder);
  - matched oriented phase → oriented bank entries within **10° fiber-axis margin** of the matched
    orientation excluded; **other orientations of the same structure remain usable** (user spec);
  - filename collisions with the enumerated eval-material list are **logged in the audit** for
    review (not blanket-excluded — user explicitly allows other orientations).

## 5. Peak bank

`physics_sim/generate_bank.py` → `cif_library/bank/bank.npz` + `bank_manifest.json`.

- Per valid CIF: 1 powder entry + up to 8 explicit random fiber orientations (recorded per entry —
  needed for the exclusion margin; `orientation='random'` would hide them), minus exclusions.
- Stored geometry-independently: (|q| Å⁻¹, chi° from q_xy axis, structure-factor intensity),
  top-200 peaks per entry. Powder entries flagged (chi = −1 sentinel).
- ExpParameters q_xy_max = q_z_max = 3.0 (bank covers |q| ≤ ~4.2); per-image q_max sampled at
  render time.

## 6. Training-time renderer

`physics_simulation.py` (NEW sibling file — **`simulation.py` is not modified**).

- `PhysicsSimulation.simulate_img()` honors the exact FastSimulation 4-tuple contract.
- Composition per image (user spec): **0–1 powder + 1–2 oriented** entries; per-entry overall
  scale 0.08–1.0 (minor/major phases → trains faint-phase sensitivity); intensity compression
  `(I/I_max)^gamma`, gamma 0.3–0.6, into the standard (2, 50) range (preserves physical ordering
  and relative contrast through the log+HE chain).
- Geometry: x = q/q_max·1024 (q_max ~ U[2.5, 4.5], the eval range), y = chi/90·512; rings =
  full-height boxes (is_ring=True), oriented spots = boxes with radial half-width 1–5 px and
  mosaic chi half-width 2–40 px (is_ring=False).
- **Appearance identical by construction**: reuses an internal FastSimulation instance for
  detector mask / dark areas / `img_from_labels` peak rendering, and the module-level chain
  (mul_perlin → add_glass → linear background → Poisson → stretch → dark areas → detector gaps →
  salt&pepper → log → HE → clip → kernel → digitalize → normalize → flip) in the same order as
  `simulate_img`. A physics image differs ONLY in peak placement + relative intensities.

## 7. Dilution splice & run plan

- `main.py` SimulationDataset: `use_physics_sim` (default off), `physics_sim_fraction`,
  `physics_bank_path`; per-sample draw picks physics vs standard generator.
- Config `DINO_4scale_swin_physics.py` + `run_detector_physics.sbatch`: warm-start ssl1,
  **fraction 0.5 for the first run** (user decision — maximize measurable impact, tune after).
- GATES (as every lever): organic/41 AP per epoch; decisive = faint(vis=1)/high-q recall probe
  (diag_compare.py) vs ssl1. Isolation: nothing in the struct-noise run, simulation.py, or any
  existing config is touched.

## 8. Verification before launch

1. bank stats (entries, q coverage, intensity-vs-q distribution vs the measured real relation);
2. exclusion audit (counts per rule; name-flag list reviewed);
3. render montage: physics vs standard vs real frames + box alignment;
4. dataset integration smoke (~50% physics rate, labels aligned, [0,1], zeros preserved);
5. py_compile + config load. Launch only after user review.

## 9. Track A appendix — q-decay pre-test: DECLINED BY MEASUREMENT (2026-07-20)

Pre-registered cheap test of the strongest single physics mechanism (high-q peaks weak). Killed
for ~30 GPU-min instead of a 1.7-day run: column-level radial profiles nearly flat in both
domains; peak-level relation INVERTED vs the hypothesis (real labeled high-q peaks are the most
contrasty; organic's faintest peaks live at LOW q). Implications: (a) an attenuation envelope
would push synth the wrong way — no run; (b) the high-q recall deficit is not labeled-peak
faintness → consistent with the structural/texture cause (phase N tests it); (c) the intensity
mismatch is a JOINT-distribution problem → exactly what this physics track fixes natively.
