# Biconic Focus Residual Phase Design

Date: 2026-05-12

## Context

The current run in `sandbox/run_biconic_focus_expand_pop.py` matches the Zemax pilot beam at focus, but not the final field width:

- Zemax final beam width: `x=0.0506606 mm`, `y=0.0511189 mm`
- POP final beam width: `x=0.1782473 mm`, `y=0.1791586 mm`
- Zemax pilot size: `0.0504519 mm`
- POP pilot size: `0.0504689 mm`

This makes the pilot parameters unlikely to be the dominant error. The suspected source is residual phase reconstruction around S23. In `pop/propagation/element.py`, the current reconstruction path uses all rays:

```python
residual_opd_waves_recon = residual_opd_waves
input_amplitude_recon = input_amplitude
reconstruction_mask = np.ones(len(ray_x_in), dtype=bool)
```

So `RECONSTRUCTION_MASK_RATIO` is not currently constraining the S23 reconstruction. Cubic `griddata` over all rays can let low-intensity edge samples dominate the residual phase grid and create ring artifacts at focus.

## Goals

1. Prove whether the final beam-width inflation is caused by residual phase rather than pilot parameters.
2. Add a controlled S23 residual-phase reconstruction path that suppresses edge artifacts.
3. Keep the existing behavior as the default unless an experiment mode is enabled.
4. Produce validation output that distinguishes geometry/pilot agreement from wavefront reconstruction failure.

## Non-Goals

- Do not change the Zemax baseline files.
- Do not tune pilot refit as the primary fix for this issue.
- Do not replace the whole POP propagation pipeline.
- Do not make destructive changes to existing debug output.

## Proposed Modes

Add an experimental residual phase mode, configured from the run script and passed through `PropagationOptions`.

`RESIDUAL_PHASE_MODE`:

- `normal`: current behavior.
- `zero_at_surfaces`: for specified element exits, force the reconstructed residual phase grid to zero.
- `core_griddata`: reconstruct residual phase only from high-intensity core rays, then taper the grid residual phase outside the core.
- `zernike`: use a low-order model as a controlled comparison or fallback.

Supporting settings:

- `ZERO_RESIDUAL_PHASE_SURFACE_INDICES`, initially `[23]`
- `CORE_RECON_SURFACE_INDICES`, initially `[23]`
- `CORE_INTENSITY_THRESHOLD`, initially `0.05` or `0.10` of peak intensity
- `CORE_TAPER_WIDTH_PIXELS`, or an equivalent beam-radius fraction
- `CORE_MIN_RAYS`, below which reconstruction falls back and records a warning
- `CORE_FALLBACK_MODE`, initially `zernike` or `normal`

## Data Flow

At element exit, after tracing and pilot update:

1. Compute `absolute_opd_waves`.
2. Compute `pilot_opd_waves` from the updated pilot.
3. Compute centered `residual_opd_waves`.
4. Select a residual phase handling mode:
   - `normal`: pass all rays to the current reconstruction path.
   - `zero_at_surfaces`: still reconstruct or preserve amplitude, but set `residual_phase_grid = 0` before forming `full_phase`.
   - `core_griddata`: build a ray mask from `input_amplitude` or intensity, reconstruct using only core rays, then taper residual phase to zero outside the core.
   - `zernike`: fit a low-order residual model, preferably on the same core mask.
5. Compute `full_phase = pilot_phase + residual_phase_grid`.
6. Write the field back to PROPER and continue propagation to S25/S26.

The first experiment should run two variants for `zero_at_surfaces`:

- keep the existing reconstructed amplitude and zero only residual phase;
- optionally use pilot Gaussian amplitude to isolate amplitude reconstruction as well.

## Validation Plan

Run and compare three output directories:

1. `normal`: reproduce the current inflated beam width near `0.178 mm`.
2. `zero_at_s23`: confirm whether final beam width returns near `0.0505 mm`.
3. `core_griddata_s23`: check whether final beam width approaches the Zemax baseline without ring artifacts.

Primary metrics:

- final `beam_width_x_mm`
- final `beam_width_y_mm`
- final `pilot_size_mm`
- S23/S25 wavefront folding diagnostics
- saved focus and S23/S25 residual phase plots

Initial pass criteria:

- `zero_at_s23` final beam width returns near the Zemax value, confirming causality.
- `core_griddata_s23` reduces beam-width relative error from about 250% to below 10%.
- `pilot_size_mm` remains near `0.05045 mm`.
- no NaN/Inf enters the reconstructed amplitude or phase grids.

The final target can then be tightened to the existing 5% validation tolerance.

## Error Handling

If the core mask produces too few rays, interpolation fails, or tapering creates invalid values:

- record surface index, mode, threshold, valid ray count, and fallback reason;
- fall back to the configured `CORE_FALLBACK_MODE`;
- preserve debug plots and JSON diagnostics for that surface;
- do not silently accept a failed reconstruction.

## Recommended Implementation Order

1. Add the `zero_at_surfaces` experiment path for S23 only.
2. Run validation to prove or reject the residual-phase hypothesis.
3. Add `core_griddata` with debug metadata and plots.
4. Compare `core_griddata` against `zernike` for S23.
5. Promote the best S23-specific mode into the expand validation script while leaving global defaults unchanged.
