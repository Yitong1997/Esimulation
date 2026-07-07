# Walkthrough: Phase Residual Calculation Update

## Objective
Update the phase residual calculation in `HybridElementPropagator` (`hybrid_element_propagator.py`) to use the phase at the beam center (chief ray) as the reference zero point, rather than subtracting the mean phase (Piston). This ensures that the absolute phase information relative to the Pilot Beam is preserved, which is critical for accurate physical optics propagation and interferometric analysis.

## Changes Made

### 1. Phase Residual Logic (`hybrid_element_propagator.py`)

**Location:** `_propagate_local_raytracing` method.

**Previous Implementation:**
The code previously performed a least-squares fit to remove the "Piston" term (mean offset). This effectively centered the residual phase around zero mean.
```python
# Old Logic
A_fit = np.column_stack([np.ones_like(x_out)])
fit_coeffs, _, _, _ = np.linalg.lstsq(A_fit, residual_opd_waves, rcond=None)
residual_opd_waves = residual_opd_waves - fit_coeffs[0]
```

**New Implementation:**
The logic was changed to identify the ray closest to the coordinate origin (the chief ray) and subtract its specific residual value from the entire wavefront.
```python
# New Logic
r_sq_out = x_out**2 + y_out**2
center_idx = np.argmin(r_sq_out)
residual_at_center = residual_opd_waves[center_idx]
residual_opd_waves = residual_opd_waves - residual_at_center
```

**Rationale:**
- **Absolute Reference:** The Pilot Beam serves as the reference phase. By forcing the residual at the chief ray (center) to be zero, we assert that the chief ray's optical path length difference matches the reference.
- **Piston Sensitivity:** Subtracting the mean (average) phase makes the global phase sensitive to the beam's intensity distribution and aberrations (asymmetry). Subtracting the center value provides a stable geometric reference.

### 2. Ray Sampling Optimization

**Location:** `_propagate_local_raytracing` and `_sample_rays_from_wavefront`.

**Changes:**
- **Grid Metadata:** The sampling function `_sample_rays_from_wavefront` now returns `grid_metadata`, which captures the original grid shape and indices (`grid_info`). This metadata is passed to the reconstructor to potentially optimize the reconstruction process using grid-based finite differences (via `GridRayReconstructor`).
- **Validity Masking:** The strict validity masking (filtering out rays based on intensity threshold) was adjusted to ensure the center ray is always included and valid indices are tracked correctly.

## Verification Checklist

To verify these changes, ensuring the following:

1.  **Zero Residual at Origin:**
    - In a debug run, checking `residual_opd_waves[center_idx]` immediately after the subtraction should yield `0.0`.

2.  **No Phase Jumps:**
    - Propagating a beam through a simple system (e.g., flat mirror) should not introduce arbitrary phase jumps (multiples of $2\pi$) due to mean subtraction artifacts.

3.  **Physical Path Length:**
    - The output phase in `PROPER` should now reflect the true physical optical path length relative to the pilot beam, rather than a "floating" relative phase.

## Related Files
- `src/hybrid_optical_propagation/hybrid_element_propagator.py`: Core logic changes.
- `src/wavefront_to_rays/reconstructor.py`: (Context only) Standard reconstructor logic.
