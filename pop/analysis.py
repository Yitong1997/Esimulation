"""Analysis helpers shared by visualization and reporting."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pop.core import PilotBeamParams, PropagationState


def compute_log_intensity(
    intensity: NDArray[np.floating],
    floor_db: float = -60.0,
) -> NDArray[np.floating]:
    intensity = np.asarray(intensity, dtype=float)
    max_val = float(np.max(intensity)) if intensity.size else 0.0
    if max_val <= 0:
        return np.zeros_like(intensity, dtype=float)
    floor_linear = 10 ** (floor_db / 10.0)
    normalized = intensity / max_val
    normalized = np.clip(normalized, floor_linear, None)
    return 10.0 * np.log10(normalized)


def compute_valid_mask(amplitude: NDArray[np.floating], threshold: float) -> NDArray[np.bool_]:
    amplitude = np.asarray(amplitude, dtype=float)
    if amplitude.size == 0:
        return np.zeros_like(amplitude, dtype=bool)
    max_amp = float(np.max(amplitude))
    if max_amp <= 0:
        return np.zeros_like(amplitude, dtype=bool)
    return amplitude > threshold * max_amp


def compute_intensity_moments(
    intensity: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
) -> Optional[tuple[float, float, float, float, float]]:
    intensity = np.asarray(intensity, dtype=float)
    if mask is not None:
        intensity = np.where(mask, intensity, 0.0)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(intensity))
    if total <= 0:
        return None
    centroid_x = float(np.sum(x_grid * intensity) / total)
    centroid_y = float(np.sum(y_grid * intensity) / total)
    var_x = float(np.sum((x_grid - centroid_x) ** 2 * intensity) / total)
    var_y = float(np.sum((y_grid - centroid_y) ** 2 * intensity) / total)
    sigma_x = float(np.sqrt(max(var_x, 0.0)))
    sigma_y = float(np.sqrt(max(var_y, 0.0)))
    return centroid_x, centroid_y, sigma_x, sigma_y, total


def _normalize_refractive_index(value: float | None) -> float:
    if value is None:
        return 1.0
    try:
        n_val = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(n_val) or n_val <= 0:
        return 1.0
    return n_val


def _compute_pilot_m2(pilot: PilotBeamParams) -> float:
    n_val = _normalize_refractive_index(pilot.current_refractive_index)
    w0 = float(getattr(pilot, "waist_radius_mm", 0.0) or 0.0)
    if not np.isfinite(w0) or w0 <= 0:
        return np.nan
    wavelength_mm = pilot.wavelength_um * 1e-3
    lambda_medium_mm = wavelength_mm / n_val
    if lambda_medium_mm <= 0:
        return np.nan
    theta = lambda_medium_mm / (np.pi * w0)
    return float(np.pi * w0 * theta / lambda_medium_mm)



def _fit_phase_curvature_scan(
    phase: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    intensity: NDArray[np.floating],
    wavelength_mm: float,
    refractive_index: float,
    scan_points: int = 50,
) -> float:
    """
    Global curvature scan to find best focus, robust against phase aliasing.
    """
    # Flatten valid pixels
    max_int = float(np.max(intensity)) if intensity.size else 0.0
    if max_int <= 0:
        return np.nan
        
    mask = intensity > (max_int * 0.1)
    if np.sum(mask) < 10:
        return np.nan

    x_flat = x_grid[mask]
    y_flat = y_grid[mask]
    ph_flat = phase[mask]
    r2_flat = x_flat**2 + y_flat**2
    
    k = 2 * np.pi * refractive_index / wavelength_mm
    
    # Hierarchical Scan Strategy
    # The Coherent Sum peak width is proportional to lambda/D^2.
    # For full aperture, peak is extremely narrow, leading to "missed peak" in coarse scan.
    # Solution: Use smaller aperture (center) for Coarse Scan to broaden the peak.
    
    max_r2 = np.max(r2_flat)
    limit_r2 = max_r2 * (0.25 ** 2) # Inner 25% radius (1/16th area)
    mask_inner = r2_flat <= limit_r2
    
    # If inner mask is too small, fallback to full
    if np.sum(mask_inner) < 100:
        mask_inner = np.ones_like(r2_flat, dtype=bool)

    # Prepare Inner Data for Coarse Scan
    r2_inner = r2_flat[mask_inner]
    ph_inner = ph_flat[mask_inner]
    phasor_inner = np.exp(1j * ph_inner)

    # Pass 1: Global Coarse Scan on Inner Aperture
    # Scan Range: +/- 0.1 mm^-1 (covering R > 10mm to flat)
    # Expanded from 0.02 to capture highly divergent beams (gradient > 20pi implies R ~ 25mm)
    c_min = -0.1
    c_max = 0.1
    c_values_coarse = np.linspace(c_min, c_max, 100)
    
    def _compute_scan_score(c_vals, r2_pts, phasor_pts):
        scores = []
        for c in c_vals:
            # phi_model = 0.5 * k * r^2 * c
            phi_model = 0.5 * k * r2_pts * c
            phasor_model = np.exp(1j * phi_model)
            # Coherent Sum
            phasor_diff = phasor_pts * np.conj(phasor_model)
            vector_sum = np.sum(phasor_diff)
            scores.append(np.abs(vector_sum))
        return scores

    scores_coarse = _compute_scan_score(c_values_coarse, r2_inner, phasor_inner)
    best_idx_coarse = np.argmax(scores_coarse)
    best_c_coarse = c_values_coarse[best_idx_coarse]
    
    # Pass 2: Local Fine Scan on FULL Aperture
    # Zoom in: +/- 0.002 mm^-1 around coarse peak
    phasor_full = np.exp(1j * ph_flat)
    
    zoom_width = 0.002
    c_min_fine = best_c_coarse - zoom_width
    c_max_fine = best_c_coarse + zoom_width
    c_values_fine = np.linspace(c_min_fine, c_max_fine, scan_points * 2) # Double points for fine
    scores_fine = _compute_scan_score(c_values_fine, r2_flat, phasor_full)
    best_idx_fine = np.argmax(scores_fine)
    best_c = c_values_fine[best_idx_fine]
    
    # Parabolic Interpolation for Sub-grid Precision
    if 0 < best_idx_fine < len(scores_fine) - 1:
        y_left = scores_fine[best_idx_fine - 1]
        y_ex = scores_fine[best_idx_fine]
        y_right = scores_fine[best_idx_fine + 1]
        
        # Denom is usually negative for a peak
        denom = y_left - 2 * y_ex + y_right
        if denom != 0:
            delta = 0.5 * (y_left - y_right) / denom
            dc = c_values_fine[1] - c_values_fine[0]
            best_c += delta * dc
    
    if best_c == 0:
        return np.inf
    return 1.0 / best_c


def _fit_phase_curvature(
    phase: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    weights: NDArray[np.floating],
    centroid_x: float,
    centroid_y: float,
    wavelength_mm: float,
    refractive_index: float,
    enable_global_scan: bool = False,
) -> float:
    if phase.size == 0:
        return np.nan
    if wavelength_mm <= 0 or refractive_index <= 0:
        return np.nan
        
    # Robust Gradient-Based Fitting (avoids 2D unwrapping issues)
    # Model: phi(x, y) = a * (x^2 + y^2)
    # dphi/dx = 2*a*x, dphi/dy = 2*a*y
    
    dx = float(x_grid[0, 1] - x_grid[0, 0])
    dy = float(y_grid[1, 0] - y_grid[0, 0])
    
    phasor = np.exp(1j * phase)
    
    # Gradient X (forward diff)
    # shape: (N, M-1)
    d_phasor_x = phasor[:, 1:] * np.conj(phasor[:, :-1])
    # angle(d_phasor) gives the phase jump, wrapped to [-pi, pi].
    # This correctly computes dphi/dx * dx as long as alias constraint holds.
    grad_x = np.angle(d_phasor_x) / dx
    
    x_mid_x = 0.5 * (x_grid[:, 1:] + x_grid[:, :-1]) - centroid_x
    # y coordinates are same along columns, but we slice to match shape
    # y_grid[:, 1:] or y_grid[:, :-1] are same
    
    # Weights X
    if weights is not None:
        w_x = np.minimum(weights[:, 1:], weights[:, :-1])
    else:
        # Implicit uniform weights where finite
        w_x = (np.isfinite(grad_x)).astype(float)
        
    # Mask invalid gradients (NaNs in phase result in NaNs in grad)
    valid_x = np.isfinite(grad_x) & (w_x > 0)
    
    # Gradient Y
    # shape: (N-1, M)
    d_phasor_y = phasor[1:, :] * np.conj(phasor[:-1, :])
    grad_y = np.angle(d_phasor_y) / dy

    # CHECK ALIASING (Added for Robustness)
    # Only if enable_global_scan is requested
    if enable_global_scan:
        max_grad = 0.0
        if grad_x.size > 0:
            max_grad = max(max_grad, np.max(np.abs(grad_x)))
        if grad_y.size > 0:
            max_grad = max(max_grad, np.max(np.abs(grad_y)))
            
        if max_grad > 0.9 * np.pi:
            print(f"[POP] Warning: Phase gradient {max_grad/np.pi:.2f} pi exceeds Nyquist limit. Switching to Global Scan.")
            # Re-use intensity/weights logic. If weights is None, we need Intensity?
            # In current usage (compute_gaussian_fit_and_m2), 'weights' IS 'intensity'.
            # If weights is None, we construct a dummy intensity?
            scan_intensity = weights if weights is not None else np.ones_like(phase)
            return _fit_phase_curvature_scan(
                phase, x_grid, y_grid, scan_intensity, wavelength_mm, refractive_index
            )
    
    y_mid_y = 0.5 * (y_grid[1:, :] + y_grid[:-1, :]) - centroid_y
    
    # Weights Y
    if weights is not None:
        w_y = np.minimum(weights[1:, :], weights[:-1, :])
    else:
        w_y = (np.isfinite(grad_y)).astype(float)
        
    valid_y = np.isfinite(grad_y) & (w_y > 0)
    
    # Least squares for C = 2*a
    # sum(w * grad * coord) / sum(w * coord^2)
    
    numer = np.sum(w_x[valid_x] * grad_x[valid_x] * x_mid_x[valid_x]) + \
            np.sum(w_y[valid_y] * grad_y[valid_y] * y_mid_y[valid_y])
            
    denom = np.sum(w_x[valid_x] * x_mid_x[valid_x]**2) + \
            np.sum(w_y[valid_y] * y_mid_y[valid_y]**2)

    if denom == 0:
        return np.inf # Flat curve assumption if no signal? Or NaN. Let's return inf (R) implies a=0.
                      # If denom=0, we have no data.
        return np.nan

    C = numer / denom
    a_coeff = C / 2.0
    
    # Coverage check
    total_points = np.count_nonzero(valid_x) + np.count_nonzero(valid_y)
    if total_points < 20:
        return np.nan

    if abs(a_coeff) < 1e-15:
        return np.inf
    k = 2.0 * np.pi * refractive_index / wavelength_mm
    return float(k / (2.0 * a_coeff))


def compute_gaussian_fit_and_m2(
    state: PropagationState, mask_threshold: float = 0.01
) -> dict[str, float]:
    sim_amp = np.asarray(state.amplitude, dtype=float)
    if sim_amp.size == 0:
        return {}
    max_amp = float(np.max(sim_amp))
    if not np.isfinite(max_amp) or max_amp <= 0:
        return {}
    valid_mask = sim_amp > mask_threshold * max_amp
    if not np.any(valid_mask):
        return {}

    grid = state.grid_sampling
    pilot = state.pilot_beam_params
    coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
    x_grid, y_grid = np.meshgrid(coords, coords)
    intensity = sim_amp**2

    moments = compute_intensity_moments(intensity, x_grid, y_grid, mask=valid_mask)
    if moments is None:
        return {}
    centroid_x, centroid_y, sigma_x, sigma_y, _total = moments
    w_x = 2.0 * sigma_x
    w_y = 2.0 * sigma_y
    w_avg = 0.5 * (w_x + w_y)

    n_val = _normalize_refractive_index(pilot.current_refractive_index)
    wavelength_mm = pilot.wavelength_um * 1e-3
    r_fit = _fit_phase_curvature(
        state.phase,
        x_grid,
        y_grid,
        intensity,
        centroid_x,
        centroid_y,
        wavelength_mm,
        n_val,
    )

    w0_fit = np.nan
    z0_fit = np.nan
    if np.isfinite(w_avg) and w_avg > 0:
        if np.isinf(r_fit):
            inv_q_real = 0.0
        elif np.isfinite(r_fit):
            inv_q_real = 1.0 / r_fit
        else:
            inv_q_real = None
        if inv_q_real is not None:
            inv_q_imag = -wavelength_mm / (np.pi * n_val * w_avg**2)
            inv_q = inv_q_real + 1j * inv_q_imag
            if np.isfinite(np.real(inv_q)) and np.isfinite(np.imag(inv_q)):
                q_param = 1.0 / inv_q if inv_q != 0 else np.nan + 1j * np.nan
                if np.isfinite(np.real(q_param)) and np.isfinite(np.imag(q_param)):
                    fit_params = PilotBeamParams.from_q_parameter(
                        q_param, pilot.wavelength_um, n_val
                    )
                    w0_fit = float(fit_params.waist_radius_mm)
                    z0_fit = float(fit_params.waist_position_mm)

    m2_x = np.nan
    m2_y = np.nan
    m2_avg = np.nan
    if np.isfinite(w_x) and np.isfinite(w_y) and w_x > 0 and w_y > 0:
        complex_field = sim_amp * np.exp(1j * state.phase)
        fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_field)))
        intensity_ff = np.abs(fft_field) ** 2
        intensity_ff = np.nan_to_num(intensity_ff, nan=0.0, posinf=0.0, neginf=0.0)
        total_ff = float(np.sum(intensity_ff))
        if total_ff > 0:
            df = 1.0 / grid.physical_size_mm
            lambda_medium_mm = wavelength_mm / n_val
            theta_coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * df * lambda_medium_mm
            theta_x_grid, theta_y_grid = np.meshgrid(theta_coords, theta_coords)
            ff_moments = compute_intensity_moments(
                intensity_ff, theta_x_grid, theta_y_grid, mask=None
            )
            if ff_moments is not None:
                _cx, _cy, sigma_tx, sigma_ty, _ = ff_moments
                theta_x = 2.0 * sigma_tx
                theta_y = 2.0 * sigma_ty
                if lambda_medium_mm > 0:
                    m2_x = float(np.pi * w_x * theta_x / lambda_medium_mm)
                    m2_y = float(np.pi * w_y * theta_y / lambda_medium_mm)
                    m2_avg = float(0.5 * (m2_x + m2_y))

    return {
        "w_x_mm": float(w_x),
        "w_y_mm": float(w_y),
        "w_avg_mm": float(w_avg),
        "curvature_radius_mm": float(r_fit),
        "waist_radius_mm": float(w0_fit),
        "waist_position_mm": float(z0_fit),
        "m2_x": float(m2_x),
        "m2_y": float(m2_y),
        "m2_avg": float(m2_avg),
        "m2_pilot": float(_compute_pilot_m2(pilot)),
    }


def compute_metrics(
    sim_amp: NDArray[np.floating],
    amp_residual: NDArray[np.floating],
    phase_residual: NDArray[np.floating],
    grid_sampling,
    mask_threshold: float,
) -> dict[str, float]:
    valid_mask = compute_valid_mask(sim_amp, mask_threshold)
    max_amp = float(np.max(sim_amp)) if sim_amp.size else 0.0
    intensity = sim_amp**2
    pixel_area = float(grid_sampling.sampling_mm) ** 2
    energy = float(np.sum(intensity) * pixel_area) if intensity.size else 0.0

    if intensity.size and np.sum(intensity) > 0:
        coords = (np.arange(grid_sampling.grid_size) - grid_sampling.grid_size // 2) * grid_sampling.sampling_mm
        x_grid, y_grid = np.meshgrid(coords, coords)
        total_intensity = np.sum(intensity)
        centroid_x = float(np.sum(x_grid * intensity) / total_intensity)
        centroid_y = float(np.sum(y_grid * intensity) / total_intensity)
    else:
        centroid_x = 0.0
        centroid_y = 0.0

    if np.any(valid_mask):
        amp_rms = float(np.std(amp_residual[valid_mask]) / max_amp * 100.0) if max_amp > 0 else np.nan
        amp_pv = float(np.ptp(amp_residual[valid_mask]) / max_amp * 100.0) if max_amp > 0 else np.nan
        phase_rms = float(np.sqrt(np.mean(phase_residual[valid_mask] ** 2)) / (2.0 * np.pi))
        phase_pv = float(np.ptp(phase_residual[valid_mask]) / (2.0 * np.pi))
    else:
        amp_rms = np.nan
        amp_pv = np.nan
        phase_rms = np.nan
        phase_pv = np.nan

    return {
        "max_amp": max_amp,
        "energy": energy,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "amp_rms_percent": amp_rms,
        "amp_pv_percent": amp_pv,
        "phase_rms_waves": phase_rms,
        "phase_pv_waves": phase_pv,
    }


def compute_state_moments(
    state: PropagationState, mask_threshold: float = 0.01
) -> dict[str, float]:
    sim_amp = np.asarray(state.amplitude, dtype=float)
    if sim_amp.size == 0:
        return {}
    max_amp = float(np.max(sim_amp))
    if max_amp <= 0:
        return {}
    valid_mask = sim_amp > mask_threshold * max_amp
    grid = state.grid_sampling
    coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
    x_grid, y_grid = np.meshgrid(coords, coords)
    intensity = sim_amp**2
    moments = compute_intensity_moments(intensity, x_grid, y_grid, mask=valid_mask)
    if moments is None:
        return {}
    centroid_x, centroid_y, sigma_x, sigma_y, total = moments
    return {
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "total_intensity": float(total),
    }


__all__ = [
    "compute_log_intensity",
    "compute_valid_mask",
    "compute_intensity_moments",
    "compute_gaussian_fit_and_m2",
    "compute_metrics",
    "compute_state_moments",
    "_fit_phase_curvature_unwrapped",
]


def _fit_phase_curvature_unwrapped(
    phase: NDArray[np.floating],
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    intensity: NDArray[np.floating],
    wavelength_mm: float,
    refractive_index: float,
) -> float:
    """
    Fit curvature by unwrapping phase and performing weighted least squares.
    
    Fits model: phi(x,y) = C0 + C1*x + C2*y + C3*(x^2 + y^2)
    Curvature R = k / (2 * C3)
    
    Returns:
        Radius of curvature (mm). Returns np.inf if flat or invalid.
    """
    try:
        from skimage.restoration import unwrap_phase
    except ImportError:
        return np.nan

    # 1. Masking
    max_int = float(np.max(intensity))
    if max_int <= 0:
        return np.nan
    mask = intensity > 0.1 * max_int
    if np.sum(mask) < 10:
        return np.nan

    # 2. Unwrap Phase
    # We unwrap the whole array, but trust values inside the mask
    # skimage unwrap handles 2D discontinuities well
    phase_unwrapped = unwrap_phase(phase)
    
    # 3. Formulate Least Squares
    # z = A * [1, x, y, r^2]
    # Minimize weighted error: sum( w * (z_data - z_model)^2 )
    
    x_flat = x_grid[mask].flatten()
    y_flat = y_grid[mask].flatten()
    r2_flat = x_flat**2 + y_flat**2
    phi_flat = phase_unwrapped[mask].flatten()
    w_flat = intensity[mask].flatten()
    
    # Normalize weights
    w_flat = w_flat / np.sum(w_flat)
    # Sqrt weights for linear least squares formulation (Ax = b => |Wx - Wy|^2)
    # But np.linalg.lstsq solves |Ax - b|^2.
    # To incorporate weights W, we solve |sqrt(W)(Ax - b)|^2.
    # So multiply A and b by sqrt(W).
    sqrt_w = np.sqrt(w_flat)
    
    # Design Matrix A: [1, x, y, r^2]
    N = len(phi_flat)
    A = np.zeros((N, 4))
    A[:, 0] = 1.0
    A[:, 1] = x_flat
    A[:, 2] = y_flat
    A[:, 3] = r2_flat
    
    # Weighted System
    A_w = A * sqrt_w[:, np.newaxis]
    b_w = phi_flat * sqrt_w
    
    # Solve
    # rcond=None lets numpy decide cutoff
    coeffs, res, rank, s = np.linalg.lstsq(A_w, b_w, rcond=None)
    
    c3 = coeffs[3] # Coeff of r^2
    
    # 4. Convert to Radius
    # phi = k * r^2 / (2R)  => c3 = k / (2R) => R = k / (2*c3)
    # k = 2pi * n / lambda
    if abs(c3) < 1e-15:
        return np.inf

    k = 2.0 * np.pi * refractive_index / wavelength_mm
    radius = k / (2.0 * c3)
    
    return float(radius)

