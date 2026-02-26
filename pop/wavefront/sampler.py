"""Wavefront to rays sampling."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pop.core import GridSampling, OpticalAxisState, PilotBeamParams
from pop.coordinates.transforms import transform_rays_to_global


def sample_rays_from_wavefront(
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    grid_sampling: GridSampling,
    entrance_axis: OpticalAxisState,
    pilot_beam_params: Optional[PilotBeamParams],
    num_rays: int = 1000,
    sampling_sigma: float = 6.0,
) -> Tuple["RealRays", "RealRays"]:

    # ... (skipping unchanged code) ...

    # beam_radius_gauss = 6.0 * sigma_max  # ~99.5% energy radius for Gaussian (w=2sigma, R=1.7w)


    from optiland.rays import RealRays

    n = grid_sampling.grid_size
    dx = grid_sampling.sampling_mm
    dy = grid_sampling.sampling_mm

    if pilot_beam_params is None:
        raise ValueError("pilot_beam_params is required for sampling")
    pilot_phase = pilot_beam_params.compute_phase_grid(
        grid_sampling.grid_size, grid_sampling.physical_size_mm
    )
    sampling_phase = phase - pilot_phase

    coords = (np.arange(n) - n // 2) * dx
    num_rays_target = max(1, int(num_rays))
    stride = max(1, int(n / np.sqrt(num_rays_target)))
    center_idx = n // 2
    offsets = np.arange(0, n // 2, stride)
    valid_offsets = np.unique(np.concatenate([offsets, -offsets]))
    sample_indices = center_idx + valid_offsets

    ix_grid, iy_grid = np.meshgrid(sample_indices, sample_indices)
    ix_flat = ix_grid.flatten()
    iy_flat = iy_grid.flatten()

    sampled_phase = sampling_phase[iy_flat, ix_flat]
    sampled_amp = amplitude[iy_flat, ix_flat]

    x_rays = coords[ix_flat]
    y_rays = coords[iy_flat]

    grad_y, grad_x = np.gradient(sampling_phase, dy, dx)
    wavelength_mm = pilot_beam_params.wavelength_um * 1e-3
    current_n = pilot_beam_params.current_refractive_index
    k = 2.0 * np.pi * current_n / wavelength_mm

    l_rays = grad_x[iy_flat, ix_flat] / k
    m_rays = grad_y[iy_flat, ix_flat] / k

    r_curv = pilot_beam_params.curvature_radius_mm
    if np.isinf(r_curv):
        l_pilot = 0.0
        m_pilot = 0.0
    else:
        l_pilot = x_rays / r_curv
        m_pilot = y_rays / r_curv
    l_rays += l_pilot
    m_rays += m_pilot

    sin_sq = l_rays**2 + m_rays**2
    threshold = 1e-5 * np.max(amplitude)
    # Layer 1: Initially find pixels with sufficient intensity to determine beam extent
    intensity_mask = sampled_amp > threshold
    
    # Calculate the maximum radius of "valid" intensity pixels
    # This defines the effective beam radius
    dist_sq = x_rays**2 + y_rays**2
    
    # --- New Logic: Gaussian Moments for Robust Beam Radius ---
    intensity_grid = amplitude**2
    intensity_sum = np.sum(intensity_grid)
    
    sigma_max = 0.0
    if intensity_sum > 0:
        # Create coordinate grids matching the full intensity grid (before flattening)
        # Note: 'coords' matches the grid definition
        # grid_size and sampling_mm are from grid_sampling
        ny, nx = intensity_grid.shape
        y_indices, x_indices = np.indices((ny, nx))
        
        # Center indices relative to grid center
        # Assuming grid is centered
        cy, cx = ny // 2, nx // 2
        x_grid = (x_indices - cx) * grid_sampling.sampling_mm
        y_grid = (y_indices - cy) * grid_sampling.sampling_mm
        
        # 1. Centroid
        x_centroid = np.sum(x_grid * intensity_grid) / intensity_sum
        y_centroid = np.sum(y_grid * intensity_grid) / intensity_sum
        
        # 2. Second Moments (Variance)
        sigma_x_sq = np.sum((x_grid - x_centroid)**2 * intensity_grid) / intensity_sum
        sigma_y_sq = np.sum((y_grid - y_centroid)**2 * intensity_grid) / intensity_sum
        
        sigma_max = np.sqrt(max(sigma_x_sq, sigma_y_sq))

    # Fallback to threshold method if sigma is suspiciously small (e.g. single pixel or all zeros)
    # or if the beam is extremely non-Gaussian and we want to be safe.
    # But usually sigma logic is robust for main beam.
    
    beam_radius_gauss = sampling_sigma * sigma_max
    
    # Calculate threshold-based radius as a sanity check/fallback
    if np.any(intensity_mask):
        max_valid_radius_sq_thresh = np.max(dist_sq[intensity_mask])
    else:
        max_valid_radius_sq_thresh = 0.0
    
    # Use the larger of the two to be safe, ensuring we capture the beam
    # If the beam is super flat-top, sigma might underestimate slightly compared to threshold edge, 
    # but 3.4 sigma is generous for Gaussian.
    # Let's rely on the Gaussian fit as requested, but ensure it's not zero if there is signal.
    if beam_radius_gauss < 1e-9 and max_valid_radius_sq_thresh > 0:
         effective_beam_radius_sq = max_valid_radius_sq_thresh
    else:
         effective_beam_radius_sq = beam_radius_gauss**2

    # Physical grid limit
    half_size = grid_sampling.physical_size_mm / 2.0
    max_physical_sq = (0.99 * half_size) ** 2
    
    # Effective mask is circle of beam radius AND circle of physical grid limit
    effective_radius_sq = min(effective_beam_radius_sq, max_physical_sq)
    
    # Layer 3: Direction validity
    sin_sq = l_rays**2 + m_rays**2
    
    # Final mask: Inside beam circle AND valid direction
    # We include pixels < threshold if they are within the beam radius to avoid holes
    valid_mask = (dist_sq <= effective_radius_sq) & (sin_sq < 1.0)

    # Always include center
    center_ray_mask = dist_sq < 1e-10
    valid_mask = valid_mask | center_ray_mask

    if not np.any(valid_mask):
        raise ValueError("No valid rays after sampling")

    x_final = x_rays[valid_mask]
    y_final = y_rays[valid_mask]
    l_final = l_rays[valid_mask]
    m_final = m_rays[valid_mask]
    n_final = np.sqrt(1.0 - (l_final**2 + m_final**2))

    local_rays = RealRays(
        x=x_final,
        y=y_final,
        z=np.zeros_like(x_final),
        L=l_final,
        M=m_final,
        N=n_final,
        wavelength=pilot_beam_params.wavelength_um,
        intensity=sampled_amp[valid_mask] ** 2,
    )
    # Preserve local rays before globalizing to avoid mixing coordinate frames.
    # CoordinateSystem.globalize() mutates in place.
    import copy

    global_rays = copy.deepcopy(local_rays)
    global_rays = transform_rays_to_global(global_rays, entrance_axis)

    full_phase = phase[iy_flat, ix_flat][valid_mask]
    initial_opd_mm = full_phase * wavelength_mm / (2.0 * np.pi)
    global_rays.opd = initial_opd_mm
    global_rays.i = sampled_amp[valid_mask] ** 2

    return local_rays, global_rays
