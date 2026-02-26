"""Ideal propagation logic for simplified components."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pop.core import OpticalAxisState, PropagationState, GridSampling
from pop.propagation.free_space import _sync_proper_gaussian_params

def propagate_ideal_mirror(
    state: PropagationState,
    exit_axis: OpticalAxisState,
    target_surface_index: int,
    focal_length_mm: Optional[float] = None,
    added_phase_map: Optional[NDArray[np.floating]] = None,
    debug: bool = False,
) -> PropagationState:
    """
    Propagate through an ideal mirror (coordinate transform + optional phase).
    
    This function bypasses local ray tracing. It assumes:
    1. The coordinate system is transformed according to `exit_axis` (typically reflection).
    2. The wavefront amplitude and phase arrays are preserved (no resampling/distortion).
       - Note: This implicitly assumes the local coordinate frame transforms in a way
         that matches the beam's geometric reflection.
    3. Optional 'thin lens' phase can be added.
    """
    
    # 1. Update Pilot Beam (ABCD Matrix for Mirror/Lens)
    old_pilot = state.pilot_beam_params
    new_pilot = old_pilot  # Start with copy (immutable dataclass, but we'll create new one via methods)
    
    # If it acts as a curved mirror (ideal thin lens phase), apply it to pilot
    if focal_length_mm is not None and not np.isinf(focal_length_mm):
         new_pilot = new_pilot.apply_lens(focal_length_mm)
         
    # 2. Phase Addition (Thin Lens / Phase Map)
    current_phase = state.phase
    if added_phase_map is not None:
        current_phase = current_phase + added_phase_map
        
    if focal_length_mm is not None and not np.isinf(focal_length_mm):
        grid = state.grid_sampling
        wavelength_mm = state.pilot_beam_params.wavelength_um * 1e-3
        k = 2 * np.pi / wavelength_mm
        
        # Calculate r^2 grid
        coords = (np.arange(grid.grid_size) - grid.grid_size // 2) * grid.sampling_mm
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_sq_mm = x_grid**2 + y_grid**2
        
        # Thin lens phase: -k * r^2 / (2f)
        phase_lens = -k * r_sq_mm / (2.0 * focal_length_mm)
        current_phase = current_phase + phase_lens

    # 3. Construct New State
    # Note: proper_wfo needs synchronization
    _sync_proper_gaussian_params(state.proper_wfo, new_pilot, update_reference_surface=True)
    
    messages = list(state.messages)
    messages.append(f"Propagated ideal mirror at surface {target_surface_index}")

    return replace(
        state,
        surface_index=target_surface_index,
        position="exit", # It's effectively immediate transition
        amplitude=state.amplitude, # Preserved
        phase=current_phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=exit_axis,
        propagation_algorithm="Ideal Mirror",
        messages=messages
    )
