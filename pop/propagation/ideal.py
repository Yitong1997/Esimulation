"""Ideal propagation logic for simplified components."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pop.core import OpticalAxisState, PropagationState, GridSampling
from pop.propagation.free_space import _sync_proper_gaussian_params
from pop.reference_frames import snapshot_reference_frame


def _transform_transverse_field(
    field: NDArray[np.complexfloating],
    *,
    entrance_axis: OpticalAxisState,
    exit_axis: OpticalAxisState,
    grid: GridSampling,
) -> tuple[NDArray[np.complexfloating], float]:
    """Map a complex field from the entrance to the exit transverse basis."""

    entrance_direction = np.asarray(entrance_axis.direction, dtype=float)
    exit_direction = np.asarray(exit_axis.direction, dtype=float)
    direction_delta = entrance_direction - exit_direction
    delta_norm = float(np.linalg.norm(direction_delta))
    if delta_norm < 1e-12:
        mirror_transform = np.eye(3, dtype=float)
    else:
        # The Householder reflection maps the incident chief direction onto
        # the outgoing chief direction.  It is the geometric transformation
        # of transverse points at an ideal planar mirror; using only the dot
        # product of the two local frames would project the field and cause a
        # spurious cos(incidence) compression.
        normal = direction_delta / delta_norm
        mirror_transform = np.eye(3, dtype=float) - 2.0 * np.outer(normal, normal)
    if not np.allclose(
        mirror_transform @ entrance_direction,
        exit_direction,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("mirror direction transform is inconsistent with the axis states")

    entrance_basis = np.asarray(entrance_axis.frame, dtype=float)[:, :2]
    exit_basis = np.asarray(exit_axis.frame, dtype=float)[:, :2]
    basis_change = exit_basis.T @ mirror_transform @ entrance_basis
    if not np.allclose(
        basis_change.T @ basis_change,
        np.eye(2),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("mirror transverse basis change is not orthogonal")

    size = int(grid.grid_size)
    coordinates = (np.arange(size, dtype=float) - size // 2) * grid.sampling_mm
    x_out, y_out = np.meshgrid(coordinates, coordinates)
    output_coordinates = np.stack((x_out, y_out), axis=0).reshape(2, -1)
    input_coordinates = basis_change.T @ output_coordinates
    input_x = input_coordinates[0].reshape(size, size) / grid.sampling_mm + size // 2
    input_y = input_coordinates[1].reshape(size, size) / grid.sampling_mm + size // 2

    # Signed permutations are exact discrete grid operations.  The PROPER
    # grid is centred at index ``size // 2``; for an even grid the reflected
    # endpoint is therefore represented by the periodic FFT index rather than
    # by clipping one edge of the array.
    rounded_basis_change = np.rint(basis_change)
    is_signed_permutation = (
        np.allclose(basis_change, rounded_basis_change, rtol=0.0, atol=1e-9)
        and np.all(np.sum(np.abs(rounded_basis_change), axis=0) == 1)
        and np.all(np.sum(np.abs(rounded_basis_change), axis=1) == 1)
    )
    if is_signed_permutation:
        input_x = np.mod(np.rint(input_x).astype(int), size)
        input_y = np.mod(np.rint(input_y).astype(int), size)
        mapped = np.asarray(field)[input_y, input_x]
        return np.asarray(mapped, dtype=np.complex128), 0.0

    valid = (
        (input_x >= 0.0)
        & (input_x <= size - 1.0)
        & (input_y >= 0.0)
        & (input_y <= size - 1.0)
    )
    edge_fraction = 1.0 - float(np.mean(valid))

    from scipy.ndimage import map_coordinates

    coordinates_for_sampling = np.stack((input_y, input_x), axis=0)
    real = map_coordinates(
        np.asarray(field.real, dtype=float),
        coordinates_for_sampling,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    imag = map_coordinates(
        np.asarray(field.imag, dtype=float),
        coordinates_for_sampling,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.asarray(real + 1j * imag, dtype=np.complex128), edge_fraction

def propagate_ideal_mirror(
    state: PropagationState,
    entrance_axis: OpticalAxisState,
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
         
    # 2. Transform the complex field into the outgoing transverse basis.
    mapped_field, edge_fraction = _transform_transverse_field(
        state.get_complex_amplitude(),
        entrance_axis=entrance_axis,
        exit_axis=exit_axis,
        grid=state.grid_sampling,
    )

    # 3. Phase Addition (Thin Lens / Phase Map)
    if added_phase_map is not None:
        if added_phase_map.shape != mapped_field.shape:
            raise ValueError("mirror phase map shape does not match the field grid")
        mapped_field = mapped_field * np.exp(1j * added_phase_map)
        
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
        mapped_field = mapped_field * np.exp(1j * phase_lens)

    # 4. Construct New State.  Rebuild wfarr from the transformed physical
    # field so the internal PROPER array cannot retain the pre-mirror field.
    _sync_proper_gaussian_params(
        state.proper_wfo,
        new_pilot,
        update_reference_surface=True,
        skip_wfarr_transition=True,
    )
    from pop.propagation.element import _amplitude_phase_to_proper

    mapped_amplitude = np.abs(mapped_field)
    mapped_phase = np.angle(mapped_field)
    _amplitude_phase_to_proper(
        state.proper_wfo,
        mapped_amplitude,
        mapped_phase,
        state.grid_sampling,
        trace_context={"propagation_type": "ideal_mirror"},
    )
    reference_relative_field, reference_phase = snapshot_reference_frame(
        state.proper_wfo,
        mapped_phase,
    )

    messages = list(state.messages)
    messages.append(f"Propagated ideal mirror at surface {target_surface_index}")
    if edge_fraction > 1e-12:
        messages.append(
            f"Mirror transverse remap clipped {edge_fraction:.3e} of the output grid"
        )

    return replace(
        state,
        surface_index=target_surface_index,
        position="exit", # It's effectively immediate transition
        amplitude=mapped_amplitude,
        phase=mapped_phase,
        pilot_beam_params=new_pilot,
        optical_axis_state=exit_axis,
        propagation_algorithm="Ideal Mirror",
        messages=messages,
        reference_relative_field=reference_relative_field,
        reference_phase=reference_phase,
    )
