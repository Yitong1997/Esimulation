from __future__ import annotations

import numpy as np
import pytest

from pop.core import OpticalAxisState
from pop.core import GridSampling, PilotBeamParams, PropagationState
from pop.propagation.ideal import propagate_ideal_mirror
from pop.propagation.free_space import _compute_signed_distance


def _axis(*, path_length: float, coordinate_z_axis: np.ndarray | None) -> OpticalAxisState:
    return OpticalAxisState(
        position=np.array([0.0, 0.0, path_length]),
        direction=np.array([0.0, 0.0, 1.0]),
        frame=np.eye(3),
        coord_sys=None,
        path_length=path_length,
        coordinate_z_axis=coordinate_z_axis,
    )


def test_local_z_axis_sign_does_not_reverse_physical_proper_distance() -> None:
    source = _axis(path_length=10.0, coordinate_z_axis=np.array([0.0, 0.0, -1.0]))
    target = _axis(path_length=20.0, coordinate_z_axis=np.array([0.0, 0.0, -1.0]))

    assert source.coordinate_axis_alignment == pytest.approx(-1.0)
    assert source.coordinate_axis_sign == -1
    assert _compute_signed_distance(source, target) == pytest.approx(10.0)


def test_missing_local_z_axis_keeps_forward_compatibility() -> None:
    source = _axis(path_length=10.0, coordinate_z_axis=None)
    target = _axis(path_length=20.0, coordinate_z_axis=None)

    assert source.coordinate_axis_sign == 1
    assert _compute_signed_distance(source, target) == pytest.approx(10.0)


def test_coordinate_sign_change_does_not_change_physical_path_distance() -> None:
    source = _axis(path_length=10.0, coordinate_z_axis=np.array([0.0, 0.0, 1.0]))
    target = _axis(path_length=20.0, coordinate_z_axis=np.array([0.0, 0.0, -1.0]))

    assert _compute_signed_distance(source, target) == pytest.approx(10.0)


def test_nearly_transverse_local_z_axis_is_rejected() -> None:
    source = _axis(path_length=10.0, coordinate_z_axis=np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="orthogonal"):
        _ = source.coordinate_axis_sign


def test_ideal_mirror_maps_a_non_symmetric_field_and_synchronizes_wfarr() -> None:
    import proper

    grid = GridSampling.create(grid_size=5, physical_size_mm=5.0)
    pilot = PilotBeamParams.from_gaussian_source(
        wavelength_um=1.0,
        w0_mm=1.0,
        z0_mm=0.0,
    )
    wfo = proper.prop_begin(5.0e-3, 1.0e-6, 5, 1.0)
    wfo.w0 = pilot.waist_radius_mm * 1e-3
    wfo.z_Rayleigh = pilot.rayleigh_length_mm * 1e-3
    wfo.z_w0 = 0.0
    wfo.reference_surface = "PLANAR"
    wfo.beam_type_old = "INSIDE_"

    field = np.arange(25, dtype=float).reshape(5, 5) + 1j * np.arange(25, dtype=float).reshape(5, 5)
    wfo.wfarr = proper.prop_shift_center(field)
    state = PropagationState(
        surface_index=1,
        position="entrance",
        amplitude=np.abs(field),
        phase=np.angle(field),
        pilot_beam_params=pilot,
        optical_axis_state=_axis(path_length=0.0, coordinate_z_axis=np.array([0.0, 0.0, 1.0])),
        grid_sampling=grid,
        proper_wfo=wfo,
    )
    entrance_axis = state.optical_axis_state
    exit_axis = OpticalAxisState(
        position=np.zeros(3),
        direction=np.array([0.0, 0.0, -1.0]),
        frame=np.diag([1.0, -1.0, -1.0]),
        coord_sys=None,
        path_length=0.0,
        coordinate_z_axis=np.array([0.0, 0.0, 1.0]),
    )

    result = propagate_ideal_mirror(
        state,
        entrance_axis=entrance_axis,
        exit_axis=exit_axis,
        target_surface_index=2,
    )

    expected = np.flip(field, axis=0)
    np.testing.assert_allclose(result.get_complex_amplitude(), expected, atol=1e-12)
    np.testing.assert_allclose(
        result.proper_wfo.wfarr,
        proper.prop_shift_center(expected),
        atol=1e-12,
    )


def test_ideal_mirror_transverse_map_is_orthogonal_for_a_tilted_reflection() -> None:
    from pop.propagation.ideal import _transform_transverse_field

    grid = GridSampling.create(grid_size=5, physical_size_mm=5.0)
    entrance = _axis(path_length=0.0, coordinate_z_axis=np.array([0.0, 0.0, 1.0]))
    entrance.frame = np.eye(3)
    angle = np.deg2rad(30.0)
    exit_direction = np.array([0.0, np.sin(angle), -np.cos(angle)])
    exit_frame = np.column_stack(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, np.cos(angle), np.sin(angle)]),
            exit_direction,
        ]
    )
    exit_axis = OpticalAxisState(
        position=np.zeros(3),
        direction=exit_direction,
        frame=exit_frame,
        coord_sys=None,
        path_length=0.0,
        coordinate_z_axis=np.array([0.0, 0.0, 1.0]),
    )
    field = np.arange(25, dtype=float).reshape(5, 5).astype(np.complex128)

    mapped, edge_fraction = _transform_transverse_field(
        field,
        entrance_axis=entrance,
        exit_axis=exit_axis,
        grid=grid,
    )

    assert mapped.shape == field.shape
    assert 0.0 <= edge_fraction < 1.0


def test_ideal_mirror_signed_permutation_preserves_even_grid_support() -> None:
    grid = GridSampling.create(grid_size=4, physical_size_mm=4.0)
    entrance = _axis(path_length=0.0, coordinate_z_axis=np.array([0.0, 0.0, 1.0]))
    exit_axis = OpticalAxisState(
        position=np.zeros(3),
        direction=np.array([0.0, 0.0, -1.0]),
        frame=np.diag([1.0, -1.0, -1.0]),
        coord_sys=None,
        path_length=0.0,
        coordinate_z_axis=np.array([0.0, 0.0, 1.0]),
    )
    field = np.arange(16, dtype=float).reshape(4, 4).astype(np.complex128)

    from pop.propagation.ideal import _transform_transverse_field

    mapped, edge_fraction = _transform_transverse_field(
        field,
        entrance_axis=entrance,
        exit_axis=exit_axis,
        grid=grid,
    )

    np.testing.assert_allclose(mapped, field.take([0, 3, 2, 1], axis=0))
    assert edge_fraction == pytest.approx(0.0)
