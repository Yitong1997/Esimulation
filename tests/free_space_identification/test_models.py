import numpy as np
import pytest

from sandbox.free_space_algorithm_identification.biconic_case import BICONIC_SEGMENTS
from sandbox.free_space_algorithm_identification.models import PointField2D, UniformGrid2D


def test_registry_contains_only_true_free_space_pairs() -> None:
    assert [(s.start_surface, s.end_surface) for s in BICONIC_SEGMENTS] == [
        (7, 8),
        (12, 13),
        (13, 14),
    ]
    assert [s.branch for s in BICONIC_SEGMENTS] == ["OO", "OI", "IO"]


def test_even_grid_uses_the_prevalidated_sample_at_zero_convention() -> None:
    grid = UniformGrid2D.centered(nx=4, ny=4, dx_mm=0.5, dy_mm=0.25)
    assert grid.x_mm.tolist() == [-1.0, -0.5, 0.0, 0.5]
    assert grid.y_mm.tolist() == [-0.5, -0.25, 0.0, 0.25]

    field = PointField2D(np.ones((4, 4), dtype=np.complex128), grid)
    with pytest.raises(ValueError):
        grid.x_mm.setflags(write=True)
    with pytest.raises(ValueError):
        field.values.setflags(write=True)


def test_segment_side_and_axis_conventions_are_fixed_without_phasor_switch() -> None:
    conventions = [
        convention
        for segment in BICONIC_SEGMENTS
        for convention in (segment.start_convention, segment.end_convention)
    ]
    by_surface = {convention.surface: convention for convention in conventions}

    assert {
        surface: (convention.side, convention.axis_sign)
        for surface, convention in by_surface.items()
    } == {
        7: ("after", -1),
        8: ("after", -1),
        12: ("after", 1),
        13: ("after", 1),
        14: ("after", 1),
    }
    assert all(not hasattr(convention, "conjugate") for convention in conventions)


def test_real_coordinate_pullback_commutes_with_but_is_not_conjugation() -> None:
    field = np.array(
        [
            [1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j],
            [7.0 - 8.0j, -9.0 - 10.0j, 11.0 + 12.0j],
        ],
        dtype=np.complex128,
    )

    pulled_back = field[:, ::-1]
    np.testing.assert_array_equal(np.conj(pulled_back), np.conj(field)[:, ::-1])
    assert not np.array_equal(pulled_back, np.conj(field))
