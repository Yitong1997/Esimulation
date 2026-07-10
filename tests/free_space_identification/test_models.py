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


def test_segment_axis_and_phasor_conventions_are_fixed() -> None:
    by_key = {s.key: s for s in BICONIC_SEGMENTS}
    assert all(
        s.start_convention.side == "after" and s.end_convention.side == "after"
        for s in BICONIC_SEGMENTS
    )
    assert by_key["S07_S08"].start_convention.axis_sign == -1
    assert by_key["S07_S08"].start_convention.conjugate is False
    assert by_key["S12_S13"].start_convention.axis_sign == 1
    assert by_key["S12_S13"].start_convention.conjugate is True
