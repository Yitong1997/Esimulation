from __future__ import annotations

import numpy as np

from pop.io.zbf import ZbfField
from sandbox.zemax_pop_benchmark.comparison import compare_pop_state_to_zbf


class Dummy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _zbf(ex: np.ndarray) -> ZbfField:
    ny, nx = ex.shape
    return ZbfField(
        path=None,
        version=1,
        nx=nx,
        ny=ny,
        is_polarized=0,
        units=0,
        dx=0.25,
        dy=0.25,
        zx=0.0,
        rx=0.0,
        wx=1.0,
        zy=0.0,
        ry=0.0,
        wy=1.0,
        wavelength=0.01,
        index=1.0,
        receiver_efficiency=0.0,
        system_efficiency=0.0,
        ex=ex,
        ey=None,
    )


def test_compare_reports_zero_residual_when_fields_match() -> None:
    ex = np.array([[1 + 0j, 1j], [-1 + 0j, 1 - 1j]], dtype=np.complex128)
    state = Dummy(
        surface_index=3,
        position="entrance",
        proper_wfo=None,
        pilot_beam_params=Dummy(wavelength_um=10.0),
        grid_sampling=Dummy(sampling_mm=0.25),
    )

    result = compare_pop_state_to_zbf(
        state=state,
        pop_reference_relative=np.conj(ex),
        pop_reference_phase=np.zeros(ex.shape),
        zbf=_zbf(ex),
        surface_name="S3",
        mask_threshold=0.0,
    )

    assert result.summary["phase_rms_waves"] < 1e-12
    assert result.summary["relative_intensity_rms"] < 1e-12


def test_compare_removes_phase_piston() -> None:
    ex = np.ones((2, 2), dtype=np.complex128)
    result = compare_pop_state_to_zbf(
        state=Dummy(
            surface_index=4,
            position="exit",
            pilot_beam_params=Dummy(wavelength_um=10.0),
            grid_sampling=Dummy(sampling_mm=0.25),
        ),
        pop_reference_relative=np.exp(1j * 0.25) * ex,
        pop_reference_phase=np.zeros(ex.shape),
        zbf=_zbf(ex),
        surface_name="S4",
        mask_threshold=0.0,
    )

    assert abs(result.summary["phase_piston_rad"] - 0.25) < 1e-12
    assert result.summary["phase_rms_waves"] < 1e-12
