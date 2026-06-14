from __future__ import annotations

from pathlib import Path

import numpy as np
import proper
import pytest

from pop import ZbfSource
from pop.io.zbf import ZbfField, write_zbf, zbf_reference_phase


def _field(
    ex: np.ndarray, *, zx: float = 0.0, rayleigh: float = 10.0, waist: float = 2.0
) -> ZbfField:
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
        zx=zx,
        rx=rayleigh,
        wx=waist,
        zy=zx,
        ry=rayleigh,
        wy=waist,
        wavelength=0.01,
        index=1.0,
        receiver_efficiency=0.0,
        system_efficiency=0.0,
        ex=ex,
        ey=None,
    )


def test_zbf_source_reference_relative_writes_ex_to_proper_wfarr(
    tmp_path: Path,
) -> None:
    ex = np.array(
        [[1 + 0j, 0.5 + 0.25j], [0.25 - 0.5j, -1 + 0j]],
        dtype=np.complex128,
    )
    zbf = _field(ex, zx=0.0)
    path = tmp_path / "input.ZBF"
    write_zbf(path, zbf)

    source = ZbfSource(path, reference_mode="reference_relative")
    amplitude, phase, pilot, wfo = source.create_initial_wavefront()

    np.testing.assert_allclose(amplitude, np.abs(ex))
    np.testing.assert_allclose(phase, np.angle(ex))
    np.testing.assert_allclose(proper.prop_shift_center(wfo.wfarr), ex)
    assert pilot.wavelength_um == pytest.approx(10.0)
    assert pilot.waist_radius_mm == pytest.approx(2.0)


def test_zbf_source_physical_mode_returns_physical_phase(tmp_path: Path) -> None:
    ex = np.ones((3, 3), dtype=np.complex128)
    zbf = _field(ex, zx=2.0, rayleigh=4.0)
    path = tmp_path / "physical.ZBF"
    write_zbf(path, zbf)

    source = ZbfSource(path, reference_mode="physical")
    amplitude, phase, _pilot, _wfo = source.create_initial_wavefront()

    np.testing.assert_allclose(amplitude, np.ones((3, 3)))
    np.testing.assert_allclose(
        np.exp(1j * phase), np.exp(1j * zbf_reference_phase(zbf))
    )


def test_zbf_source_rejects_polarized_input_by_default(tmp_path: Path) -> None:
    zbf = _field(np.ones((2, 2), dtype=np.complex128))
    zbf.is_polarized = 1
    zbf.ey = zbf.ex.copy()
    path = tmp_path / "polarized.ZBF"
    write_zbf(path, zbf)

    with pytest.raises(ValueError, match="polarized"):
        ZbfSource(path).create_initial_wavefront()


def test_zbf_source_rejects_astigmatic_header_by_default(tmp_path: Path) -> None:
    zbf = _field(np.ones((2, 2), dtype=np.complex128))
    zbf.wy = 3.0
    path = tmp_path / "astigmatic.ZBF"
    write_zbf(path, zbf)

    with pytest.raises(ValueError, match="astigmatic"):
        ZbfSource(path).create_initial_wavefront()
