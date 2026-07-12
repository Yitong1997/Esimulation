from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from pop.io.zbf import ZbfField, read_zbf, write_zbf, zbf_reference_phase


def _write_minimal_zbf(
    path: Path,
    ex: np.ndarray,
    *,
    dx: float = 0.25,
    dy: float = 0.25,
    zx: float = 2.0,
    zy: float = 2.0,
    rayleigh: float = 10.0,
    waist: float = 1.5,
    wavelength: float = 0.01,
    index: float = 1.0,
    polarized: bool = False,
) -> None:
    ny, nx = ex.shape
    ints = [1, nx, ny, int(polarized), 0, 0, 0, 0, 0]
    dbls = [
        dx,
        dy,
        zx,
        rayleigh,
        waist,
        zy,
        rayleigh,
        waist,
        wavelength,
        index,
        0.0,
        0.0,
        *([0.0] * 8),
    ]
    flat = ex.reshape(-1)
    interleaved = np.empty(2 * flat.size, dtype="<f8")
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    with path.open("wb") as f:
        f.write(struct.pack("<9i", *ints))
        f.write(struct.pack("<20d", *dbls))
        f.write(interleaved.tobytes())
        if polarized:
            f.write(interleaved.tobytes())


def test_read_zbf_preserves_complex_field_and_header(tmp_path: Path) -> None:
    ex = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128
    )
    path = tmp_path / "known.ZBF"
    _write_minimal_zbf(path, ex, dx=0.125, dy=0.25)

    zbf = read_zbf(path)

    assert isinstance(zbf, ZbfField)
    assert zbf.nx == 2
    assert zbf.ny == 2
    assert zbf.dx == 0.125
    assert zbf.dy == 0.25
    assert zbf.wavelength == 0.01
    np.testing.assert_array_equal(zbf.ex, ex)
    np.testing.assert_allclose(zbf.amplitude, np.abs(ex))


def test_write_zbf_round_trips(tmp_path: Path) -> None:
    ex = np.array(
        [[1 + 0j, 0 + 1j], [2 - 1j, -3 + 0.5j]], dtype=np.complex128
    )
    zbf = ZbfField(
        path=None,
        version=1,
        nx=2,
        ny=2,
        is_polarized=0,
        units=0,
        dx=0.5,
        dy=0.5,
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
    path = tmp_path / "roundtrip.ZBF"

    write_zbf(path, zbf)
    actual = read_zbf(path)

    np.testing.assert_array_equal(actual.ex, ex)
    assert actual.dx == 0.5


def test_reference_phase_uses_spherical_header_metadata(tmp_path: Path) -> None:
    ex = np.ones((3, 3), dtype=np.complex128)
    path = tmp_path / "ref.ZBF"
    _write_minimal_zbf(path, ex, dx=0.5, dy=0.5, zx=4.0, zy=4.0, rayleigh=2.0)
    zbf = read_zbf(path)

    phase = zbf_reference_phase(zbf)

    xg, yg = np.meshgrid(zbf.x_coords, zbf.y_coords)
    expected = (2.0 * np.pi * zbf.index / zbf.wavelength) * (
        np.sqrt(zbf.zx**2 + xg**2 + yg**2) - abs(zbf.zx)
    )
    np.testing.assert_allclose(phase, expected)


def test_read_zbf_marks_polarized_input(tmp_path: Path) -> None:
    path = tmp_path / "pol.ZBF"
    _write_minimal_zbf(path, np.ones((2, 2), dtype=np.complex128), polarized=True)

    zbf = read_zbf(path)

    assert zbf.is_polarized == 1
    assert zbf.ey is not None
