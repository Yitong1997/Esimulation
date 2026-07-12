"""Zemax Beam File (ZBF) read/write helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_HEADER_INTS = 9
N_HEADER_DBLS = 20


@dataclass
class ZbfField:
    """ZBF complex field and header metadata."""

    path: Path | None
    version: int
    nx: int
    ny: int
    is_polarized: int
    units: int
    dx: float
    dy: float
    zx: float
    rx: float
    wx: float
    zy: float
    ry: float
    wy: float
    wavelength: float
    index: float
    receiver_efficiency: float
    system_efficiency: float
    ex: np.ndarray
    ey: np.ndarray | None = None

    @property
    def x_coords(self) -> np.ndarray:
        return (np.arange(self.nx) - self.nx // 2) * self.dx

    @property
    def y_coords(self) -> np.ndarray:
        return (np.arange(self.ny) - self.ny // 2) * self.dy

    @property
    def amplitude(self) -> np.ndarray:
        amp_sq = np.abs(self.ex) ** 2
        if self.ey is not None:
            amp_sq = amp_sq + np.abs(self.ey) ** 2
        return np.sqrt(amp_sq)

    @property
    def physical_field(self) -> np.ndarray:
        return self.ex * np.exp(-1j * zbf_reference_phase(self))

    @property
    def reference_relative_field_pop_convention(self) -> np.ndarray:
        return zbf_reference_relative_field_pop_convention(self)

    @property
    def physical_field_pop_convention(self) -> np.ndarray:
        return zbf_physical_field_pop_convention(self)


def zemax_zbf_field_to_pop_convention(field: np.ndarray) -> np.ndarray:
    """Convert raw Zemax ZBF samples to the POP complex-field convention."""

    return np.conj(np.asarray(field, dtype=np.complex128))


def pop_field_to_zemax_zbf_convention(field: np.ndarray) -> np.ndarray:
    """Convert a POP complex field to the raw Zemax ZBF convention."""

    return np.conj(np.asarray(field, dtype=np.complex128))


def zbf_reference_relative_field_pop_convention(zbf: ZbfField) -> np.ndarray:
    """Return raw ZBF ``Ex`` as a POP reference-relative field."""

    return zemax_zbf_field_to_pop_convention(zbf.ex)


def zbf_physical_field_pop_convention(zbf: ZbfField) -> np.ndarray:
    """Return the ZBF physical field in the POP phasor convention."""

    return zbf_physical_field_pop_convention_for_axis(zbf, axis_sign=1.0)


def zbf_physical_field_pop_convention_for_axis(
    zbf: ZbfField,
    *,
    axis_sign: float,
) -> np.ndarray:
    """Lift the ZBF residual field with its signed local-axis reference phase."""

    sign = 1.0 if float(axis_sign) >= 0.0 else -1.0
    return zbf_reference_relative_field_pop_convention(zbf) * np.exp(
        1j * sign * zbf_reference_phase(zbf)
    )


def read_zbf(path: str | Path) -> ZbfField:
    """Read a Zemax Beam File from disk."""

    zbf_path = Path(path)
    with zbf_path.open("rb") as f:
        ints_raw = f.read(N_HEADER_INTS * 4)
        if len(ints_raw) != N_HEADER_INTS * 4:
            raise ValueError(f"Incomplete ZBF integer header: {zbf_path}")
        ints = struct.unpack("<9i", ints_raw)
        version, nx, ny, is_polarized, units = ints[:5]
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Invalid ZBF dimensions: nx={nx}, ny={ny}")

        dbls_raw = f.read(N_HEADER_DBLS * 8)
        if len(dbls_raw) != N_HEADER_DBLS * 8:
            raise ValueError(f"Incomplete ZBF double header: {zbf_path}")
        dbls = struct.unpack("<20d", dbls_raw)

        ex = _read_complex_grid(f, nx, ny, zbf_path, "Ex")
        ey = _read_complex_grid(f, nx, ny, zbf_path, "Ey") if is_polarized else None

    return ZbfField(
        path=zbf_path,
        version=version,
        nx=nx,
        ny=ny,
        is_polarized=is_polarized,
        units=units,
        dx=float(dbls[0]),
        dy=float(dbls[1]),
        zx=float(dbls[2]),
        rx=float(dbls[3]),
        wx=float(dbls[4]),
        zy=float(dbls[5]),
        ry=float(dbls[6]),
        wy=float(dbls[7]),
        wavelength=float(dbls[8]),
        index=float(dbls[9]),
        receiver_efficiency=float(dbls[10]),
        system_efficiency=float(dbls[11]),
        ex=ex,
        ey=ey,
    )


def write_zbf(path: str | Path, zbf: ZbfField) -> None:
    """Write a Zemax Beam File to disk."""

    zbf_path = Path(path)
    zbf_path.parent.mkdir(parents=True, exist_ok=True)
    if zbf.ex.shape != (zbf.ny, zbf.nx):
        raise ValueError(f"Ex shape {zbf.ex.shape} does not match {(zbf.ny, zbf.nx)}")
    with zbf_path.open("wb") as f:
        f.write(
            struct.pack(
                "<9i",
                zbf.version,
                zbf.nx,
                zbf.ny,
                zbf.is_polarized,
                zbf.units,
                0,
                0,
                0,
                0,
            )
        )
        f.write(
            struct.pack(
                "<20d",
                zbf.dx,
                zbf.dy,
                zbf.zx,
                zbf.rx,
                zbf.wx,
                zbf.zy,
                zbf.ry,
                zbf.wy,
                zbf.wavelength,
                zbf.index,
                zbf.receiver_efficiency,
                zbf.system_efficiency,
                *([0.0] * 8),
            )
        )
        _write_complex_grid(f, zbf.ex)
        if zbf.is_polarized:
            if zbf.ey is None:
                raise ValueError("Polarized ZBF requires ey")
            _write_complex_grid(f, zbf.ey)


def zbf_reference_phase(zbf: ZbfField) -> np.ndarray:
    """Compute the validated reference phase represented by the ZBF header."""

    x_grid, y_grid = np.meshgrid(zbf.x_coords, zbf.y_coords)
    phase = np.zeros((zbf.ny, zbf.nx), dtype=np.float64)
    if zbf.wavelength <= 0:
        return phase

    rcx = _zbf_reference_radius(zbf.zx, zbf.rx)
    rcy = _zbf_reference_radius(zbf.zy, zbf.ry)
    k = 2.0 * np.pi * zbf.index / zbf.wavelength
    return k * _validated_reference_opd(x_grid**2, y_grid**2, rcx, rcy)


def _read_complex_grid(f, nx: int, ny: int, path: Path, label: str) -> np.ndarray:
    raw = f.read(nx * ny * 2 * 8)
    if len(raw) != nx * ny * 2 * 8:
        raise ValueError(f"Incomplete ZBF {label} data: {path}")
    pairs = np.frombuffer(raw, dtype="<f8")
    return (pairs[0::2] + 1j * pairs[1::2]).reshape(ny, nx)


def _write_complex_grid(f, field: np.ndarray) -> None:
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    pairs = np.empty(2 * flat.size, dtype="<f8")
    pairs[0::2] = flat.real
    pairs[1::2] = flat.imag
    f.write(pairs.tobytes())


def _zbf_reference_radius(waist_position: float, rayleigh_range: float) -> float:
    z = float(waist_position)
    zr = abs(float(rayleigh_range))
    if abs(z) < 1e-15 or abs(z) < zr:
        return float("inf")
    return z


def _signed_spherical_opd(transverse_sq: np.ndarray, radius: float) -> np.ndarray:
    if not np.isfinite(radius) or abs(radius) < 1e-15:
        return np.zeros_like(transverse_sq, dtype=np.float64)
    abs_radius = abs(radius)
    return np.sign(radius) * (np.sqrt(abs_radius**2 + transverse_sq) - abs_radius)


def _validated_reference_opd(
    x_sq: np.ndarray,
    y_sq: np.ndarray,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    finite_x = np.isfinite(radius_x)
    finite_y = np.isfinite(radius_y)
    if finite_x and finite_y:
        if not np.isclose(radius_x, radius_y, rtol=1e-12, atol=1e-15):
            raise NotImplementedError(
                "Unequal X/Y ZBF reference radii are not validated; use the "
                "reference-relative field or an explicit plane-reference export."
            )
        return _signed_spherical_opd(x_sq + y_sq, radius_x)
    if finite_x:
        return _signed_spherical_opd(x_sq, radius_x)
    if finite_y:
        return _signed_spherical_opd(y_sq, radius_y)
    return np.zeros_like(x_sq, dtype=np.float64)
