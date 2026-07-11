"""Paraxial free-space operators with explicit physical normalization.

All public fields are continuous point values on millimetre grids.  Scaled
Fourier transforms act on cell-energy samples so that their unitary discrete
norm represents the physical ``sum |U|^2 dx dy`` quadrature.  Public physical
propagators restore the known ``exp(+i k d)`` axial carrier exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fourier import evaluate_field_fourier_czt
from .models import PointField2D, UniformGrid2D


def _readonly_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128, order="C")
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("cell-energy samples must be a finite two-dimensional array")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.complex128).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


def _medium_parameters(
    *, wavelength_vacuum_mm: float, refractive_index: float
) -> tuple[float, float]:
    values = np.asarray(
        [wavelength_vacuum_mm, refractive_index], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("wavelength and refractive index must be finite and positive")
    wavelength_medium_mm = float(wavelength_vacuum_mm / refractive_index)
    return wavelength_medium_mm, float(2.0 * np.pi / wavelength_medium_mm)


def _signed_nonzero_distance(distance_mm: float) -> float:
    if not np.isfinite(distance_mm) or distance_mm == 0.0:
        raise ValueError("signed propagation distance must be finite and nonzero")
    return float(distance_mm)


def _positive_distance(distance_mm: float) -> float:
    if not np.isfinite(distance_mm) or distance_mm <= 0.0:
        raise ValueError("physical model distance must be finite and positive")
    return float(distance_mm)


def _axial_phase(k_per_mm: float, distance_mm: float) -> tuple[float, complex]:
    nominal = float(k_per_mm * distance_mm)
    reduced = float(np.remainder(nominal, 2.0 * np.pi))
    return reduced, complex(np.exp(1j * reduced))


def _quadratic_phase(
    grid: UniformGrid2D, *, k_per_mm: float, signed_distance_mm: float
) -> np.ndarray:
    distance = _signed_nonzero_distance(signed_distance_mm)
    x, y = np.meshgrid(grid.x_mm, grid.y_mm)
    return k_per_mm * (x * x + y * y) / (2.0 * distance)


@dataclass(frozen=True)
class ScaledDftResult:
    """One centered unitary DFT/IDFT on cell-energy samples."""

    cell_samples: np.ndarray
    grid: UniformGrid2D
    signed_distance_mm: float

    def __post_init__(self) -> None:
        values = _readonly_complex(self.cell_samples)
        if values.shape != (self.grid.ny, self.grid.nx):
            raise ValueError("scaled DFT samples do not match the output grid")
        object.__setattr__(self, "cell_samples", values)
        object.__setattr__(
            self,
            "signed_distance_mm",
            _signed_nonzero_distance(self.signed_distance_mm),
        )


def scaled_dft_analytic_factor(signed_distance_mm: float) -> complex:
    """Return the fixed Fresnel factor omitted by PROPER's unitary DFT."""

    distance = _signed_nonzero_distance(signed_distance_mm)
    return complex(-1j * np.sign(distance))


def scaled_dft_cell_samples(
    cell_samples: np.ndarray,
    input_grid: UniformGrid2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    signed_distance_mm: float,
) -> ScaledDftResult:
    """Apply PROPER's centered, norm-preserving signed scaled DFT.

    The analytic ``-i*sgn(d)`` Fresnel multiplier is deliberately not applied;
    callers compose it once per STW/WTS transform at the branch boundary.
    """

    if not isinstance(input_grid, UniformGrid2D):
        raise ValueError("input_grid must be a UniformGrid2D")
    values = np.asarray(cell_samples, dtype=np.complex128)
    if values.shape != (input_grid.ny, input_grid.nx):
        raise ValueError("cell-energy sample shape does not match input grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("cell-energy samples must be finite")
    wavelength_medium_mm, _ = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    distance = _signed_nonzero_distance(signed_distance_mm)
    uncentered = np.fft.ifftshift(values)
    if distance > 0.0:
        transformed = np.fft.fft2(uncentered, norm="ortho")
    else:
        transformed = np.fft.ifft2(uncentered, norm="ortho")
    centered = np.fft.fftshift(transformed)
    output_grid = UniformGrid2D.centered(
        nx=input_grid.nx,
        ny=input_grid.ny,
        dx_mm=wavelength_medium_mm
        * abs(distance)
        / (input_grid.nx * input_grid.dx_mm),
        dy_mm=wavelength_medium_mm
        * abs(distance)
        / (input_grid.ny * input_grid.dy_mm),
    )
    return ScaledDftResult(
        cell_samples=centered,
        grid=output_grid,
        signed_distance_mm=distance,
    )


def _ptp_carrier_removed_cell_samples(
    cell_samples: np.ndarray,
    grid: UniformGrid2D,
    *,
    wavelength_medium_mm: float,
    signed_distance_mm: float,
) -> np.ndarray:
    distance = _signed_nonzero_distance(signed_distance_mm)
    values = np.asarray(cell_samples, dtype=np.complex128)
    if values.shape != (grid.ny, grid.nx) or not np.all(np.isfinite(values)):
        raise ValueError("PTP cell-energy samples must match the finite grid")
    fx = np.fft.fftfreq(grid.nx, grid.dx_mm)
    fy = np.fft.fftfreq(grid.ny, grid.dy_mm)
    fx2, fy2 = np.meshgrid(fx, fy)
    transfer = np.exp(
        -1j
        * np.pi
        * wavelength_medium_mm
        * distance
        * (fx2 * fx2 + fy2 * fy2)
    )
    spectrum = np.fft.fft2(np.fft.ifftshift(values))
    return np.fft.fftshift(np.fft.ifft2(spectrum * transfer))


def propagate_ptp_fresnel(
    field: PointField2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    distance_mm: float,
) -> PointField2D:
    """Propagate point values with the same-grid periodic Fresnel kernel."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    wavelength_medium_mm, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    distance = _signed_nonzero_distance(distance_mm)
    cell_samples = np.sqrt(field.grid.pixel_area_mm2) * field.values
    propagated = _ptp_carrier_removed_cell_samples(
        cell_samples,
        field.grid,
        wavelength_medium_mm=wavelength_medium_mm,
        signed_distance_mm=distance,
    ) / np.sqrt(field.grid.pixel_area_mm2)
    _, carrier = _axial_phase(k_per_mm, distance)
    return PointField2D(propagated * carrier, field.grid)


def propagate_scaled_fresnel(
    field: PointField2D,
    target_grid: UniformGrid2D,
    *,
    wavelength_vacuum_mm: float,
    refractive_index: float,
    distance_mm: float,
    batch_size: int = 128,
) -> PointField2D:
    """Evaluate the finite-domain scaled Fresnel integral in absolute units."""

    if not isinstance(field, PointField2D):
        raise ValueError("field must be a PointField2D")
    if not isinstance(target_grid, UniformGrid2D):
        raise ValueError("target_grid must be a UniformGrid2D")
    distance = _positive_distance(distance_mm)
    wavelength_medium_mm, k_per_mm = _medium_parameters(
        wavelength_vacuum_mm=wavelength_vacuum_mm,
        refractive_index=refractive_index,
    )
    q_input = _quadratic_phase(
        field.grid, k_per_mm=k_per_mm, signed_distance_mm=distance
    )
    chirped = PointField2D(field.values * np.exp(1j * q_input), field.grid)
    frequencies_x = target_grid.x_mm / (wavelength_medium_mm * distance)
    frequencies_y = target_grid.y_mm / (wavelength_medium_mm * distance)
    spectrum = evaluate_field_fourier_czt(
        chirped,
        frequencies_x,
        frequencies_y,
        batch_size=batch_size,
    )
    q_output = _quadratic_phase(
        target_grid, k_per_mm=k_per_mm, signed_distance_mm=distance
    )
    envelope = (
        spectrum.values
        * np.exp(1j * q_output)
        / (1j * wavelength_medium_mm * distance)
    )
    _, carrier = _axial_phase(k_per_mm, distance)
    return PointField2D(envelope * carrier, target_grid)


__all__ = [
    "ScaledDftResult",
    "propagate_ptp_fresnel",
    "propagate_scaled_fresnel",
    "scaled_dft_analytic_factor",
    "scaled_dft_cell_samples",
]
